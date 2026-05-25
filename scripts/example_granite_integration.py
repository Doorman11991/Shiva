"""
example_granite_integration.py — End-to-end Granite-in-the-brain demo.

Shows three things:

1. The canonical "boot the brain" sequence with GraniteEmbedder warm-loaded
   into the thalamus once, then handed to every region that needs it.

2. The natural integration point in the hippocampus: storing a sequence of
   raw English sentences as a single episodic memory via `store_text`, and
   recalling identity context with a natural-language query.

3. The thalamus SensoryEncoder accepting a raw string in its "text"
   modality, returning a (B, 1, D) latent token ready for the transformer
   backbone.

Run from the project root:
    python scripts/example_granite_integration.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# Make the project importable when run as a script
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# 1. Boot the brain (the canonical assembly order)
# ---------------------------------------------------------------------------
print("=" * 70)
print("Booting Chip brain with Granite-125m text sensorium")
print("=" * 70)

from brainstem.device import pick_device, describe_device

device = pick_device()
print(f"[brainstem] picked device: {describe_device(device)}")

# Warm-load the granite embedder once. Every other region that wants a
# text embedding will reuse this instance via thalamus.get_embedder().
from thalamus import get_embedder

t0 = time.time()
embedder = get_embedder()
print(f"[thalamus]  GraniteEmbedder ready in {time.time() - t0:.2f}s")
print(f"            {embedder.info()}")


# ---------------------------------------------------------------------------
# 2. Sanity: similarity, batch encoding, top-k retrieval
# ---------------------------------------------------------------------------
print("\n" + "-" * 70)
print("Embedder sanity checks")
print("-" * 70)

z_a = embedder.encode("The agent considers a risky shortcut.")
z_b = embedder.encode("Should I take the dangerous path or play it safe?")
z_c = embedder.encode("The cat sat on the mat.")

print(f"single encode shape: {tuple(z_a.shape)}  (expected: (512,))")

batch = embedder.encode([
    "Curiosity pulls the agent toward the unknown.",
    "Fear holds it back from the cliff edge.",
    "It is content to rest by the campfire.",
])
print(f"batch encode shape:  {tuple(batch.shape)}  (expected: (3, 512))")

sim_ab = float(embedder.similarity(
    "The agent considers a risky shortcut.",
    "Should I take the dangerous path or play it safe?",
).item())
sim_ac = float(embedder.similarity(
    "The agent considers a risky shortcut.",
    "The cat sat on the mat.",
).item())
print(f"sim(risk-related, risk-related) = {sim_ab:+.4f}")
print(f"sim(risk-related, unrelated)    = {sim_ac:+.4f}")
assert sim_ab > sim_ac, "semantic similarity ordering broken"

candidates = [
    "I want to explore an unfamiliar room.",
    "I want to stay in my safe zone.",
    "I want to write a poem about the moon.",
]
top = embedder.most_similar("Curiosity drives me into the unknown.", candidates, top_k=2)
print("top-2 most similar to 'Curiosity drives me into the unknown.':")
for text, score in top:
    print(f"   {score:+.4f}   {text}")


# ---------------------------------------------------------------------------
# 3. Hippocampus: encode a memory directly from text
# ---------------------------------------------------------------------------
print("\n" + "-" * 70)
print("Hippocampus: text-to-episode memory storage and recall")
print("-" * 70)

from hippocampus.episodic_memory import EpisodicMemory

memory = EpisodicMemory(
    latent_dim=embedder.output_dim,
    capacity=256,
    sequence_length=4,
    narrative_window=4,
)

# A scene unfolds across 4 ticks — store it as a single episode with
# slightly positive valence. The hippocampus calls the granite embedder
# under the hood; we never touch tensors here.
memory.set_current_step(0)
memory.store_text(
    [
        "I see an unfamiliar door at the end of the corridor.",
        "I hear footsteps behind me, but no one is there.",
        "I cautiously turn the doorknob and step inside.",
        "The room smells like old books and rain.",
    ],
    valence=+0.4,
    empowerment_score=0.3,
)

memory.set_current_step(1)
memory.store_text(
    [
        "A loud crash erupts from the kitchen.",
        "Glass shatters on the tile floor.",
        "I freeze, heart pounding, every muscle tense.",
        "Slowly I peer around the corner, dreading what I might see.",
    ],
    valence=-0.7,
    empowerment_score=0.1,
)

memory.set_current_step(2)
memory.store_text(
    [
        "Sunlight filters through the kitchen window.",
        "I sip my coffee and watch the steam rise.",
        "The cat brushes against my leg.",
        "Everything feels still and ordinary.",
    ],
    valence=+0.6,
    empowerment_score=0.2,
)

print(f"stored episodes: {memory.size}")

# Recall: what does the hippocampus return when probed with a natural
# language query? The identity context blends the agent's narrative GRU
# state with the granite-encoded query.
identity = memory.query_by_text("Tell me about a tense moment in the kitchen.")
assert identity is not None
print(f"identity context shape from text query: {tuple(identity.shape)}  (expected: (1, 512))")
print(f"identity vector norm: {torch.norm(identity).item():.3f}")


# ---------------------------------------------------------------------------
# 4. Thalamus SensoryEncoder: raw string in, brain-ready token out
# ---------------------------------------------------------------------------
print("\n" + "-" * 70)
print("Thalamus SensoryEncoder: text modality auto-routes to granite")
print("-" * 70)

from thalamus.sensory_encoder import SensoryEncoder

senses = SensoryEncoder(d_model=embedder.output_dim)
print(f"registered modalities: {list(senses.encoders.keys())}")

# Single text input → sensory token ready for the transformer backbone
token = senses.encode("The world feels alive with possibility.", modality="text")
print(f"text -> sensory token shape: {tuple(token.shape)}  (expected: (1, 1, 512))")

# Multi-modal in one call: text + a 64-D proprioception vector
senses.register_modality("proprioception", input_dim=64)
multi = senses.encode_multi({
    "text": "I lean cautiously over the railing.",
    "proprioception": torch.randn(1, 64),
})
print(f"multi-modal token sequence: {tuple(multi.shape)}  (expected: (1, 2, 512))")


# ---------------------------------------------------------------------------
# 5. End-to-end: text → thalamus → cerebrum policy → action
# ---------------------------------------------------------------------------
print("\n" + "-" * 70)
print("Full pipeline: text → policy.get_action")
print("-" * 70)

from thalamus.transformer_backbone import TransformerEncoderBlock
from cerebrum.chip_policy import (
    ContinuousActor, ContinuousSACPolicy, DoubleQCritic,
)

D = embedder.output_dim
ACTION_DIM = 4

backbone = TransformerEncoderBlock(D, num_heads=8, max_seq_len=64)
policy = ContinuousSACPolicy(
    backbone=backbone,
    actor1=ContinuousActor(D, ACTION_DIM),
    actor2=ContinuousActor(D, ACTION_DIM),
    memory=memory,                                # the same hippocampus we built above
    critic=DoubleQCritic(D, ACTION_DIM),
    d_model=D,
)

# Encode a fresh observation as a 1-token sequence and feed it to the policy.
obs_token = senses.encode(
    "A door creaks open behind me. Should I look?",
    modality="text",
)  # (1, 1, D)

action, log_prob, gate = policy.get_action(obs_token)
print(f"text observation → action {tuple(action.shape)}, "
      f"|a|={action.abs().mean().item():.3f}, log_prob={log_prob.item():+.3f}, "
      f"gate={gate.item():.3f}")

print("\n" + "=" * 70)
print("All integration checks passed.")
print("=" * 70)
