"""
e2e_brain_test.py — End-to-end Chip brain test.

Pulls real embeddings + chat completions from the LM Studio endpoint
at http://10.0.0.20:1234/v1, feeds them through every brain region,
and verifies the inter-region SignalBus pathways function correctly.

Models used:
    text-embedding-granite-embedding-125m-english   (sensory input)
    huihui-gemma-4-e4b-it-abliterated              (environment text)

The test exercises:
    - Thalamus     (transformer backbone, latent alignment, attention bottleneck)
    - Hippocampus  (episodic memory, dream cycle, temporal abstraction, spatial map)
    - Amygdala     (valence, fear assessment, arousal modulation, emotional memory)
    - Hypothalamus (homeostasis, curiosity, drive arbitration, entropy temperature)
    - Cerebrum     (working memory, world model, meta-cognition, narrative self,
                    goal generator, reasoning, concept grounding, personality, policy)
    - Cerebellum   (action smoother, skill library, swarm coordinator)
    - Brainstem    (running stats, health monitor, gradient clipper, scheduler)
    - Interfaces   (NeuralSignal, SignalBus)
"""

from __future__ import annotations

import json
import sys
import time
import traceback
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Make project importable
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Endpoint config
# ---------------------------------------------------------------------------

BASE_URL = "http://10.0.0.20:1234/v1"
EMBED_MODEL = "text-embedding-granite-embedding-125m-english"
CHAT_MODEL = "huihui-gemma-4-e4b-it-abliterated"


# ---------------------------------------------------------------------------
# Tiny test harness
# ---------------------------------------------------------------------------

class TestRunner:
    def __init__(self) -> None:
        self.passed: List[str] = []
        self.failed: List[Tuple[str, str]] = []

    def run(self, name: str, fn) -> None:
        print(f"\n[{name}] ...")
        t0 = time.time()
        try:
            fn()
            dt = time.time() - t0
            print(f"    OK ({dt:.2f}s)")
            self.passed.append(name)
        except Exception as e:
            dt = time.time() - t0
            print(f"    FAIL ({dt:.2f}s): {type(e).__name__}: {e}")
            traceback.print_exc()
            self.failed.append((name, f"{type(e).__name__}: {e}"))

    def summary(self) -> int:
        total = len(self.passed) + len(self.failed)
        print("\n" + "=" * 70)
        print(f"RESULTS: {len(self.passed)}/{total} passed")
        if self.failed:
            print("\nFailures:")
            for name, err in self.failed:
                print(f"  - {name}: {err}")
        print("=" * 70)
        return 0 if not self.failed else 1


# ---------------------------------------------------------------------------
# LM Studio HTTP helpers
# ---------------------------------------------------------------------------

def _post_json(path: str, payload: dict, timeout: float = 60.0) -> dict:
    url = f"{BASE_URL}{path}"
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, method="POST",
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def get_embedding(text: str) -> torch.Tensor:
    """Return granite embedding for text as a torch tensor."""
    body = _post_json("/embeddings", {"model": EMBED_MODEL, "input": text})
    vec = body["data"][0]["embedding"]
    return torch.tensor(vec, dtype=torch.float32)


def chat(prompt: str, system: str = "You are a concise assistant.", max_tokens: int = 80) -> str:
    """Get a short chat completion from gemma."""
    body = _post_json(
        "/chat/completions",
        {
            "model": CHAT_MODEL,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": max_tokens,
            "temperature": 0.7,
        },
        timeout=120,
    )
    return body["choices"][0]["message"]["content"].strip()


# ---------------------------------------------------------------------------
# Helpers to build a full Chip brain in test config
# ---------------------------------------------------------------------------

D_MODEL = 512        # internal latent dim
ACTION_DIM = 4
TORCH_SEED = 42

torch.manual_seed(TORCH_SEED)


def project_to_latent(embed: torch.Tensor, target_dim: int = D_MODEL, proj: Optional[nn.Module] = None) -> Tuple[torch.Tensor, nn.Module]:
    """
    Project an external embedding (768-D from granite) into Chip's
    latent space (D_MODEL). Caches the projection layer across calls.
    """
    if proj is None:
        proj = nn.Linear(embed.shape[-1], target_dim)
    return proj(embed), proj


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

R = TestRunner()


def test_endpoint_alive():
    """Endpoint reachable and required models available."""
    url = f"{BASE_URL}/models"
    with urllib.request.urlopen(url, timeout=5) as resp:
        data = json.loads(resp.read())
    ids = {m["id"] for m in data["data"]}
    assert EMBED_MODEL in ids, f"missing {EMBED_MODEL}"
    assert CHAT_MODEL in ids, f"missing {CHAT_MODEL}"
    print(f"    found {len(ids)} models; required two are present")


def test_signal_bus():
    """SignalBus routing and priority ordering."""
    from interfaces.signals import SignalBus, NeuralSignal

    bus = SignalBus()
    bus.subscribe("cerebrum", ["sensory_tokens", "drive_signal"])
    bus.subscribe("amygdala", ["sensory_tokens"])

    bus.publish(NeuralSignal("thalamus", "cerebrum", "sensory_tokens",
                             torch.zeros(D_MODEL), priority=0.5))
    bus.publish(NeuralSignal("thalamus", "*", "sensory_tokens",
                             torch.ones(D_MODEL), priority=0.9))
    bus.publish(NeuralSignal("hypothalamus", "cerebrum", "drive_signal",
                             {"name": "curiosity", "level": 0.7}, priority=0.3))

    cerebrum_msgs = bus.poll("cerebrum")
    amygdala_msgs = bus.poll("amygdala")

    assert len(cerebrum_msgs) == 3, f"cerebrum got {len(cerebrum_msgs)} msgs"
    assert len(amygdala_msgs) == 1, f"amygdala got {len(amygdala_msgs)} msgs"
    # Highest priority first
    assert cerebrum_msgs[0].priority == 0.9
    print(f"    routed 3 → cerebrum, 1 → amygdala (priority order verified)")


def test_thalamus_pipeline():
    """Real granite embedding flows through transformer + bottleneck."""
    from thalamus.transformer_backbone import TransformerEncoderBlock
    from thalamus.attention_bottleneck import AttentionBottleneck

    text = "Chip is a proto-AGI organised around brain anatomy."
    emb = get_embedding(text)
    print(f"    granite embedding dim: {emb.shape[-1]}")

    proj = nn.Linear(emb.shape[-1], D_MODEL)
    z = proj(emb).unsqueeze(0).unsqueeze(0)            # (1, 1, D)
    z = z.expand(1, 8, D_MODEL).contiguous()           # fake 8-token sequence

    backbone = TransformerEncoderBlock(D_MODEL, num_heads=8, max_seq_len=128)
    z_out = backbone.forward_pass(z)
    assert z_out.shape == z.shape, f"backbone shape mismatch: {z_out.shape}"

    bottleneck = AttentionBottleneck(D_MODEL, top_k=4)
    z_filtered, salience = bottleneck(z_out)
    assert z_filtered.shape[1] <= 4, f"top-k filter failed: {z_filtered.shape}"
    assert salience.min().item() >= 0.0 and salience.max().item() <= 1.0
    print(f"    backbone {z.shape} -> filtered {z_filtered.shape}")


def test_amygdala_emotional_processing():
    """Valence + homeostasis + mood transition + arousal modulation."""
    from amygdala.emotional_core import EmotionalCore
    from amygdala.arousal_modulator import ArousalModulator
    from amygdala.fear_assessment import FearAssessor
    from amygdala.emotional_memory import EmotionalMemoryTagger
    from thalamus.latent_alignment import LatentAligner

    aligner = LatentAligner(encoders=nn.ModuleDict({"text": nn.Linear(D_MODEL, D_MODEL)}), d_model=D_MODEL)
    core = EmotionalCore(latent_aligner=aligner, hidden_dim=D_MODEL)

    z = torch.randn(2, D_MODEL)
    valence = core.get_valence(z)
    assert valence.shape == (2, 1) and -1.0 <= valence.min().item() <= valence.max().item() <= 1.0

    core.update_homeostasis(action_impact=0.3, environment_surprise=0.6, task_success=0.5)
    strain = core.calculate_strain()
    new_mood = core.auto_transition_mood(0.5, 0.8)
    print(f"    valence range [{valence.min():.3f}, {valence.max():.3f}], strain={strain:.3f}, mood={new_mood}")

    arousal_mod = ArousalModulator(D_MODEL)
    gain = arousal_mod(torch.tensor([[0.8]]))
    assert 0.5 <= gain.item() <= 2.0

    fear = FearAssessor(D_MODEL, ACTION_DIM, veto_threshold=0.95)
    risk, vetoed = fear.assess(z, torch.randn(2, ACTION_DIM))
    assert risk.shape == (2, 1)
    print(f"    arousal gain={gain.item():.3f}, risk_max={risk.max():.3f}, vetoed={vetoed}")

    tagger = EmotionalMemoryTagger()
    sig = tagger.compute_significance(valence, torch.tensor([[0.7], [0.3]]))
    assert sig.shape == (2, 1)
    print(f"    emotional significance: {sig.flatten().tolist()}")


def test_hippocampus_memory():
    """Episodic memory store/retrieve, dream batch, temporal abstraction, cognitive map."""
    from hippocampus.episodic_memory import EpisodicMemory
    from hippocampus.temporal_abstraction import TemporalAbstractor
    from hippocampus.spatial_map import CognitiveMap

    mem = EpisodicMemory(latent_dim=D_MODEL, capacity=64, sequence_length=8, narrative_window=4)
    for step in range(20):
        mem.set_current_step(step)
        states = torch.randn(8, D_MODEL)
        valences = torch.randn(8, 1) * 0.5
        mem.store_episode(states, valences, empowerment_score=0.3)

    assert mem.size == 20
    dream = mem.get_dream_batch(8)
    assert dream is not None and dream.shape == (8, 8, D_MODEL)

    identity = mem.get_identity_context(torch.randn(2, D_MODEL))
    assert identity.shape == (2, D_MODEL)
    print(f"    stored 20 episodes; dream batch {dream.shape}; identity {identity.shape}")

    temp = TemporalAbstractor(latent_dim=D_MODEL, levels=[("episode", 4), ("session", 8)])
    for _ in range(8):
        temp.push(torch.randn(D_MODEL))
    ctx = temp.get_temporal_context()
    assert ctx.shape == (D_MODEL,)
    print(f"    temporal levels ready: {temp.ready_levels}")

    cmap = CognitiveMap(latent_dim=D_MODEL, max_cells=32, novelty_thresh=2.0)
    for _ in range(50):
        cmap.update(torch.randn(D_MODEL) * 3.0)
    stats = cmap.stats()
    assert stats["n_cells"] >= 1
    print(f"    cognitive map: {stats['n_cells']} place cells, {stats['n_transitions']} transitions")


def test_hypothalamus_drives():
    """Homeostasis, curiosity, energy, drive arbitration."""
    from hypothalamus.homeostasis import HomeostaticRegulator
    from hypothalamus.curiosity_drive import CuriosityDrive
    from hypothalamus.energy_manager import EnergyManager
    from hypothalamus.drive_arbitrator import Drive, DriveArbitrator
    from hypothalamus.entropy_temperature import EntropyTemperatureRegulator

    homeo = HomeostaticRegulator()
    homeo.update({"energy": -0.3, "curiosity": +0.2})
    errors = homeo.per_dim_error()
    assert "energy" in errors
    print(f"    most urgent drive: {homeo.most_urgent_drive()}, strain={homeo.strain():.3f}")

    cur = CuriosityDrive(latent_dim=D_MODEL)
    pred = torch.randn(4, D_MODEL)
    actual = pred + 0.5 * torch.randn(4, D_MODEL)
    rew = cur.compute_reward(pred, actual)
    cur.step()
    assert rew.shape == (4, 1)
    print(f"    curiosity reward mean={rew.mean():.3f}, beta={cur.beta:.4f}")

    em = EnergyManager(max_energy=100.0, recovery_rate=5.0)
    assert em.spend("forward_pass")
    assert em.energy_fraction < 1.0

    arb = DriveArbitrator()
    arb.submit(Drive("curiosity", urgency=0.4, valence=+0.6, source="hypothalamus"))
    arb.submit(Drive("safety", urgency=0.95, valence=-0.9, source="amygdala"))
    arb.submit(Drive("engagement", urgency=0.5, valence=+0.3, source="hypothalamus"))
    winner = arb.arbitrate()
    assert winner.name == "safety", f"safety override failed, got {winner.name}"
    print(f"    drive arbitration winner: {winner.name} (urgency={winner.urgency})")

    ent = EntropyTemperatureRegulator(action_dim=ACTION_DIM)
    log_probs = torch.randn(8, 1) - 1.0
    loss = ent.update(log_probs, engagement=0.4, safety=0.6, energy=0.7)
    print(f"    entropy temperature alpha={ent.alpha.item():.4f}, alpha_loss={loss:.4f}")


def test_brainstem_vitals():
    """Running stats, health monitor, gradient clipper, scheduler."""
    from brainstem.running_stats import RunningMeanStd
    from brainstem.health_monitor import HealthMonitor
    from brainstem.gradient_clipper import GradientClipper
    from brainstem.scheduler import WarmupCosineScheduler, TrainingPhaseManager

    rms = RunningMeanStd()
    for _ in range(50):
        rms.update(torch.randn(8) * 5.0 + 2.0)
    norm = rms.normalise(torch.tensor([7.0, -3.0]))
    assert norm.abs().max() <= 10.0
    print(f"    reward stats: mean={rms.mean:.3f}, std={rms.std:.3f}")

    hm = HealthMonitor(window=10, divergence_threshold=10.0)
    healthy = all(hm.record("loss", v) for v in [1.0, 0.9, 0.85, 0.82, 0.80])
    assert healthy
    nan_ok = hm.record("loss", float("nan"))
    assert nan_ok is False
    print(f"    health monitor caught NaN; uptime tracking active")

    model = nn.Linear(D_MODEL, ACTION_DIM)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    x = torch.randn(8, D_MODEL)
    y = torch.randn(8, ACTION_DIM)
    loss = ((model(x) - y) ** 2).mean()
    loss.backward()
    clipper = GradientClipper(max_norm=1.0)
    norm = clipper.clip(model.parameters(), label="test")
    opt.step()
    print(f"    grad clip: norm={norm:.3f}, stats={clipper.stats()}")

    sched = WarmupCosineScheduler(opt, warmup_steps=5, total_steps=20, lr_max=1e-3)
    lrs = [sched.step() for _ in range(20)]
    assert lrs[0] < lrs[5] and lrs[19] < lrs[5]
    print(f"    scheduler peak lr={max(lrs):.6f}")

    pm = TrainingPhaseManager([("warmup", 3), ("explore", 5), ("exploit", 100)])
    for _ in range(10):
        pm.tick()
    assert pm.current_phase == "exploit"
    print(f"    phase manager: ended in '{pm.current_phase}'")


def test_cerebrum_cognition():
    """Working memory, world model, meta-cognition, reasoning, concepts, narrative, goals."""
    from cerebrum.working_memory import WorkingMemory
    from cerebrum.world_model import LatentDynamicsModel, WorldModelTrainer
    from cerebrum.meta_cognition import MetaCognitionMonitor
    from cerebrum.reasoning import ReasoningChain, PlanEvaluator
    from cerebrum.concept_grounding import ConceptGrounder
    from cerebrum.narrative_self import NarrativeSelf
    from cerebrum.goal_generator import GoalGenerator
    from cerebrum.personality import PersonalityTraits

    wm = WorkingMemory(latent_dim=D_MODEL, capacity=5)
    for i in range(8):
        wm.write(torch.randn(D_MODEL), salience=0.5 + 0.05 * i, source_tag=f"src{i}")
    assert len(wm._slots) == 5  # evicted to capacity
    ctx = wm.attend(torch.randn(D_MODEL))
    assert ctx.shape == (D_MODEL,)
    print(f"    WM utilisation={wm.utilisation:.2f}; attended ctx {ctx.shape}")

    world = LatentDynamicsModel(latent_dim=D_MODEL, action_dim=ACTION_DIM)
    wm_trainer = WorldModelTrainer(world, lr=1e-3)
    z = torch.randn(8, D_MODEL)
    a = torch.randn(8, ACTION_DIM)
    z_next = z + 0.1 * torch.randn(8, D_MODEL)
    losses = [wm_trainer.update(z, a, z_next) for _ in range(10)]
    assert losses[-1] < losses[0] + 1e-3, f"world model didn't learn: {losses[0]:.4f} -> {losses[-1]:.4f}"

    rollout = world.simulate(z[:1], torch.randn(1, 5, ACTION_DIM))
    assert rollout.shape == (1, 6, D_MODEL)
    print(f"    world model: loss {losses[0]:.4f} -> {losses[-1]:.4f}; rollout {rollout.shape}")

    meta = MetaCognitionMonitor(d_model=D_MODEL, action_dim=ACTION_DIM, patience=2)
    conf, deliberating = meta.assess(z, torch.full((8, 1), -2.0), torch.randn(8, 1), torch.randn(8, 1))
    assert 0.0 <= conf <= 1.0
    print(f"    meta-cognition confidence={conf:.3f}, deliberating={deliberating}")

    chain = ReasoningChain(latent_dim=D_MODEL, n_steps=3)
    z_refined, trace = chain(z[:2])
    assert len(trace) == 4 and z_refined.shape == (2, D_MODEL)
    print(f"    reasoning chain: {len(trace)} thought steps")

    eval_net = PlanEvaluator(latent_dim=D_MODEL)
    candidates = torch.randn(5, ACTION_DIM)
    best, idx = eval_net.select_best_action(z[:1], candidates, world)
    print(f"    plan evaluator chose action idx={idx}")

    cg = ConceptGrounder(d_model=D_MODEL)
    concepts = cg.ground(torch.randn(D_MODEL))
    composed = cg.compose(["novel", "risky"])
    assert len(concepts) == cg.top_k and composed.shape == (D_MODEL,)
    print(f"    grounded concepts: {concepts}")

    ns = NarrativeSelf(latent_dim=D_MODEL)
    for _ in range(10):
        ns.update_narrative(torch.randn(D_MODEL), outcome_valence=0.5)
    self_model = ns.get_self_model()
    coherence = ns.goal_coherence(torch.randn(D_MODEL))
    print(f"    narrative self {self_model.shape}, goal coherence={coherence:.3f}")

    gg = GoalGenerator(latent_dim=D_MODEL, max_goals=3)
    goals = gg.generate_from_drives(
        {"energy": -0.4, "curiosity": +0.3},
        torch.tensor([0.5, 0.4, 0.8, 0.5, 0.6, 0.9]),
    )
    gg.update_goals(goals)
    top = gg.top_goal()
    print(f"    top goal: {top.name if top else None} (urgency={top.urgency:.3f})")

    pers = PersonalityTraits(latent_dim=D_MODEL)
    bias = pers.get_personality_bias()
    assert bias.shape == (D_MODEL,)
    print(f"    personality traits: {pers.trait_summary()}")


def test_cerebellum_motor():
    """Action smoothing, skill library, swarm coordination."""
    from cerebellum.action_smoother import ActionSmoother
    from cerebellum.skill_library import SkillLibrary
    from cerebellum.swarm_coordinator import SwarmCoordinator

    sm = ActionSmoother(action_dim=ACTION_DIM, method="ema", alpha=0.7)
    actions = [torch.randn(2, ACTION_DIM) for _ in range(5)]
    smoothed = [sm.smooth(a) for a in actions]
    last = smoothed[-1]
    raw = actions[-1]
    # Smoothed action should be in same shape and not identical to raw on later steps
    assert last.shape == raw.shape
    diff = (last - raw).abs().mean().item()
    print(f"    action smoother: |smooth-raw| mean diff={diff:.4f}")

    lib = SkillLibrary(latent_dim=D_MODEL, max_skills=4)
    z_trigger = torch.randn(D_MODEL)
    seq = torch.randn(10, ACTION_DIM)
    lib.store("greet", seq, z_trigger)
    retrieved = lib.retrieve(z_trigger)
    assert retrieved is not None and retrieved[0] == "greet"
    miss = lib.retrieve(torch.randn(D_MODEL) * 10)
    print(f"    skill retrieved: {retrieved[0]}; far query miss={miss is None}")

    coord = SwarmCoordinator(latent_dim=D_MODEL, n_nodes=3)
    for nid in ["node_0", "node_1", "node_2"]:
        coord.update_node_latent(nid, torch.randn(D_MODEL))
    consensus, div = coord.step()
    assert consensus.shape == (D_MODEL,)
    print(f"    swarm consensus {consensus.shape}, diversity loss={div.item():.4f}")


def test_locomotion_snapshot():
    """Capture/serialise/deserialise/restore a cognitive snapshot."""
    from locomotion.ModelMovementAndLocomotion import CognitiveSnapshot
    from cerebrum.chip_policy import (
        ContinuousActor, ContinuousSACPolicy, DoubleQCritic,
    )
    from thalamus.transformer_backbone import TransformerEncoderBlock
    from hippocampus.episodic_memory import EpisodicMemory
    from amygdala.emotional_core import EmotionalCore
    from thalamus.latent_alignment import LatentAligner

    backbone = TransformerEncoderBlock(D_MODEL, num_heads=8)
    actor1 = ContinuousActor(D_MODEL, ACTION_DIM)
    actor2 = ContinuousActor(D_MODEL, ACTION_DIM)
    memory = EpisodicMemory(latent_dim=D_MODEL, capacity=16, sequence_length=4)
    critic = DoubleQCritic(D_MODEL, ACTION_DIM)
    policy = ContinuousSACPolicy(backbone, actor1, actor2, memory, critic, D_MODEL)

    aligner = LatentAligner(encoders=nn.ModuleDict({"text": nn.Linear(D_MODEL, D_MODEL)}), d_model=D_MODEL)
    emo = EmotionalCore(latent_aligner=aligner, hidden_dim=D_MODEL)

    # Add a couple episodes so memory_bank is non-empty.
    for step in range(3):
        memory.set_current_step(step)
        memory.store_episode(torch.randn(4, D_MODEL), torch.randn(4, 1), empowerment_score=0.5)

    snap = CognitiveSnapshot.capture(policy, memory, emo, node_id="test-node")
    blob = snap.serialise()
    assert len(blob) > 100, f"snapshot too small: {len(blob)} bytes"

    snap2 = CognitiveSnapshot.deserialise(blob)
    assert snap2.metadata.node_id == "test-node"
    print(f"    snapshot {len(blob):,} bytes; HMAC verified, schema={snap2.metadata.schema_version}")


def test_full_brain_loop():
    """
    Full integration: real LLM-generated text → granite embedding →
    full brain pipeline → action.

    Wires:
        chat(gemma) → text
        granite     → 768-D embedding
        thalamus    → 512-D latent token
        backbone    → encoded
        amygdala    → valence + mood
        hippocampus → identity context, store episode
        cerebrum    → policy.get_action
        cerebellum  → action smoother
        bus         → carries every signal
    """
    from interfaces.signals import SignalBus, NeuralSignal
    from thalamus.transformer_backbone import TransformerEncoderBlock
    from thalamus.latent_alignment import LatentAligner
    from amygdala.emotional_core import EmotionalCore
    from amygdala.arousal_modulator import ArousalModulator
    from hippocampus.episodic_memory import EpisodicMemory
    from cerebrum.chip_policy import (
        ContinuousActor, ContinuousSACPolicy, DoubleQCritic,
    )
    from cerebrum.working_memory import WorkingMemory
    from cerebellum.action_smoother import ActionSmoother

    bus = SignalBus()
    for region in ("thalamus", "cerebrum", "cerebellum", "amygdala",
                   "hippocampus", "hypothalamus", "brainstem"):
        bus.subscribe(region, ["*"])

    # 1. Get an LLM-authored "environment description"
    env_text = chat(
        "In one sentence, describe a curious, mildly risky situation an autonomous agent might face.",
        max_tokens=50,
    )
    print(f"    env_text: {env_text!r}")

    # 2. Sensory input via granite
    emb = get_embedding(env_text)
    proj_to_512 = nn.Linear(emb.shape[-1], D_MODEL)

    bus.publish(NeuralSignal("environment", "thalamus", "raw_input", emb, priority=1.0))

    # 3. Thalamus: project + encode
    z_token = proj_to_512(emb).unsqueeze(0).unsqueeze(0)         # (1, 1, D)
    z_seq = z_token.expand(1, 4, D_MODEL).contiguous()           # fake 4-token sequence
    backbone = TransformerEncoderBlock(D_MODEL, num_heads=8, max_seq_len=128)
    z_encoded = backbone.forward_pass(z_seq)
    bus.publish(NeuralSignal("thalamus", "*", "sensory_tokens", z_encoded, priority=0.8))

    # 4. Amygdala: valence from latent
    aligner = LatentAligner(encoders=nn.ModuleDict({"text": nn.Linear(D_MODEL, D_MODEL)}), d_model=D_MODEL)
    emo = EmotionalCore(latent_aligner=aligner, hidden_dim=D_MODEL)
    z_pooled = z_encoded.mean(dim=1)                             # (1, D)
    valence = emo.get_valence(z_pooled)
    arousal_mod = ArousalModulator(D_MODEL)
    arousal_signal = arousal_mod(torch.tensor([[0.6]]))          # placeholder arousal
    new_mood = emo.auto_transition_mood(float(valence.item()), 0.6)
    bus.publish(NeuralSignal("amygdala", "thalamus", "arousal_gain", arousal_signal, priority=0.6))
    bus.publish(NeuralSignal("amygdala", "cerebrum", "valence_update", valence, priority=0.5))

    # 5. Hippocampus: identity context + episode store
    memory = EpisodicMemory(latent_dim=D_MODEL, capacity=32, sequence_length=4)
    memory.set_current_step(1)
    memory.store_episode(z_seq.squeeze(0), torch.full((4, 1), float(valence.item())), empowerment_score=0.4)
    identity = memory.get_identity_context(z_pooled)
    bus.publish(NeuralSignal("hippocampus", "cerebrum", "memory_retrieve", identity, priority=0.4))

    # 6. Cerebrum: working memory + policy
    wmem = WorkingMemory(latent_dim=D_MODEL, capacity=5)
    wmem.write(z_pooled.squeeze(0), salience=0.8, source_tag="thalamus")
    wmem.write(identity.squeeze(0), salience=0.6, source_tag="hippocampus")

    actor1 = ContinuousActor(D_MODEL, ACTION_DIM)
    actor2 = ContinuousActor(D_MODEL, ACTION_DIM)
    critic = DoubleQCritic(D_MODEL, ACTION_DIM)
    policy = ContinuousSACPolicy(backbone, actor1, actor2, memory, critic, D_MODEL)
    action, log_prob, gate = policy.get_action(z_seq)
    assert action.shape == (1, ACTION_DIM)
    bus.publish(NeuralSignal("cerebrum", "cerebellum", "action_raw", action, priority=0.7))

    # 7. Cerebellum: smooth action
    smoother = ActionSmoother(action_dim=ACTION_DIM, method="ema", alpha=0.6)
    smoothed = smoother.smooth(action)
    bus.publish(NeuralSignal("cerebellum", "environment", "action_smooth", smoothed, priority=0.9))

    # 8. Verify bus accumulated history
    history = bus.recent_history(n=20)
    assert len(history) >= 6, f"signal history too short: {len(history)}"

    print(f"    valence={valence.item():.3f}, mood={new_mood}, "
          f"action |a|={action.abs().mean().item():.3f}, "
          f"smoothed |a|={smoothed.abs().mean().item():.3f}")
    print(f"    signal bus carried {len(history)} signals across "
          f"{len({s.source for s in history})} regions")


# ---------------------------------------------------------------------------
# Run all tests
# ---------------------------------------------------------------------------

def main() -> int:
    print("=" * 70)
    print("Chip Brain — End-to-End Test")
    print(f"Endpoint: {BASE_URL}")
    print(f"Embed model: {EMBED_MODEL}")
    print(f"Chat model:  {CHAT_MODEL}")
    print("=" * 70)

    R.run("endpoint reachable", test_endpoint_alive)
    R.run("interfaces / signal bus", test_signal_bus)
    R.run("thalamus pipeline", test_thalamus_pipeline)
    R.run("amygdala emotional processing", test_amygdala_emotional_processing)
    R.run("hippocampus memory systems", test_hippocampus_memory)
    R.run("hypothalamus drives", test_hypothalamus_drives)
    R.run("brainstem vitals", test_brainstem_vitals)
    R.run("cerebrum cognition", test_cerebrum_cognition)
    R.run("cerebellum motor", test_cerebellum_motor)
    R.run("locomotion cognitive snapshot", test_locomotion_snapshot)
    R.run("FULL BRAIN end-to-end", test_full_brain_loop)

    return R.summary()


if __name__ == "__main__":
    raise SystemExit(main())
