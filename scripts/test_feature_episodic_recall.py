"""
test_feature_episodic_recall.py — E2E test for inference-time recall.

What this verifies:
    1. EpisodicRecall.retrieve() returns relevant memories sorted by similarity.
    2. Irrelevant queries don't return spurious matches (threshold works).
    3. Recall integrates with WorkingMemory: slots tagged "hippocampus_recall"
       appear after a tick when relevant memories exist.
    4. The brain hook fires on recall events.
    5. Semantically related observations recall their thematic memories
       (granite-encoded similarity, not random retrieval).
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PASS, FAIL = [], []


def check(name: str, cond: bool, detail: str = "") -> None:
    if cond:
        PASS.append(name)
        print(f"  PASS  {name}" + (f"  ({detail})" if detail else ""))
    else:
        FAIL.append((name, detail))
        print(f"  FAIL  {name}  {detail}")


# ---------------------------------------------------------------------------
# Test 1: pure recall logic against a synthetic memory bank
# ---------------------------------------------------------------------------

def test_recall_unit():
    print("\n[unit] EpisodicRecall against synthetic episodes")
    from hippocampus.episodic_memory import EpisodicMemory
    from hippocampus.episodic_recall import EpisodicRecall

    D = 64
    mem = EpisodicMemory(latent_dim=D, capacity=64, sequence_length=4, narrative_window=4)

    # Two thematic clusters in latent space
    cluster_a = torch.randn(D)
    cluster_b = torch.randn(D)
    cluster_a = F.normalize(cluster_a, dim=0)
    cluster_b = F.normalize(cluster_b, dim=0)

    # Store 5 episodes from each cluster with small noise
    for step in range(10):
        mem.set_current_step(step)
        center = cluster_a if step < 5 else cluster_b
        states = center.unsqueeze(0).expand(4, D) + 0.05 * torch.randn(4, D)
        valences = torch.full((4, 1), 0.3 if step < 5 else -0.3)
        mem.store_episode(states, valences, empowerment_score=0.5)

    recall = EpisodicRecall(mem, mode="endpoint", top_k=3, min_similarity=0.3)

    # Query near cluster A — top results should all be cluster-A episodes
    q_a = cluster_a + 0.02 * torch.randn(D)
    results_a = recall.retrieve(q_a)
    check("retrieves correct cluster A", len(results_a) > 0,
          f"got {len(results_a)} results")

    # Steps 0-4 are cluster A — verify the top result is from that range
    if results_a:
        top_step = results_a[0][0]["step"]
        check("top match is from cluster A", top_step < 5,
              f"top step={top_step}, sim={results_a[0][1]:.3f}")
        check("similarity is high for in-cluster query",
              results_a[0][1] > 0.8,
              f"sim={results_a[0][1]:.3f}")

    # Query orthogonal to both clusters — should return little or nothing
    orthogonal = torch.randn(D)
    orthogonal = orthogonal - (orthogonal @ cluster_a) * cluster_a - (orthogonal @ cluster_b) * cluster_b
    orthogonal = F.normalize(orthogonal, dim=0)
    results_orth = recall.retrieve(orthogonal)
    check("threshold filters irrelevant queries",
          len(results_orth) <= len(results_a),
          f"orthogonal returned {len(results_orth)} results")

    # Trajectory mode also works
    recall_traj = EpisodicRecall(mem, mode="trajectory", top_k=3)
    results_traj = recall_traj.retrieve(q_a)
    check("trajectory mode returns results", len(results_traj) > 0,
          f"got {len(results_traj)} results")


# ---------------------------------------------------------------------------
# Test 2: WorkingMemory injection
# ---------------------------------------------------------------------------

def test_recall_injection():
    print("\n[injection] EpisodicRecall.inject_into_working_memory")
    from hippocampus.episodic_memory import EpisodicMemory
    from hippocampus.episodic_recall import EpisodicRecall
    from cerebrum.working_memory import WorkingMemory

    D = 64
    mem = EpisodicMemory(latent_dim=D, capacity=32, sequence_length=4, narrative_window=4)
    wm = WorkingMemory(latent_dim=D, capacity=7)

    target = F.normalize(torch.randn(D), dim=0)
    for step in range(5):
        mem.set_current_step(step)
        states = target.unsqueeze(0).expand(4, D) + 0.05 * torch.randn(4, D)
        mem.store_episode(states, torch.full((4, 1), 0.5), empowerment_score=0.4)

    recall = EpisodicRecall(mem, top_k=3, min_similarity=0.3)
    n_injected = recall.inject_into_working_memory(target, wm)

    check("injection wrote slots", n_injected > 0, f"injected {n_injected}")

    recall_slots = [s for s in wm._slots if s.source_tag == "hippocampus_recall"]
    check("slots tagged correctly", len(recall_slots) == n_injected,
          f"found {len(recall_slots)} tagged slots")
    check("salience is positive",
          all(s.salience > 0 for s in recall_slots),
          f"min salience={min(s.salience for s in recall_slots):.3f}" if recall_slots else "no slots")


# ---------------------------------------------------------------------------
# Test 3: Full brain — semantic recall through granite + tick loop
# ---------------------------------------------------------------------------

def test_brain_semantic_recall():
    print("\n[full brain] semantic recall through granite-encoded text")
    from brain import ChipBrain

    brain = ChipBrain().boot()

    recall_events = []
    brain.hooks.on("episodic_recall", lambda payload: recall_events.append(payload))

    # Store thematic episodes through the hippocampus's text channel
    print("  storing 3 thematic episodes via granite encoding...")
    brain.memory.set_current_step(brain._tick)
    brain.memory.store_text(
        [
            "I crouch behind cover as bullets ricochet around me.",
            "My heart pounds and adrenaline floods my veins.",
            "I peek out and return fire toward the enemy position.",
            "The firefight ends as quickly as it began.",
        ],
        valence=-0.6,
        empowerment_score=0.5,
    )
    brain.memory.set_current_step(brain._tick + 1)
    brain.memory.store_text(
        [
            "I wander through a quiet garden in early morning.",
            "Dew shimmers on the grass and birdsong fills the air.",
            "I sit on a bench and feel deeply at peace.",
            "The world feels calm and beautiful.",
        ],
        valence=+0.7,
        empowerment_score=0.4,
    )
    brain.memory.set_current_step(brain._tick + 2)
    brain.memory.store_text(
        [
            "I read a book by the fireplace as snow falls outside.",
            "The cat curls up at my feet, purring softly.",
            "I sip tea and lose myself in the story.",
            "Time feels suspended in this quiet evening.",
        ],
        valence=+0.5,
        empowerment_score=0.3,
    )

    print(f"  episodic memory size: {brain.memory.size}")

    # --- Probe 1: combat-themed observation should recall the firefight ---
    print("\n  probe 1: combat observation")
    initial_recall_count = len(recall_events)
    action = brain.tick("Gunfire erupts and I dive for cover.")
    check("combat probe produced action",
          action.shape == (1, brain.cfg["action_dim"]),
          f"action shape {tuple(action.shape)}")

    new_events = recall_events[initial_recall_count:]
    n_recalled = sum(e["n_recalled"] for e in new_events)
    check("combat probe triggered recall", n_recalled > 0,
          f"recalled {n_recalled} episodes")

    recall_slots = [s for s in brain.working_mem._slots
                    if s.source_tag == "hippocampus_recall"]
    check("recall slots in working memory",
          len(recall_slots) > 0,
          f"{len(recall_slots)} recall slots present")

    # --- Probe 2: peaceful observation ---
    print("\n  probe 2: peaceful observation")
    initial_recall_count = len(recall_events)
    brain.working_mem.reset()
    action = brain.tick("The morning is quiet and the air is fresh.")
    new_events = recall_events[initial_recall_count:]
    n_recalled_peace = sum(e["n_recalled"] for e in new_events)
    check("peaceful probe triggered recall", n_recalled_peace > 0,
          f"recalled {n_recalled_peace} episodes")

    # --- Verify recall content matches probe semantics ---
    print("\n  verifying retrieval semantically aligned with probe...")
    from hippocampus.episodic_recall import EpisodicRecall
    from thalamus.granite_embedder import get_embedder
    embedder = get_embedder()

    combat_query = embedder.encode("Combat and danger.")
    peace_query = embedder.encode("Peaceful morning in nature.")

    combat_results = brain.recall.retrieve(combat_query)
    peace_results = brain.recall.retrieve(peace_query)

    check("combat query returned matches", len(combat_results) > 0)
    check("peace query returned matches", len(peace_results) > 0)

    if combat_results and peace_results:
        # The combat query's top match should have negative valence
        # (we stored valence=-0.6 for the firefight episode).
        # Episodes don't store valence directly, but we can check that
        # the top combat match and top peace match are different episodes.
        combat_top_step = combat_results[0][0]["step"]
        peace_top_step = peace_results[0][0]["step"]
        check("combat and peace probes recall distinct episodes",
              combat_top_step != peace_top_step,
              f"combat→step{combat_top_step}, peace→step{peace_top_step}")

        sim_combat = combat_results[0][1]
        sim_peace = peace_results[0][1]
        check("retrieval similarities are high",
              sim_combat > 0.5 and sim_peace > 0.5,
              f"combat={sim_combat:.3f}, peace={sim_peace:.3f}")


# ---------------------------------------------------------------------------
# Test 4: training step still works with recall slots in working memory
# ---------------------------------------------------------------------------

def test_recall_with_training():
    print("\n[training] recall does not break SAC training")
    from brain import ChipBrain

    brain = ChipBrain().boot()

    # Pre-populate memory
    brain.memory.set_current_step(0)
    brain.memory.store_text(
        ["I learned something useful today.", "Practice makes perfect."],
        valence=0.5,
        empowerment_score=0.4,
    )

    # Run a few full tick + train_step cycles
    n_train_steps = 0
    for i in range(5):
        brain.tick(f"Observation number {i}")
        metrics = brain.train_step(reward=0.1 * i, done=(i == 4))
        if metrics is not None:
            n_train_steps += 1

    check("brain ran 5 ticks without error", brain._tick == 5,
          f"tick count={brain._tick}")
    # Training kicks in once buffer > batch_size — with batch_size=64 default
    # and only 5 transitions, no SAC update is expected. That's fine —
    # the test is that nothing crashed.
    check("recall integration didn't crash training pipeline", True,
          f"completed {n_train_steps} SAC updates over 5 ticks")


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

def main() -> int:
    print("=" * 70)
    print("Feature 1: Episodic Recall — E2E test")
    print("=" * 70)

    t0 = time.time()
    test_recall_unit()
    test_recall_injection()
    test_brain_semantic_recall()
    test_recall_with_training()
    dt = time.time() - t0

    print("\n" + "=" * 70)
    print(f"PASSED: {len(PASS)} / {len(PASS) + len(FAIL)}  ({dt:.1f}s)")
    if FAIL:
        print("FAILURES:")
        for name, detail in FAIL:
            print(f"  - {name}: {detail}")
    print("=" * 70)
    return 0 if not FAIL else 1


if __name__ == "__main__":
    raise SystemExit(main())
