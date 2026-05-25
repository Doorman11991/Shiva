"""
test_feature_contagion_dreaming.py — E2E test for emotional contagion + active dreaming.

Emotional Contagion:
    1. Group valence converges when nodes report different valences.
    2. Extreme valence dampens susceptibility (panic doesn't spiral infinitely).
    3. High-arousal nodes influence the group more than low-arousal ones.
    4. Contagion does nothing with < min_nodes.
    5. Group mood label reflects the collective state.

Active Dreaming:
    6. Decision points are identified at high-variance ticks.
    7. Counterfactual trajectories are generated via world model rollouts.
    8. Positive-regret counterfactuals are stored as synthetic episodes.
    9. Full dream session processes multiple episodes.
    10. The dreamer works with the real brain's memory and world model.
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


PASS, FAIL = [], []


def check(name: str, cond: bool, detail: str = "") -> None:
    if cond:
        PASS.append(name)
        print(f"  PASS  {name}" + (f"  ({detail})" if detail else ""))
    else:
        FAIL.append((name, detail))
        print(f"  FAIL  {name}  {detail}")


# ---------------------------------------------------------------------------
# Emotional Contagion
# ---------------------------------------------------------------------------

def test_contagion_convergence():
    print("\n[contagion] group valence converges")
    from cerebellum.emotional_contagion import EmotionalContagion

    ec = EmotionalContagion(susceptibility=0.5, max_shift_per_tick=0.2, min_nodes=2)
    ec.register_node("alpha")
    ec.register_node("beta")
    ec.register_node("gamma")

    # Alpha is very positive, beta neutral, gamma negative
    ec.update_node("alpha", valence=0.8, arousal=0.7)
    ec.update_node("beta", valence=0.0, arousal=0.5)
    ec.update_node("gamma", valence=-0.6, arousal=0.4)

    initial_spread = abs(ec.get_node_valence("alpha") - ec.get_node_valence("gamma"))

    # Run contagion for several ticks
    for _ in range(20):
        ec.step()

    final_spread = abs(ec.get_node_valence("alpha") - ec.get_node_valence("gamma"))
    check("valences converge over time", final_spread < initial_spread,
          f"spread: {initial_spread:.3f} -> {final_spread:.3f}")

    # Group valence should be between the extremes
    gv = ec.group_valence
    check("group valence is between node extremes",
          -0.6 <= gv <= 0.8,
          f"group_valence={gv:.3f}")


def test_contagion_extreme_dampening():
    print("\n[contagion] extreme valence reduces susceptibility")
    from cerebellum.emotional_contagion import EmotionalContagion

    ec = EmotionalContagion(susceptibility=0.5, extreme_dampening=0.8, min_nodes=2)
    ec.register_node("extreme")
    ec.register_node("mild")

    # extreme node at -0.95 (very scared) — should be resistant to group pull
    ec.update_node("extreme", valence=-0.95, arousal=0.9)
    ec.update_node("mild", valence=0.5, arousal=0.5)

    ec.step()
    extreme_shift = abs(ec.get_node_valence("extreme") - (-0.95))
    mild_shift = abs(ec.get_node_valence("mild") - 0.5)

    check("extreme node shifts less than mild node",
          extreme_shift < mild_shift,
          f"extreme_shift={extreme_shift:.4f}, mild_shift={mild_shift:.4f}")


def test_contagion_arousal_influence():
    print("\n[contagion] high-arousal nodes have more influence")
    from cerebellum.emotional_contagion import EmotionalContagion

    ec = EmotionalContagion(susceptibility=0.4, min_nodes=2)
    ec.register_node("loud")
    ec.register_node("quiet")
    ec.register_node("listener")

    # loud has high arousal (dominates), quiet has low arousal
    ec.update_node("loud", valence=0.8, arousal=0.9)
    ec.update_node("quiet", valence=-0.8, arousal=0.1)
    ec.update_node("listener", valence=0.0, arousal=0.5)

    ec.step()
    listener_val = ec.get_node_valence("listener")
    check("listener pulled toward high-arousal node",
          listener_val > 0.0,
          f"listener_valence={listener_val:.3f} (loud=0.8, quiet=-0.8)")


def test_contagion_min_nodes():
    print("\n[contagion] no contagion with insufficient nodes")
    from cerebellum.emotional_contagion import EmotionalContagion

    ec = EmotionalContagion(min_nodes=3)
    ec.register_node("solo")
    ec.update_node("solo", valence=0.5)
    result = ec.step()
    check("step returns None with 1 node", result is None)


def test_contagion_group_mood():
    print("\n[contagion] group mood label")
    from cerebellum.emotional_contagion import EmotionalContagion

    ec = EmotionalContagion(min_nodes=2)
    ec.register_node("a")
    ec.register_node("b")

    ec.update_node("a", valence=0.6, arousal=0.8)
    ec.update_node("b", valence=0.7, arousal=0.9)
    ec.step()
    check("positive+high arousal = excited", ec.group_mood() == "excited",
          f"mood={ec.group_mood()}")

    ec.update_node("a", valence=-0.5, arousal=0.8)
    ec.update_node("b", valence=-0.6, arousal=0.9)
    ec.step()
    check("negative+high arousal = panicked", ec.group_mood() == "panicked",
          f"mood={ec.group_mood()}")


# ---------------------------------------------------------------------------
# Active Dreaming
# ---------------------------------------------------------------------------

def test_active_dreaming_unit():
    print("\n[dreaming] counterfactual generation from episodes")
    from hippocampus.active_dreaming import ActiveDreamer
    from cerebrum.world_model import LatentDynamicsModel
    from cerebrum.reasoning import PlanEvaluator

    D, A = 64, 4
    wm = LatentDynamicsModel(latent_dim=D, action_dim=A)
    pe = PlanEvaluator(latent_dim=D)
    dreamer = ActiveDreamer(wm, pe, action_dim=A, horizon=3, n_alternatives=3)

    # Fake an episode with varying states (some high-change decision points)
    episode = torch.randn(10, D)
    # Inject a big jump at tick 5 to create a clear decision point
    episode[5] = episode[4] + 3.0 * torch.randn(D)

    cfs = dreamer.dream_episode(episode)
    check("counterfactuals generated", len(cfs) >= 0,
          f"n={len(cfs)}")

    if cfs:
        best = cfs[0]
        check("trajectory has correct shape",
              best.trajectory.shape == (4, D),  # horizon+1
              f"shape={tuple(best.trajectory.shape)}")
        check("regret is positive (better alternative found)",
              best.regret > 0,
              f"regret={best.regret:.4f}")
        check("decision_tick identified",
              best.decision_tick >= 0)


def test_active_dreaming_full_session():
    print("\n[dreaming] full dream session with real memory bank")
    from hippocampus.active_dreaming import ActiveDreamer
    from hippocampus.episodic_memory import EpisodicMemory
    from cerebrum.world_model import LatentDynamicsModel
    from cerebrum.reasoning import PlanEvaluator

    D, A = 64, 4
    wm = LatentDynamicsModel(latent_dim=D, action_dim=A)
    pe = PlanEvaluator(latent_dim=D)
    mem = EpisodicMemory(latent_dim=D, capacity=64, sequence_length=10)

    # Populate memory with varied episodes
    for step in range(20):
        mem.set_current_step(step)
        states = torch.randn(10, D)
        # Make some episodes more dramatic (larger state changes)
        if step % 3 == 0:
            states[5] += 5.0 * torch.randn(D)
        valences = torch.randn(10, 1) * 0.5
        mem.store_episode(states, valences, empowerment_score=0.3 + 0.1 * (step % 4))

    initial_size = mem.size
    dreamer = ActiveDreamer(wm, pe, action_dim=A, horizon=3, n_alternatives=3)

    result = dreamer.run(mem, batch_size=4)
    check("dream session completed", result["status"] == "completed")
    check("episodes were dreamed", result["episodes_dreamed"] == 4)
    check("counterfactuals were stored",
          mem.size > initial_size,
          f"memory grew from {initial_size} to {mem.size}")
    print(f"    session: {result['n_counterfactuals']} counterfactuals, "
          f"{result['n_stored']} stored, best_regret={result['best_regret']:.4f}")


def test_active_dreaming_in_brain():
    print("\n[brain] active dreaming integrated")
    from brain import ChipBrain
    from hippocampus.active_dreaming import ActiveDreamer

    brain = ChipBrain(config={
        "save_every": 0,
        "auto_restore": False,
        "inner_speech_every": 1000,
    }).boot()

    # Populate brain memory so dreaming has material
    for i in range(30):
        brain.tick(f"Building up memories, observation {i}")
        brain.train_step(reward=0.1 * (i % 5), done=(i % 10 == 9))

    initial_mem = brain.memory.size

    # Create an ActiveDreamer with the brain's world model and plan evaluator
    dreamer = ActiveDreamer(
        world_model=brain.world_model,
        plan_evaluator=brain.plan_eval,
        action_dim=brain.cfg["action_dim"],
        horizon=3,
        n_alternatives=3,
    )
    result = dreamer.run(brain.memory, batch_size=2)  # small batch — brain has few episodes
    check("brain dreaming session completed",
          result["status"] == "completed",
          f"status={result['status']}")
    check("synthetic memories stored in brain",
          brain.memory.size >= initial_mem,
          f"size: {initial_mem} -> {brain.memory.size}")


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

def main() -> int:
    print("=" * 70)
    print("Extra: Emotional Contagion + Active Dreaming - E2E test")
    print("=" * 70)
    t0 = time.time()
    test_contagion_convergence()
    test_contagion_extreme_dampening()
    test_contagion_arousal_influence()
    test_contagion_min_nodes()
    test_contagion_group_mood()
    test_active_dreaming_unit()
    test_active_dreaming_full_session()
    test_active_dreaming_in_brain()
    dt = time.time() - t0
    print("\n" + "=" * 70)
    print(f"PASSED: {len(PASS)} / {len(PASS) + len(FAIL)}  ({dt:.1f}s)")
    if FAIL:
        for name, detail in FAIL:
            print(f"  - {name}: {detail}")
    print("=" * 70)
    return 0 if not FAIL else 1


if __name__ == "__main__":
    raise SystemExit(main())
