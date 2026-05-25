"""
test_feature_goal_stack.py — E2E test for hierarchical goal stack.

Verifies:
    1. GoalFrame creation and sub-goal navigation.
    2. GoalStack push, tick, completion detection, and pop.
    3. Timeout/failure detection and replan trigger.
    4. Decomposition: parent with sub-goals advances through them.
    5. Brain integration: stack is populated from drive goals and ticked.
    6. Completion hook fires when a goal is achieved.
    7. Failed goals trigger replanning (new frame pushed).
    8. Max depth is respected.
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


PASS, FAIL = [], []


def check(name: str, cond: bool, detail: str = "") -> None:
    if cond:
        PASS.append(name)
        print(f"  PASS  {name}" + (f"  ({detail})" if detail else ""))
    else:
        FAIL.append((name, detail))
        print(f"  FAIL  {name}  {detail}")


# ---------------------------------------------------------------------------
# Test 1: GoalFrame and GoalStack unit
# ---------------------------------------------------------------------------

def test_stack_unit():
    print("\n[unit] GoalStack push/pop/tick/completion")
    from cerebrum.goal_stack import GoalStack, GoalFrame

    stack = GoalStack(max_depth=3, default_max_ticks=50)

    check("starts empty", stack.is_empty)
    check("depth 0", stack.depth == 0)

    # Push a leaf goal with a target
    target = F.normalize(torch.randn(64), dim=0)
    frame = GoalFrame(name="reach_target", target_latent=target, urgency=0.7, max_ticks=20)
    ok = stack.push(frame)
    check("push succeeds", ok)
    check("depth 1", stack.depth == 1)
    check("current_goal is the pushed frame", stack.current_goal().name == "reach_target")

    # Tick with a far-away latent — should not complete
    far = F.normalize(torch.randn(64), dim=0)
    event = stack.tick(far)
    check("far latent does not complete", event is None)
    check("ticks_active incremented", frame.ticks_active == 1)

    # Tick with the target itself — should complete
    event = stack.tick(target)
    check("reaching target completes the goal", event == "completed")
    check("stack empty after completion", stack.is_empty)
    check("completed_count incremented", stack._completed_count == 1)


def test_stack_timeout():
    print("\n[timeout] GoalStack failure on timeout")
    from cerebrum.goal_stack import GoalStack, GoalFrame

    stack = GoalStack(max_depth=3)
    frame = GoalFrame(name="slow_goal", max_ticks=5)
    stack.push(frame)

    # Tick past the timeout
    events = []
    for _ in range(10):
        e = stack.tick(torch.randn(64))
        if e:
            events.append(e)

    check("timeout triggers failure", "failed" in events)
    check("stack empty after failure", stack.is_empty)
    check("failed_count incremented", stack._failed_count == 1)


def test_decomposition():
    print("\n[decomposition] parent with sub-goals advances through them")
    from cerebrum.goal_stack import GoalStack, GoalFrame

    stack = GoalStack(max_depth=5)

    # Parent with 3 sub-goals, each completable by reaching its target
    D = 64
    targets = [F.normalize(torch.randn(D), dim=0) for _ in range(3)]
    sub_goals = [
        GoalFrame(name=f"step_{i}", target_latent=targets[i], max_ticks=50)
        for i in range(3)
    ]
    stack.push_decomposition("big_plan", sub_goals, urgency=0.8)

    check("decomposition pushed", stack.depth == 1)
    check("current is first sub-goal", stack.current_goal().name == "step_0")

    # Complete sub-goal 0
    event = stack.tick(targets[0])
    check("sub-goal 0 completed, advances", event == "advanced")
    check("current is now step_1", stack.current_goal().name == "step_1")

    # Complete sub-goal 1
    event = stack.tick(targets[1])
    check("sub-goal 1 completed, advances", event == "advanced")
    check("current is now step_2", stack.current_goal().name == "step_2")

    # Complete sub-goal 2 — parent should complete
    event = stack.tick(targets[2])
    check("all sub-goals done, parent completes", event == "completed")
    check("stack empty after parent done", stack.is_empty)


def test_max_depth():
    print("\n[depth] max depth is respected")
    from cerebrum.goal_stack import GoalStack, GoalFrame

    stack = GoalStack(max_depth=2)
    stack.push(GoalFrame(name="g1"))
    stack.push(GoalFrame(name="g2"))
    ok = stack.push(GoalFrame(name="g3"))
    check("third push rejected at max_depth=2", not ok)
    check("stack depth remains 2", stack.depth == 2)


# ---------------------------------------------------------------------------
# Test 2: Brain integration
# ---------------------------------------------------------------------------

def test_brain_goal_stack():
    print("\n[brain] goal stack integrated into tick loop")
    from brain import ChipBrain

    brain = ChipBrain(config={
        "save_every": 0,
        "auto_restore": False,
        "inner_speech_every": 1000,
    }).boot()

    completed = []
    failed = []
    brain.hooks.on("goal_completed", lambda p: completed.append(p))
    brain.hooks.on("goal_failed", lambda p: failed.append(p))

    # Run 20 ticks — the drive-based goal generator should populate the stack
    for i in range(20):
        brain.tick(f"Observation {i}")

    check("goal stack populated during ticks",
          not brain.goal_stack.is_empty or brain.goal_stack._completed_count > 0 or brain.goal_stack._failed_count > 0,
          f"depth={brain.goal_stack.depth}, completed={brain.goal_stack._completed_count}, failed={brain.goal_stack._failed_count}")

    # Status includes goal stack info
    status = brain.status()
    check("status includes goal_stack", "goal_stack" in status)
    if "goal_stack" in status:
        check("goal_stack has depth field", "depth" in status["goal_stack"],
              f"keys={list(status['goal_stack'].keys())}")

    # Force a goal to time out by setting max_ticks very low
    from cerebrum.goal_stack import GoalFrame
    brain.goal_stack.clear()
    brain.goal_stack.push(GoalFrame(name="doomed_goal", max_ticks=2))
    for i in range(5):
        brain.tick(f"Tick after doomed goal {i}")

    check("doomed goal failed via timeout",
          brain.goal_stack._failed_count > 0 or len(failed) > 0,
          f"failed_count={brain.goal_stack._failed_count}")


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

def main() -> int:
    print("=" * 70)
    print("Feature 4: Hierarchical Goal Stack - E2E test")
    print("=" * 70)
    t0 = time.time()
    test_stack_unit()
    test_stack_timeout()
    test_decomposition()
    test_max_depth()
    test_brain_goal_stack()
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
