"""
test_feature_persistence.py — E2E test for Cryostasis autosave/restore.

What this verifies:
    1. Cryostasis writes a snapshot on demand and the file is non-empty.
    2. Atomic write: no .tmp file lingers after a successful save.
    3. Rolling backups accumulate and rotate (oldest evicted).
    4. Auto-save fires on the configured interval inside the brain tick loop.
    5. A second brain booting from the same state_dir auto-restores.
    6. Mood, episodic memory size, and homeostasis survive a save/restore.
    7. HMAC tampering causes restore to fail safely.
    8. Corrupt latest.snap falls back to a rolling backup automatically.
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
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
# Test 1: manual save + atomic write
# ---------------------------------------------------------------------------

def test_manual_save():
    print("\n[manual] save / atomic write / rolling rotation")
    state_dir = tempfile.mkdtemp(prefix="chip_test_")
    try:
        from brain import ChipBrain
        brain = ChipBrain(config={
            "state_dir": state_dir,
            "save_every": 0,           # manual only
            "auto_restore": False,
        }).boot()

        ok = brain.save()
        check("manual save returns True", ok)

        latest = Path(state_dir) / "latest.snap"
        check("latest.snap created", latest.exists())
        check("latest.snap non-empty", latest.stat().st_size > 1000,
              f"{latest.stat().st_size:,} bytes")

        tmp = Path(state_dir) / "latest.snap.tmp"
        check("no lingering .tmp file", not tmp.exists())

        # Save again to trigger rotation
        brain.save()
        rolling = list(Path(state_dir).glob("rolling_*.snap"))
        check("rolling backup created on second save", len(rolling) >= 1,
              f"{len(rolling)} rolling files")

        # Spam saves to test rotation cap
        for _ in range(8):
            brain.save()
        rolling = list(Path(state_dir).glob("rolling_*.snap"))
        check("rolling backups capped at rolling_keep",
              len(rolling) <= brain.cryo.rolling_keep,
              f"{len(rolling)} rolling files, cap={brain.cryo.rolling_keep}")

        meta = Path(state_dir) / "metadata.json"
        check("metadata.json written", meta.exists())
    finally:
        shutil.rmtree(state_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Test 2: autosave fires on schedule
# ---------------------------------------------------------------------------

def test_autosave_schedule():
    print("\n[auto] autosave fires on tick interval")
    state_dir = tempfile.mkdtemp(prefix="chip_test_")
    try:
        from brain import ChipBrain

        save_events = []
        brain = ChipBrain(config={
            "state_dir": state_dir,
            "save_every": 5,           # fast for testing
            "auto_restore": False,
        }).boot()
        brain.hooks.on("autosave", lambda p: save_events.append(p))

        # 12 ticks should trigger 2 autosaves (at tick 5, 10)
        for i in range(12):
            brain.tick(f"observation {i}")

        check("autosave fired at least once", len(save_events) >= 1,
              f"saw {len(save_events)} autosaves")
        check("autosave count matches expected schedule",
              len(save_events) == 2,
              f"expected 2, saw {len(save_events)}")
    finally:
        shutil.rmtree(state_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Test 3: state survives save/restore round-trip
# ---------------------------------------------------------------------------

def test_state_round_trip():
    print("\n[round-trip] state survives save -> shutdown -> reboot")
    state_dir = tempfile.mkdtemp(prefix="chip_test_")
    try:
        from brain import ChipBrain

        # First brain: do some work, then save
        brain1 = ChipBrain(config={
            "state_dir": state_dir,
            "save_every": 0,
            "auto_restore": False,
        }).boot()

        # Plant identifiable state
        brain1.memory.store_text(
            ["A vivid memory of a sunny afternoon."],
            valence=0.8,
            empowerment_score=0.5,
        )
        brain1.emotions.set_mood_happy("test plant")
        before_mood = brain1.emotions.current_mood()[0]
        before_size = brain1.memory.size
        before_homeo = brain1.emotions._homeostasis._state.detach().clone()

        ok = brain1.shutdown()
        check("shutdown saved successfully", ok)

        # Second brain: boot with auto_restore, verify state matches
        brain2 = ChipBrain(config={
            "state_dir": state_dir,
            "save_every": 0,
            "auto_restore": True,
        }).boot()

        after_mood = brain2.emotions.current_mood()[0]
        after_size = brain2.memory.size
        after_homeo = brain2.emotions._homeostasis._state.detach().clone()

        check("mood restored", after_mood == before_mood,
              f"{before_mood} -> {after_mood}")
        check("episodic memory size restored", after_size == before_size,
              f"{before_size} -> {after_size}")
        check("homeostasis vector restored",
              torch.allclose(before_homeo, after_homeo, atol=1e-5),
              f"max diff = {(before_homeo - after_homeo).abs().max().item():.2e}")
    finally:
        shutil.rmtree(state_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Test 4: HMAC tamper detection
# ---------------------------------------------------------------------------

def test_hmac_tamper_detection():
    print("\n[security] HMAC tamper detection")
    state_dir = tempfile.mkdtemp(prefix="chip_test_")
    try:
        from brain import ChipBrain
        from brainstem.cryostasis import Cryostasis

        # Save with one secret
        cryo_a = Cryostasis(state_dir=state_dir, save_every=0,
                            hmac_secret=b"secret-a")
        brain = ChipBrain(config={
            "state_dir": state_dir,
            "save_every": 0,
            "auto_restore": False,
        }, cryostasis=cryo_a).boot()
        brain.save()

        # Try to restore with a different secret — should fail
        cryo_b = Cryostasis(state_dir=state_dir, save_every=0,
                            hmac_secret=b"secret-b")
        result = cryo_b.restore_if_available(
            policy=brain.policy,
            episodic_memory=brain.memory,
            emotional_core=brain.emotions,
        )
        check("wrong-secret restore returns None", result is None)

        # And the right secret still works
        result_ok = cryo_a.restore_if_available(
            policy=brain.policy,
            episodic_memory=brain.memory,
            emotional_core=brain.emotions,
        )
        check("correct-secret restore returns dict",
              result_ok is not None,
              f"got {type(result_ok).__name__}")
    finally:
        shutil.rmtree(state_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Test 5: corrupt latest.snap falls back to rolling backup
# ---------------------------------------------------------------------------

def test_corrupt_fallback():
    print("\n[recovery] corrupt latest falls back to rolling backup")
    state_dir = tempfile.mkdtemp(prefix="chip_test_")
    try:
        from brain import ChipBrain
        brain = ChipBrain(config={
            "state_dir": state_dir,
            "save_every": 0,
            "auto_restore": False,
        }).boot()
        # Save twice so we get a rolling backup
        brain.save()
        brain.save()

        # Corrupt the latest snapshot
        latest = Path(state_dir) / "latest.snap"
        with latest.open("wb") as f:
            f.write(b"\x00" * 200)  # garbage

        # Restore should fall back to rolling
        result = brain.cryo.restore_if_available(
            policy=brain.policy,
            episodic_memory=brain.memory,
            emotional_core=brain.emotions,
        )
        check("corrupt latest falls back to rolling",
              result is not None,
              f"recovered from {result['path'] if result else 'nothing'}")
    finally:
        shutil.rmtree(state_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

def main() -> int:
    print("=" * 70)
    print("Feature 3: Persistent State / Autosave — E2E test")
    print("=" * 70)
    t0 = time.time()
    test_manual_save()
    test_autosave_schedule()
    test_state_round_trip()
    test_hmac_tamper_detection()
    test_corrupt_fallback()
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
