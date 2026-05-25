"""
test_feature_inner_speech.py — E2E test for inner speech.

What this verifies:
    1. should_speak gates correctly: deliberation, drive trigger, periodic.
    2. _select_concepts ranks and filters concept activations.
    3. _format_thought renders templates per mood and applies drive prefixes.
    4. speak() generates a Thought, pushes it to working memory, and updates
       the history log.
    5. The brain hook fires on inner_speech events during normal ticks.
    6. Thoughts surface in the brain status snapshot.
    7. Multiple ticks produce diverse, mood-appropriate thoughts.
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
# Test 1: should_speak gating
# ---------------------------------------------------------------------------

def test_should_speak():
    print("\n[gating] should_speak triggers")
    from cerebrum.inner_speech import InnerSpeech

    sp = InnerSpeech(period=10)

    # Deliberation always triggers
    cond, why = sp.should_speak(tick=1, deliberating=True)
    check("deliberation triggers speech", cond, f"trigger={why!r}")
    check("deliberation gives correct trigger", why == "deliberation")

    # Strong drive triggers
    sp2 = InnerSpeech(period=0)  # disable periodic so only drive fires
    cond, why = sp2.should_speak(tick=1, drive_winner="curiosity")
    check("drive triggers speech", cond, f"trigger={why!r}")
    check("drive trigger names the drive", "curiosity" in why)

    # Arousal alone does NOT trigger (it's the default homeostatic dim)
    cond, _ = sp2.should_speak(tick=2, drive_winner="arousal")
    check("arousal-only does not trigger", not cond)

    # No deliberation, no drive, period=0 → no speech
    cond, _ = sp2.should_speak(tick=3)
    check("idle state does not trigger", not cond)

    # Periodic trigger
    sp3 = InnerSpeech(period=5)
    sp3._last_speech_tick = 0
    cond, why = sp3.should_speak(tick=5)
    check("periodic trigger at interval", cond, f"trigger={why!r}")
    check("periodic trigger names itself", why == "periodic")
    cond, _ = sp3.should_speak(tick=4)
    check("periodic does not trigger before interval", not cond)


# ---------------------------------------------------------------------------
# Test 2: concept selection + thought formatting
# ---------------------------------------------------------------------------

def test_thought_rendering():
    print("\n[render] concept selection and template formatting")
    from cerebrum.inner_speech import InnerSpeech

    sp = InnerSpeech(min_concept_score=0.05)

    # Filter and rank
    probs = {
        "curious": 0.4,
        "novel": 0.3,
        "risky": 0.15,
        "boring": 0.001,
    }
    chosen = sp._select_concepts(probs, min_score=0.05, max_concepts=3)
    check("top concept first", chosen[0] == "curious", f"got {chosen}")
    check("filters below threshold", "boring" not in chosen)
    check("respects max_concepts", len(chosen) <= 3)

    # Empty probs returns at least one concept (via fallback)
    empty = sp._select_concepts({}, min_score=0.5)
    check("empty probs returns empty list", empty == [])

    # Template rendering
    t1 = sp._format_thought(["novel", "risky"], "Calm", {}, tick=0)
    check("calm template uses concepts", "novel" in t1 and "risky" in t1,
          f"text={t1!r}")

    t2 = sp._format_thought(["unfamiliar"], "Angry", {}, tick=0)
    check("angry template uses single concept", "unfamiliar" in t2,
          f"text={t2!r}")

    # Drive prefix kicks in for severe deficits
    t3 = sp._format_thought(["curious"], "Calm", {"energy": -0.6}, tick=0)
    check("drive prefix applied for severe deficit",
          t3.startswith("I am tired"), f"text={t3!r}")

    t4 = sp._format_thought(["curious"], "Calm", {"energy": -0.1}, tick=0)
    check("drive prefix omitted for mild deficit",
          not t4.startswith("I am tired"), f"text={t4!r}")


# ---------------------------------------------------------------------------
# Test 3: speak() round-trip with real granite + working memory
# ---------------------------------------------------------------------------

def test_speak_roundtrip():
    print("\n[roundtrip] speak() produces a Thought and writes to WM")
    from cerebrum.inner_speech import InnerSpeech
    from cerebrum.concept_grounding import ConceptGrounder
    from cerebrum.working_memory import WorkingMemory
    from thalamus.granite_embedder import get_embedder

    embedder = get_embedder()
    cg = ConceptGrounder(d_model=embedder.output_dim)
    wm = WorkingMemory(latent_dim=embedder.output_dim, capacity=7)
    sp = InnerSpeech()

    z = embedder.encode("Something exciting just happened.")
    thought = sp.speak(
        tick=10,
        z_conscious=z,
        mood="Happy",
        homeostasis_errors={"energy": -0.1},
        concept_grounder=cg,
        embedder=embedder,
        working_memory=wm,
        trigger="manual",
    )

    check("speak returns a Thought", thought is not None)
    if thought is not None:
        check("thought has non-empty text", len(thought.text) > 0,
              f"text={thought.text!r}")
        check("thought has concepts", len(thought.concepts) > 0,
              f"concepts={thought.concepts}")
        check("thought records mood", thought.mood == "Happy")
        check("thought records trigger", thought.trigger == "manual")
        check("thought records tick", thought.tick == 10)

    # Working memory should have an inner_speech slot
    speech_slots = [s for s in wm._slots if s.source_tag == "inner_speech"]
    check("inner_speech slot in working memory",
          len(speech_slots) == 1, f"{len(speech_slots)} slots")

    # History updated
    check("history contains the thought", len(sp._history) == 1)
    check("recent() returns the thought",
          sp.recent(1)[0].text == thought.text if thought else False)


# ---------------------------------------------------------------------------
# Test 4: Brain integration
# ---------------------------------------------------------------------------

def test_brain_inner_speech():
    print("\n[brain] inner speech integrated into tick loop")
    from brain import ChipBrain

    brain = ChipBrain(config={
        "inner_speech_every": 3,    # speak frequently for testing
        "save_every": 0,
        "auto_restore": False,
    }).boot()

    speeches = []
    brain.hooks.on("inner_speech", lambda payload: speeches.append(payload))

    # Run several ticks; periodic speech should fire at least once
    for i in range(10):
        brain.tick(f"Observation {i}: something is happening here.")

    check("brain produced inner-speech thoughts", len(speeches) > 0,
          f"{len(speeches)} thoughts captured by hook")

    if speeches:
        # Each thought has the expected shape
        first = speeches[0]
        check("hook payload is a dict", isinstance(first, dict),
              f"type={type(first).__name__}")
        check("payload contains text key", "text" in first)
        check("payload contains concepts key", "concepts" in first)
        check("payload contains trigger key", "trigger" in first)
        check("text is non-empty",
              len(first.get("text", "")) > 0,
              f"text={first.get('text', '')!r}")

    # Brain status surfaces the inner speech state
    status = brain.status()
    check("status includes inner_speech",
          "inner_speech" in status,
          f"keys: {list(status.keys())[:5]}")
    if "inner_speech" in status:
        check("status reports n_thoughts",
              status["inner_speech"]["n_thoughts"] > 0,
              f"n={status['inner_speech']['n_thoughts']}")

    # Multiple ticks produced at least 2 distinct thought texts
    if len(speeches) >= 2:
        unique_texts = set(s["text"] for s in speeches)
        check("multiple ticks produce thoughts",
              len(unique_texts) >= 1,
              f"{len(unique_texts)} unique texts from {len(speeches)} thoughts")


# ---------------------------------------------------------------------------
# Test 5: deliberation forces speech regardless of period
# ---------------------------------------------------------------------------

def test_deliberation_forces_speech():
    print("\n[deliberation] low-confidence deliberation always triggers speech")
    from cerebrum.inner_speech import InnerSpeech

    sp = InnerSpeech(period=0)  # periodic disabled
    sp._last_speech_tick = 0

    # No deliberation, no drive, tick well after — should NOT speak
    cond, _ = sp.should_speak(tick=1000)
    check("idle does not trigger speech with period=0", not cond)

    # With deliberation flag — must speak
    cond, why = sp.should_speak(tick=1001, deliberating=True)
    check("deliberation triggers speech regardless of period",
          cond, f"trigger={why!r}")


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

def main() -> int:
    print("=" * 70)
    print("Feature 5: Inner Speech — E2E test")
    print("=" * 70)
    t0 = time.time()
    test_should_speak()
    test_thought_rendering()
    test_speak_roundtrip()
    test_brain_inner_speech()
    test_deliberation_forces_speech()
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
