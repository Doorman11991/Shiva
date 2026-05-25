"""
run.py — Interactive Chip brain session.

Just run it:
    python run.py

Type observations, see what Chip thinks. Type 'status' for brain state,
'quit' to exit. State auto-saves between sessions.
"""

import sys
import os

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from brain import ChipBrain


def main():
    print("=" * 60)
    print("  Chip Brain — Interactive Session")
    print("=" * 60)
    print()
    print("  Commands:")
    print("    [text]     Feed an observation to the brain")
    print("    status     Show brain state")
    print("    mood       Show current mood and drives")
    print("    thoughts   Show recent inner speech")
    print("    goals      Show active goals")
    print("    memory     Show memory stats")
    print("    save       Force save to disk")
    print("    quit       Save and exit")
    print()

    brain = ChipBrain(config={
        "save_every": 50,
        "inner_speech_every": 5,
    }).boot()
    print()

    tick_count = 0

    while True:
        try:
            user_input = input("\nyou > ").strip()
        except (KeyboardInterrupt, EOFError):
            break

        if not user_input:
            continue

        cmd = user_input.lower()

        if cmd == "quit" or cmd == "exit":
            break

        elif cmd == "status":
            s = brain.status()
            print(f"\n  tick:       {s['tick']}")
            print(f"  device:     {s['device']}")
            print(f"  mood:       {s['mood']}")
            print(f"  top goal:   {s['top_goal']}")
            print(f"  confidence: {s['meta_cognition']['mean_confidence']:.3f}")
            print(f"  memories:   {s['episodic_memory_size']}")
            print(f"  wm slots:   {s['working_memory']['n_slots']}/{s['working_memory']['capacity']}")

        elif cmd == "mood":
            s = brain.status()
            print(f"\n  mood:    {s['mood']}")
            print(f"  drives:  {s['homeostasis']}")
            print(f"  urgent:  {s['most_urgent_drive']}")

        elif cmd == "thoughts":
            recent = brain.inner_speech.recent(5)
            if not recent:
                print("\n  (no thoughts yet — keep talking to me)")
            else:
                print()
                for t in recent:
                    print(f"  [{t.mood}] {t.text}")

        elif cmd == "goals":
            gs = brain.goal_stack.status()
            if gs["is_empty"]:
                print("\n  (no active goals)")
            else:
                print(f"\n  depth: {gs['depth']}")
                for f in gs["stack"]:
                    print(f"    {'>' if f['is_leaf'] else '-'} {f['name']} "
                          f"(urgency={f.get('urgency', '?')}, ticks={f['ticks_active']})")

        elif cmd == "memory":
            s = brain.status()
            print(f"\n  episodes:     {s['episodic_memory_size']}")
            print(f"  place cells:  {s['cognitive_map']['n_cells']}")
            print(f"  transitions:  {s['cognitive_map']['n_transitions']}")
            print(f"  temporal:     {s['temporal_levels_ready']}")

        elif cmd == "save":
            ok = brain.save()
            print(f"\n  {'saved' if ok else 'save failed'}")

        else:
            # Treat as an observation
            action = brain.tick(user_input)
            tick_count += 1

            # Show what happened
            a_mag = action.abs().mean().item()
            mood, _ = brain.emotions.current_mood()
            conf = brain.meta._confidence_history[-1] if brain.meta._confidence_history else 0.5

            print(f"\n  [tick {brain._tick}] mood={mood}, confidence={conf:.2f}, |action|={a_mag:.3f}")

            # Show inner speech if it fired this tick
            recent = brain.inner_speech.recent(1)
            if recent and recent[-1].tick == brain._tick:
                print(f"  thought: \"{recent[-1].text}\"")

            # Feed a small positive reward for engagement
            brain.train_step(reward=0.05, done=False)

    # Shutdown
    print("\n\nSaving brain state...")
    brain.shutdown()
    print(f"Done. {tick_count} observations processed.")


if __name__ == "__main__":
    main()
