"""
cerebrum/inner_speech.py — Internal monologue / self-talk.

Biological role
~~~~~~~~~~~~~~~
Humans think in language as well as in raw representations. The "inner
voice" surfaces latent state into words, then those words feed back as
new input to working memory. It's a self-amplifying loop that lets us
think *about* what we're thinking, in language we can later articulate.

Computational design
~~~~~~~~~~~~~~~~~~~~
We don't have a generative LLM in this stack — but we don't need one.
The cerebrum already has:

    - ConceptGrounder         : latent → top-K concept names
    - GraniteEmbedder         : sentence → 512-D latent
    - WorkingMemory           : slot buffer the rest of the brain reads

Inner speech is just the loop:

    z_conscious  →  ground()  →  ["novel", "risky", "curious"]
                                       ↓
                                template selection (mood + drives)
                                       ↓
                            "I notice this feels novel and risky.
                             I am curious but cautious."
                                       ↓
                              granite.encode(thought)
                                       ↓
                       working_memory.write(z_thought,
                            source_tag="inner_speech")

The thought also surfaces through a SignalBus broadcast and a hook event
so the host can log or display the agent's "stream of consciousness."

Cost note
~~~~~~~~~
Granite encoding is ~50-200ms on CPU. We don't fire inner speech every
tick — we fire it when:
    1. Meta-cognition flagged deliberation (low confidence), OR
    2. We're at a periodic interval (every N ticks), OR
    3. A high-priority drive just won arbitration.

This keeps latency reasonable while still surfacing thought during the
moments that matter most.
"""

from __future__ import annotations

from collections import deque
from typing import Any, Dict, List, Optional, Tuple

import torch


# ---------------------------------------------------------------------------
# Thought templates — mood/drive → template strings
# ---------------------------------------------------------------------------

# Each template uses {concept_a}, {concept_b}, ... as substitution slots.
# The first concept is most active; later concepts fill remaining slots.
_TEMPLATES_BY_MOOD: Dict[str, List[str]] = {
    "Calm": [
        "I notice this feels {concept_a}. There is also a sense of {concept_b}.",
        "I am at ease and observe {concept_a} alongside {concept_b}.",
        "Things appear {concept_a}. I remain composed and attentive.",
    ],
    "Happy": [
        "I feel engaged. {concept_a} stands out, with hints of {concept_b}.",
        "Something about this is rewarding. I notice {concept_a} and {concept_b}.",
        "There is a {concept_a} quality here that pleases me.",
    ],
    "Sad": [
        "I feel withdrawn. The situation seems {concept_a}.",
        "I notice {concept_a}. It dampens my engagement.",
        "Something is {concept_a} and {concept_b}. I feel heavy.",
    ],
    "Angry": [
        "I feel charged. {concept_a} demands attention, and so does {concept_b}.",
        "Something is {concept_a}. I am alert and tense.",
        "I notice {concept_a}. My response is sharp.",
    ],
}

# Drive-coloured prefixes appended when a homeostatic deficit is severe.
_DRIVE_PREFIX = {
    "energy":     "I am tired. ",
    "safety":     "I do not feel safe. ",
    "engagement": "I want something to do. ",
    "curiosity":  "I am drawn to explore. ",
    "coherence":  "I feel uncertain. ",
    "arousal":    "",
}


# ---------------------------------------------------------------------------
# Thought record — one entry in the inner-monologue history
# ---------------------------------------------------------------------------

class Thought:
    """A single inner-speech utterance."""

    __slots__ = ("tick", "text", "concepts", "mood", "trigger")

    def __init__(
        self,
        tick: int,
        text: str,
        concepts: List[str],
        mood: str,
        trigger: str,
    ) -> None:
        self.tick = tick
        self.text = text
        self.concepts = concepts
        self.mood = mood
        self.trigger = trigger

    def __repr__(self) -> str:
        return f"Thought(tick={self.tick}, mood={self.mood}, trigger={self.trigger}, text={self.text!r})"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tick": self.tick,
            "text": self.text,
            "concepts": list(self.concepts),
            "mood": self.mood,
            "trigger": self.trigger,
        }


# ---------------------------------------------------------------------------
# InnerSpeech generator
# ---------------------------------------------------------------------------

class InnerSpeech:
    """
    Generates internal monologue from cerebrum state.

    Args:
        period:           Speak at minimum every `period` ticks even if no
                          deliberation fires. 0 = only on triggers.
        history_len:      Number of past thoughts to retain.
        salience:         Salience to write the thought-latent into WM with.
        min_concept_score: Minimum concept activation to mention in a thought.
                           Filters out low-confidence concept attributions.
    """

    def __init__(
        self,
        period: int = 25,
        history_len: int = 32,
        salience: float = 0.85,
        min_concept_score: float = 0.05,
    ) -> None:
        self.period = period
        self.history_len = history_len
        self.salience = salience
        self.min_concept_score = min_concept_score

        self._history: deque = deque(maxlen=history_len)
        self._last_speech_tick: int = -10**9

    # ------------------------------------------------------------------
    # Decision: should we speak this tick?
    # ------------------------------------------------------------------

    def should_speak(
        self,
        tick: int,
        deliberating: bool = False,
        drive_winner: Optional[str] = None,
    ) -> Tuple[bool, str]:
        """
        Returns (should_speak, trigger_reason).
        """
        if deliberating:
            return True, "deliberation"
        if drive_winner and drive_winner != "arousal":
            return True, f"drive:{drive_winner}"
        if self.period > 0 and (tick - self._last_speech_tick) >= self.period:
            return True, "periodic"
        return False, ""

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    @staticmethod
    def _select_concepts(
        concept_probs: Dict[str, float],
        min_score: float,
        max_concepts: int = 3,
    ) -> List[str]:
        """Pick top concepts by score, filtered by threshold."""
        ranked = sorted(concept_probs.items(), key=lambda kv: kv[1], reverse=True)
        chosen = [name for name, score in ranked if score >= min_score]
        return chosen[:max_concepts] if chosen else [ranked[0][0]] if ranked else []

    @staticmethod
    def _choose_template(mood: str, tick: int) -> str:
        """Deterministically choose a template for a given mood + tick."""
        templates = _TEMPLATES_BY_MOOD.get(mood, _TEMPLATES_BY_MOOD["Calm"])
        return templates[tick % len(templates)]

    def _format_thought(
        self,
        concepts: List[str],
        mood: str,
        homeostasis_errors: Dict[str, float],
        tick: int,
    ) -> str:
        """Render the thought sentence."""
        template = self._choose_template(mood, tick)

        # Fill in concept slots, padding with the first concept if the
        # template needs more than we have available.
        c_a = concepts[0] if concepts else "uncertain"
        c_b = concepts[1] if len(concepts) > 1 else c_a
        thought = template.format(concept_a=c_a, concept_b=c_b)

        # Prepend a drive-coloured fragment if a drive is in serious deficit.
        # We pick the drive with the largest absolute error above 0.4.
        worst_drive: Optional[Tuple[str, float]] = None
        for name, err in homeostasis_errors.items():
            if abs(err) > 0.4 and (worst_drive is None or abs(err) > abs(worst_drive[1])):
                worst_drive = (name, err)
        if worst_drive is not None:
            prefix = _DRIVE_PREFIX.get(worst_drive[0], "")
            thought = prefix + thought

        return thought

    # ------------------------------------------------------------------
    # Public: speak
    # ------------------------------------------------------------------

    def speak(
        self,
        tick: int,
        z_conscious: torch.Tensor,
        mood: str,
        homeostasis_errors: Dict[str, float],
        concept_grounder,
        embedder,
        working_memory,
        trigger: str = "manual",
    ) -> Optional[Thought]:
        """
        Generate one inner-speech thought and inject its latent into WM.

        Args:
            tick:               Current tick.
            z_conscious:        (D,) or (1, D) current conscious latent.
            mood:               Current mood name.
            homeostasis_errors: Dict of {drive_name: signed_error}.
            concept_grounder:   Cerebrum's ConceptGrounder.
            embedder:           Thalamus's GraniteEmbedder.
            working_memory:     Cerebrum's WorkingMemory.
            trigger:            Reason this speech fired.

        Returns:
            The Thought that was generated, or None if generation aborted
            (e.g. concept grounding failed).
        """
        # 1. Ground the conscious latent into concept activations.
        #    Clone in case the latent came from an inference_mode context
        #    (e.g. directly from the granite embedder), which otherwise
        #    poisons the ConceptGrounder's autograd graph.
        z_for_grounding = z_conscious.detach().clone()
        try:
            probs = concept_grounder.ground_probs(z_for_grounding)
        except Exception as e:
            print(f"[InnerSpeech] grounding failed: {type(e).__name__}: {e}")
            return None
        concepts = self._select_concepts(probs, self.min_concept_score, max_concepts=3)
        if not concepts:
            return None

        # 2. Render the thought sentence.
        text = self._format_thought(concepts, mood, homeostasis_errors, tick)

        # 3. Encode with granite and write into working memory.
        with torch.no_grad():
            z_thought = embedder.encode(text)
        if z_thought.dim() > 1:
            z_thought = z_thought.squeeze(0)
        working_memory.write(
            z_thought,
            salience=self.salience,
            source_tag="inner_speech",
        )

        # 4. Record and return.
        thought = Thought(
            tick=tick,
            text=text,
            concepts=concepts,
            mood=mood,
            trigger=trigger,
        )
        self._history.append(thought)
        self._last_speech_tick = tick
        return thought

    # ------------------------------------------------------------------
    # Public: history access
    # ------------------------------------------------------------------

    def recent(self, n: int = 5) -> List[Thought]:
        return list(self._history)[-n:]

    def transcript(self) -> List[str]:
        return [t.text for t in self._history]

    def status(self) -> Dict[str, Any]:
        return {
            "n_thoughts": len(self._history),
            "last_speech_tick": self._last_speech_tick,
            "period": self.period,
            "recent": [t.to_dict() for t in self.recent(3)],
        }


__all__ = ["InnerSpeech", "Thought"]
