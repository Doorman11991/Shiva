"""
amygdala/emotional_core.py — Emotion processing centre.

The amygdala processes emotions, especially fear and threat responses.
It tags memories with emotional significance and modulates attention
based on arousal state. This module is the computational equivalent:
mood tracking, homeostatic drives, and the valence network that maps
latent states to emotional tone.

Moved from: core/emotional_core.py
"""

from __future__ import annotations
from typing import Tuple
import torch
import torch.nn as nn
from thalamus.latent_alignment import LatentAligner


class MoodState:
    """
    Tracks a single named emotion and the reason for the most recent change.
    Performs no tensor operations — pure bookkeeping.
    """

    def __init__(self, initial_mood: str, valid_vocab: dict) -> None:
        if initial_mood not in valid_vocab:
            raise ValueError(
                f"Initial mood '{initial_mood}' not in vocabulary: "
                f"{list(valid_vocab.keys())}"
            )
        self._mood: str = initial_mood
        self._reason: str = "Initialization"
        self._vocab: dict = valid_vocab

    @property
    def name(self) -> str:
        return self._mood

    @property
    def reason(self) -> str:
        return self._reason

    def transition(self, new_mood: str, reason: str) -> None:
        if new_mood not in self._vocab:
            print(f"[MoodState] Unknown mood '{new_mood}'; transition ignored.")
            return
        self._mood = new_mood
        self._reason = reason


class HomeostasisState:
    """
    4-D homeostatic drive vector: [arousal, energy, safety, engagement].
    Clamped to [0,1]. strain() = L2 distance from target setpoint.
    """

    def __init__(
        self,
        initial: Tuple[float, float, float, float] = (0.5, 0.8, 1.0, 0.5),
        target: Tuple[float, float, float, float] = (0.8, 1.0, 0.7, 0.6),
    ) -> None:
        self._state = nn.Parameter(torch.tensor(list(initial)), requires_grad=False)
        self._target = torch.tensor(list(target))

    @property
    def vector(self) -> torch.Tensor:
        return self._state

    def update(
        self,
        action_impact: float,
        environment_surprise: float,
        policy_violation: float = 0.0,
        tool_failure_rate: float = 0.0,
        task_success: float = 0.0,
    ) -> None:
        self._state[0] += 0.05 * environment_surprise
        self._state[1] -= 0.02 * action_impact
        self._state[2] -= 0.10 * float(policy_violation)
        self._state[3] -= 0.05 * float(tool_failure_rate)
        self._state[3] += 0.03 * float(task_success)
        self._state.clamp_(0, 1)

    def strain(self) -> torch.Tensor:
        target = self._target.to(self._state.device)
        return torch.norm(self._state - target)


class EmotionalCore(nn.Module):
    """
    Amygdala: orchestrates mood + homeostasis + valence network.

    Responsibilities:
      1. Discrete mood bookkeeping (MoodState).
      2. Homeostatic drive updates (HomeostasisState).
      3. Learned valence network: (latent_state ‖ internal_state) → scalar ∈ (−1, +1).

    Publishes to SignalBus:
      "valence_update"     → cerebrum   (current emotional valence)
      "arousal_gain"       → thalamus   (attention sensitivity scale)
      "homeostasis_update" → hypothalamus (internal state vector)
    """

    def __init__(
        self,
        latent_aligner: LatentAligner,
        initial_mood: str = "Calm",
        hidden_dim: int = 512,
    ) -> None:
        super().__init__()
        self._aligner = latent_aligner
        self._mood = MoodState(initial_mood, latent_aligner.emotion_vocab)
        self._homeostasis = HomeostasisState()
        self.internal_state = self._homeostasis._state
        self.valence_network = nn.Sequential(
            nn.Linear(hidden_dim + 4, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Tanh(),
        )

    def current_mood(self) -> Tuple[str, torch.Tensor]:
        mood_id = self._aligner.emotion_vocab[self._mood.name]
        idx = torch.tensor(
            [mood_id], dtype=torch.long,
            device=self._aligner.emotion_embeddings.weight.device,
        )
        vector = self._aligner.emotion_embeddings(idx).squeeze(0).detach()
        return self._mood.name, vector

    def mood_swing(self, new_mood: str, reason: str) -> None:
        self._mood.transition(new_mood, reason)

    def reason_for_mood_change(self) -> str:
        return self._mood.reason

    def set_mood_angry(self, reason: str) -> None:
        self.mood_swing("Angry", reason)

    def set_mood_sad(self, reason: str) -> None:
        self.mood_swing("Sad", reason)

    def set_mood_happy(self, reason: str) -> None:
        self.mood_swing("Happy", reason)

    def set_mood_calm(self, reason: str) -> None:
        self.mood_swing("Calm", reason)

    def auto_transition_mood(
        self, valence: float, arousal: float, reason: str = "auto_circumplex",
    ) -> str:
        v = float(max(-1.0, min(1.0, valence)))
        a = float(max(0.0, min(1.0, arousal)))
        v_pos = v > 0.10
        v_neg = v < -0.10
        a_high = a > 0.60
        a_low = a < 0.40

        if a_high and v_pos:
            new_mood = "Happy"
        elif a_high and v_neg:
            new_mood = "Angry"
        elif a_low and v_neg:
            new_mood = "Sad"
        elif a_low and v_pos:
            new_mood = "Calm"
        else:
            return self._mood.name

        if new_mood != self._mood.name:
            self._mood.transition(new_mood, reason)
        return new_mood

    def update_homeostasis(
        self,
        action_impact: float,
        environment_surprise: float,
        policy_violation: float = 0.0,
        tool_failure_rate: float = 0.0,
        task_success: float = 0.0,
    ) -> None:
        self._homeostasis.update(
            action_impact=action_impact,
            environment_surprise=environment_surprise,
            policy_violation=policy_violation,
            tool_failure_rate=tool_failure_rate,
            task_success=task_success,
        )

    def calculate_strain(self) -> torch.Tensor:
        return self._homeostasis.strain()

    def get_valence(self, latent_state: torch.Tensor) -> torch.Tensor:
        s = self._homeostasis.vector
        if latent_state.dim() > 1:
            s = s.unsqueeze(0).expand(latent_state.size(0), -1)
        combined = torch.cat([latent_state, s], dim=-1)
        return self.valence_network(combined)
