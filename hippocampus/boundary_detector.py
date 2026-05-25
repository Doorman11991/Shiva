"""
hippocampus/boundary_detector.py — Automatic episodic boundary detection.

Biological role
~~~~~~~~~~~~~~~
The hippocampus segments continuous experience into discrete episodes
at "event boundaries" — moments where the prediction error spikes
(something unexpected happens). Humans naturally remember the boundaries
between events better than the middle of an event. This module detects
those boundaries automatically from the world model's prediction error.

Design
~~~~~~
Track the world model's prediction error as a rolling signal. When the
error exceeds a threshold (based on recent statistics), a boundary is
declared: the current episode closes, is stored in memory, and a fresh
episode begins.

This replaces the need for the caller to manually signal done=True.
The brain can now segment its own stream of consciousness without
external supervision.

Integration
~~~~~~~~~~~
Called each tick after the world model prediction error is computed.
When a boundary fires:
    1. The current episode buffer is flushed to episodic memory.
    2. A "boundary_detected" signal is published on the bus.
    3. Working memory is partially decayed (not fully reset — some
       context should carry across boundaries).
    4. The temporal abstractor is notified.
"""

from __future__ import annotations

from collections import deque
from typing import Dict, List, Optional, Tuple

import torch


class BoundaryDetector:
    """
    Detects event boundaries from prediction error spikes.

    A boundary is declared when:
        prediction_error > mean(recent_errors) + k * std(recent_errors)

    Where k is the `sensitivity` parameter (higher = fewer boundaries).

    Args:
        sensitivity:     Number of standard deviations above mean to trigger.
        min_episode_len: Minimum ticks between boundaries (prevents micro-segmentation).
        window:          Rolling window size for error statistics.
        warmup:          Ticks before detection activates (need baseline stats).
    """

    def __init__(
        self,
        sensitivity: float = 2.0,
        min_episode_len: int = 8,
        window: int = 50,
        warmup: int = 10,
    ) -> None:
        self.sensitivity = sensitivity
        self.min_episode_len = min_episode_len
        self.window = window
        self.warmup = warmup

        self._error_history: deque = deque(maxlen=window)
        self._ticks_since_boundary: int = 0
        self._total_boundaries: int = 0
        self._tick: int = 0

        # Episode buffer: latents accumulated since last boundary
        self._episode_buffer: List[torch.Tensor] = []
        self._episode_valences: List[float] = []

    def tick(
        self,
        prediction_error: float,
        z_current: Optional[torch.Tensor] = None,
        valence: float = 0.0,
    ) -> bool:
        """
        Process one tick. Returns True if a boundary was detected.

        Args:
            prediction_error: World model prediction error for this tick.
            z_current:        (D,) current latent to buffer for the episode.
            valence:          Scalar valence for this tick.

        Returns:
            True if boundary detected, False otherwise.
        """
        self._tick += 1
        self._ticks_since_boundary += 1
        self._error_history.append(prediction_error)

        # Buffer the latent and valence for eventual episode storage
        if z_current is not None:
            self._episode_buffer.append(z_current.detach().cpu())
        self._episode_valences.append(valence)

        # Not enough data yet
        if self._tick < self.warmup:
            return False

        # Minimum episode length not reached
        if self._ticks_since_boundary < self.min_episode_len:
            return False

        # Compute threshold from recent error statistics
        errors = list(self._error_history)
        if len(errors) < 5:
            return False

        mean_err = sum(errors) / len(errors)
        var_err = sum((e - mean_err) ** 2 for e in errors) / len(errors)
        std_err = var_err ** 0.5

        threshold = mean_err + self.sensitivity * std_err

        if prediction_error > threshold:
            self._total_boundaries += 1
            self._ticks_since_boundary = 0
            return True

        return False

    def flush_episode(self) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Flush the accumulated episode buffer.

        Returns:
            (state_sequence, valence_sequence) or None if buffer is empty.
            state_sequence: (T, D) tensor
            valence_sequence: (T, 1) tensor
        """
        if not self._episode_buffer:
            return None

        states = torch.stack(self._episode_buffer)
        valences = torch.tensor(self._episode_valences, dtype=torch.float32).unsqueeze(1)

        # Clear the buffer for the next episode
        self._episode_buffer.clear()
        self._episode_valences.clear()

        return states, valences

    def reset(self) -> None:
        """Full reset (e.g. on environment reset)."""
        self._error_history.clear()
        self._ticks_since_boundary = 0
        self._tick = 0
        self._episode_buffer.clear()
        self._episode_valences.clear()

    @property
    def episode_length(self) -> int:
        """Ticks accumulated in the current episode buffer."""
        return len(self._episode_buffer)

    def status(self) -> Dict:
        errors = list(self._error_history)
        mean_err = sum(errors) / len(errors) if errors else 0.0
        return {
            "total_boundaries": self._total_boundaries,
            "ticks_since_boundary": self._ticks_since_boundary,
            "current_episode_length": self.episode_length,
            "mean_error": mean_err,
            "tick": self._tick,
        }


__all__ = ["BoundaryDetector"]
