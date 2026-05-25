"""
brainstem/circadian.py — Sleep/wake cycle gating.

Instead of dream cycles and memory consolidation firing on fixed tick
counters (every 50 ticks, every 200 ticks), this module decides WHEN
the brain should "sleep" based on accumulated cognitive load, recent
reward trajectory, and energy levels.

Sleep triggers when:
    - Energy is low (hypothalamus says fatigued)
    - Recent reward has plateaued (not learning from waking experience)
    - Prediction error is low (world model has learned the current regime)

During sleep:
    - Dream cycles run (active dreaming + passive replay)
    - Memory consolidation fires
    - World model gets extra training steps
    - Working memory is cleared

Wake triggers when:
    - Sleep budget is exhausted (max sleep duration)
    - A novel signal arrives (dishabituation)
    - Energy recovers above threshold
"""

from __future__ import annotations

from collections import deque
from typing import Dict, Optional


class CircadianCycle:
    """
    Manages the sleep/wake cycle.

    Args:
        sleep_threshold:    Energy fraction below which sleep is triggered.
        wake_threshold:     Energy fraction above which the brain wakes.
        min_wake_ticks:     Minimum ticks between sleep episodes.
        max_sleep_ticks:    Maximum duration of a sleep episode.
        reward_plateau_window: Window for detecting reward plateau.
        reward_plateau_threshold: Reward variance below this = plateau.
    """

    def __init__(
        self,
        sleep_threshold: float = 0.2,
        wake_threshold: float = 0.6,
        min_wake_ticks: int = 50,
        max_sleep_ticks: int = 20,
        reward_plateau_window: int = 30,
        reward_plateau_threshold: float = 0.01,
    ) -> None:
        self.sleep_threshold = sleep_threshold
        self.wake_threshold = wake_threshold
        self.min_wake_ticks = min_wake_ticks
        self.max_sleep_ticks = max_sleep_ticks
        self.reward_plateau_window = reward_plateau_window
        self.reward_plateau_threshold = reward_plateau_threshold

        self._sleeping = False
        self._sleep_ticks = 0
        self._ticks_since_sleep = 0
        self._total_sleep_episodes = 0
        self._reward_history: deque = deque(maxlen=reward_plateau_window)

    def record_reward(self, reward: float) -> None:
        """Record a reward observation for plateau detection."""
        self._reward_history.append(reward)

    def should_sleep(
        self,
        energy_fraction: float,
        mean_prediction_error: float = 0.0,
    ) -> bool:
        """
        Determine if the brain should enter sleep.

        Args:
            energy_fraction:      Current energy level [0, 1].
            mean_prediction_error: World model's recent mean error.

        Returns:
            True if sleep should begin.
        """
        if self._sleeping:
            return False  # already sleeping
        if self._ticks_since_sleep < self.min_wake_ticks:
            return False  # too soon since last sleep

        # Condition 1: energy is low
        if energy_fraction <= self.sleep_threshold:
            return True

        # Condition 2: reward has plateaued AND prediction error is low
        if len(self._reward_history) >= self.reward_plateau_window:
            rewards = list(self._reward_history)
            mean_r = sum(rewards) / len(rewards)
            var_r = sum((r - mean_r) ** 2 for r in rewards) / len(rewards)
            if var_r < self.reward_plateau_threshold and mean_prediction_error < 0.1:
                return True

        return False

    def should_wake(
        self,
        energy_fraction: float,
        novelty_spike: bool = False,
    ) -> bool:
        """
        Determine if the brain should wake from sleep.

        Args:
            energy_fraction: Current energy level [0, 1].
            novelty_spike:   True if a novel signal arrived during sleep.

        Returns:
            True if the brain should wake.
        """
        if not self._sleeping:
            return False

        # Condition 1: max sleep duration reached
        if self._sleep_ticks >= self.max_sleep_ticks:
            return True

        # Condition 2: energy recovered
        if energy_fraction >= self.wake_threshold:
            return True

        # Condition 3: novel signal demands attention
        if novelty_spike:
            return True

        return False

    def enter_sleep(self) -> None:
        """Transition to sleep state."""
        self._sleeping = True
        self._sleep_ticks = 0
        self._total_sleep_episodes += 1

    def wake_up(self) -> None:
        """Transition to wake state."""
        self._sleeping = False
        self._ticks_since_sleep = 0

    def tick(self) -> None:
        """Advance one tick. Call every brain tick."""
        if self._sleeping:
            self._sleep_ticks += 1
        else:
            self._ticks_since_sleep += 1

    @property
    def is_sleeping(self) -> bool:
        return self._sleeping

    def status(self) -> Dict:
        return {
            "sleeping": self._sleeping,
            "sleep_ticks": self._sleep_ticks,
            "ticks_since_sleep": self._ticks_since_sleep,
            "total_sleep_episodes": self._total_sleep_episodes,
        }


__all__ = ["CircadianCycle"]
