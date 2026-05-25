"""
brainstem/scheduler.py — Learning rate and training phase management.

The brainstem regulates circadian rhythms and sleep-wake cycles —
transitions between different physiological states. This module manages
the analogous transitions in training: warmup, decay, and phase shifts
between exploration-heavy and exploitation-heavy regimes.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional

import torch
from torch.optim import Optimizer


class WarmupCosineScheduler:
    """
    Linear warmup followed by cosine annealing.

    During warmup (steps 0..warmup_steps), lr scales linearly from
    lr_min to lr_max. After warmup, cosine decay brings it back to lr_min
    over the remaining steps.

    Args:
        optimizer:      The optimizer whose lr groups will be managed.
        warmup_steps:   Number of linear warmup steps.
        total_steps:    Total training steps (warmup + cosine decay).
        lr_max:         Peak learning rate (reached at end of warmup).
        lr_min:         Minimum learning rate (floor of cosine decay).
    """

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        total_steps: int,
        lr_max: float,
        lr_min: float = 1e-6,
    ) -> None:
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.lr_max = lr_max
        self.lr_min = lr_min
        self._step = 0

    def step(self) -> float:
        """Advance one step and update optimizer lr. Returns current lr."""
        self._step += 1
        lr = self._compute_lr(self._step)
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        return lr

    def _compute_lr(self, step: int) -> float:
        if step <= self.warmup_steps:
            return self.lr_min + (self.lr_max - self.lr_min) * (step / max(self.warmup_steps, 1))
        progress = (step - self.warmup_steps) / max(self.total_steps - self.warmup_steps, 1)
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
        return self.lr_min + (self.lr_max - self.lr_min) * cosine

    @property
    def current_lr(self) -> float:
        return self._compute_lr(self._step)

    @property
    def step_count(self) -> int:
        return self._step


class TrainingPhaseManager:
    """
    Controls transitions between named training phases.

    Phases are defined as a list of (name, duration_steps) pairs.
    The manager advances through them in order, broadcasting phase
    change events to registered callbacks.

    Example phases:
        [("warmup", 1000), ("exploration", 10000), ("exploitation", 50000)]

    Args:
        phases: List of (phase_name, duration_in_steps) tuples.
    """

    def __init__(self, phases: List[tuple]) -> None:
        self._phases = phases
        self._phase_idx = 0
        self._phase_step = 0
        self._global_step = 0
        self._callbacks: List = []

    def register_callback(self, fn) -> None:
        """Register a callback(phase_name: str) called on phase transitions."""
        self._callbacks.append(fn)

    def tick(self) -> str:
        """Advance one step. Returns current phase name."""
        self._global_step += 1
        self._phase_step += 1

        if self._phase_idx < len(self._phases):
            _, duration = self._phases[self._phase_idx]
            if self._phase_step >= duration:
                self._phase_idx = min(self._phase_idx + 1, len(self._phases) - 1)
                self._phase_step = 0
                new_phase = self.current_phase
                print(f"[TrainingPhaseManager] Entering phase: {new_phase}")
                for cb in self._callbacks:
                    cb(new_phase)

        return self.current_phase

    @property
    def current_phase(self) -> str:
        if self._phase_idx >= len(self._phases):
            return self._phases[-1][0]
        return self._phases[self._phase_idx][0]

    @property
    def phase_progress(self) -> float:
        """Fraction through the current phase (0.0 to 1.0)."""
        if self._phase_idx >= len(self._phases):
            return 1.0
        _, duration = self._phases[self._phase_idx]
        return min(self._phase_step / max(duration, 1), 1.0)

    def status(self) -> Dict:
        return {
            "global_step": self._global_step,
            "current_phase": self.current_phase,
            "phase_progress": self.phase_progress,
            "phase_step": self._phase_step,
        }


__all__ = ["WarmupCosineScheduler", "TrainingPhaseManager"]
