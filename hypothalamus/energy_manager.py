"""
hypothalamus/energy_manager.py — Compute budget and fatigue modelling.

The hypothalamus regulates energy metabolism — it knows when the body
is fatigued and throttles activity accordingly. This module tracks
compute budget as "energy" and models cognitive fatigue: diminishing
returns from sustained computation on the same problem.
"""

from __future__ import annotations

import time
from collections import deque
from typing import Dict, Optional

import torch


class EnergyManager:
    """
    Tracks cognitive energy and allocates compute budget across regions.

    Energy model:
        - Each forward pass costs energy proportional to model size.
        - Energy recovers at a fixed rate per wall-clock second.
        - When energy is low, expensive operations (world model rollouts,
          dream cycles) are deferred.

    Fatigue model:
        - Sustained computation on the same task reduces efficiency.
        - Fatigue accumulates per task_id and decays during task switches.

    Args:
        max_energy:         Maximum energy budget (arbitrary units).
        recovery_rate:      Energy recovered per second of wall time.
        fatigue_decay:      Fatigue decay rate per task switch.
        low_energy_thresh:  Fraction of max_energy below which expensive
                            ops are blocked.
    """

    def __init__(
        self,
        max_energy: float = 1000.0,
        recovery_rate: float = 10.0,
        fatigue_decay: float = 0.95,
        low_energy_thresh: float = 0.2,
    ) -> None:
        self.max_energy = max_energy
        self.recovery_rate = recovery_rate
        self.fatigue_decay = fatigue_decay
        self.low_energy_thresh = low_energy_thresh

        self._energy = max_energy
        self._last_tick = time.time()
        self._fatigue: Dict[str, float] = {}
        self._current_task: Optional[str] = None
        self._op_costs: Dict[str, float] = {
            "forward_pass": 1.0,
            "world_model_rollout": 5.0,
            "dream_cycle": 10.0,
            "swarm_consensus": 2.0,
            "memory_consolidation": 8.0,
        }

    def _recover(self) -> None:
        """Recover energy based on elapsed wall time."""
        now = time.time()
        elapsed = now - self._last_tick
        self._energy = min(self.max_energy, self._energy + self.recovery_rate * elapsed)
        self._last_tick = now

    def can_afford(self, operation: str) -> bool:
        """Check if there's enough energy for an operation without spending it."""
        self._recover()
        cost = self._op_costs.get(operation, 1.0)
        return self._energy >= cost

    def spend(self, operation: str, task_id: Optional[str] = None) -> bool:
        """
        Spend energy for an operation. Returns True if successful, False if
        insufficient energy (operation should be deferred).
        """
        self._recover()
        cost = self._op_costs.get(operation, 1.0)

        # Apply fatigue multiplier for the current task
        if task_id is not None:
            fatigue = self._fatigue.get(task_id, 0.0)
            cost *= (1.0 + fatigue)
            self._fatigue[task_id] = min(fatigue + 0.01, 2.0)

        if self._energy < cost:
            return False

        self._energy -= cost

        # Task switch: decay fatigue on old task
        if task_id != self._current_task:
            if self._current_task in self._fatigue:
                self._fatigue[self._current_task] *= self.fatigue_decay
            self._current_task = task_id

        return True

    @property
    def energy_fraction(self) -> float:
        """Current energy as a fraction of maximum (0.0 to 1.0)."""
        self._recover()
        return self._energy / self.max_energy

    @property
    def is_fatigued(self) -> bool:
        """True when energy is below the low-energy threshold."""
        return self.energy_fraction < self.low_energy_thresh

    def register_op(self, name: str, cost: float) -> None:
        """Register a custom operation cost."""
        self._op_costs[name] = cost

    def status(self) -> Dict:
        self._recover()
        return {
            "energy": self._energy,
            "energy_fraction": self.energy_fraction,
            "is_fatigued": self.is_fatigued,
            "current_task": self._current_task,
            "fatigue_levels": dict(self._fatigue),
        }


__all__ = ["EnergyManager"]
