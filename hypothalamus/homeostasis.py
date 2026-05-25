"""
hypothalamus/homeostasis.py — System-wide homeostatic regulation.

The hypothalamus maintains the body's internal equilibrium — temperature,
hunger, thirst. This module defines system-wide setpoints and computes
error signals when actual state deviates from target, driving the agent
toward self-correcting behaviour.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn


class HomeostaticRegulator(nn.Module):
    """
    Maintains setpoints for key cognitive drives and computes error signals.

    Drive dimensions:
        [0] arousal     — target: moderate stimulation
        [1] energy      — target: full (decreases with computation cost)
        [2] safety      — target: high (decreases with risky actions)
        [3] engagement  — target: moderate (decreases with failure, increases with success)
        [4] curiosity   — target: moderate novelty-seeking
        [5] coherence   — target: high (decreases when world model is wrong)

    The error signal (strain) drives goal generation in the cerebrum:
    large strain on a dimension → that drive becomes the primary goal.
    """

    DIM_NAMES = ["arousal", "energy", "safety", "engagement", "curiosity", "coherence"]
    N_DIMS = len(DIM_NAMES)

    def __init__(
        self,
        initial: Tuple[float, ...] = (0.5, 0.8, 1.0, 0.5, 0.5, 0.7),
        target: Tuple[float, ...] = (0.6, 0.9, 0.8, 0.6, 0.5, 0.9),
    ) -> None:
        super().__init__()
        assert len(initial) == self.N_DIMS and len(target) == self.N_DIMS
        self._state = nn.Parameter(torch.tensor(list(initial), dtype=torch.float32), requires_grad=False)
        self._target = torch.tensor(list(target), dtype=torch.float32)

    @property
    def state(self) -> torch.Tensor:
        return self._state.data

    @property
    def target(self) -> torch.Tensor:
        return self._target.to(self._state.device)

    def update(self, deltas: Dict[str, float]) -> None:
        """
        Apply named delta updates to drive dimensions.

        Args:
            deltas: Dict mapping dimension name → signed delta value.
                    E.g. {"energy": -0.05, "curiosity": +0.1}
        """
        for name, delta in deltas.items():
            if name in self.DIM_NAMES:
                idx = self.DIM_NAMES.index(name)
                self._state.data[idx] = torch.clamp(
                    self._state.data[idx] + delta, 0.0, 1.0
                )

    def strain(self) -> torch.Tensor:
        """L2 distance from target setpoint — overall homeostatic error."""
        return torch.norm(self._state.data - self.target)

    def per_dim_error(self) -> Dict[str, float]:
        """Signed error per dimension (target - actual). Positive = deficit."""
        errors = (self.target - self._state.data).tolist()
        return dict(zip(self.DIM_NAMES, errors))

    def most_urgent_drive(self) -> str:
        """Return the dimension with the largest absolute deficit."""
        errors = self.per_dim_error()
        return max(errors, key=lambda k: abs(errors[k]))

    def as_vector(self) -> torch.Tensor:
        """Return the full state vector for injection into other modules."""
        return self._state.data.clone()

    def status(self) -> Dict[str, float]:
        state = self._state.data.tolist()
        return dict(zip(self.DIM_NAMES, state))


__all__ = ["HomeostaticRegulator"]
