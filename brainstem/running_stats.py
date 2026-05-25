"""
brainstem/running_stats.py — Reward normalisation (blood pressure regulation).

The brainstem regulates blood pressure — keeping it in a stable range
regardless of external conditions. This module does the same for reward
signals: Welford's algorithm tracks running mean/variance so the critic
always sees a normalised reward distribution, preventing loss explosions
when the task mix changes dramatically.

Moved from: core/running_stats.py
"""

from __future__ import annotations

from typing import Iterable, Union

import torch


Tensorish = Union[torch.Tensor, float, Iterable[float]]


class RunningMeanStd:
    """
    Tracks the running mean and variance of a 1-D scalar stream.

    Uses the parallel-update (Chan et al. 1979) form for numerical
    stability and unbiased variance estimation.
    """

    __slots__ = ("mean", "var", "count", "epsilon")

    def __init__(self, epsilon: float = 1e-4) -> None:
        self.mean: float = 0.0
        self.var: float = 1.0
        self.count: float = epsilon
        self.epsilon = epsilon

    def update(self, x: Tensorish) -> None:
        if isinstance(x, torch.Tensor):
            arr = x.detach().reshape(-1).float().cpu()
        elif hasattr(x, "__iter__"):
            arr = torch.tensor(list(x), dtype=torch.float32)
        else:
            arr = torch.tensor([float(x)], dtype=torch.float32)

        if arr.numel() == 0:
            return

        batch_mean = arr.mean().item()
        batch_var = arr.var(unbiased=False).item() if arr.numel() > 1 else 0.0
        batch_count = arr.numel()
        self._merge(batch_mean, batch_var, batch_count)

    def _merge(self, b_mean: float, b_var: float, b_count: int) -> None:
        delta = b_mean - self.mean
        tot = self.count + b_count

        new_mean = self.mean + delta * (b_count / tot)
        m_a = self.var * self.count
        m_b = b_var * b_count
        new_var = (m_a + m_b + delta * delta * self.count * b_count / tot) / tot

        self.mean = new_mean
        self.var = max(new_var, 0.0)
        self.count = tot

    @property
    def std(self) -> float:
        return float(self.var) ** 0.5

    def normalise(self, x: torch.Tensor) -> torch.Tensor:
        """Return (x - mean) / (std + eps), clipped to ±10 to bound outliers."""
        denom = self.std + 1e-6
        return torch.clamp((x - self.mean) / denom, -10.0, 10.0)


__all__ = ["RunningMeanStd"]
