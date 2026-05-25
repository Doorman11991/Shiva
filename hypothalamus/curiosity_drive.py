"""
hypothalamus/curiosity_drive.py — Prediction-error driven exploration hunger.

Hunger is a hypothalamic drive — the body signals a deficit and motivates
behaviour to correct it. Curiosity is the cognitive equivalent: when the
world model is surprised (high prediction error), the agent is "hungry"
for more information about that region of state space.

This module computes intrinsic curiosity rewards from world model prediction
error and decays them as the model improves, so novelty-seeking naturally
moves to new frontiers.
"""

from __future__ import annotations

from collections import deque
from typing import Optional

import torch
import torch.nn as nn


class CuriosityDrive(nn.Module):
    """
    Intrinsic motivation from world model prediction error.

    Curiosity reward:
        r_curiosity = β · ||ẑ_{t+1} - z_{t+1}||₂

    Where:
        ẑ_{t+1} = world model prediction
        z_{t+1} = actual next latent state
        β       = curiosity scale (annealed over time)

    The scale β decays as the world model improves on a topic, so the
    agent naturally moves on to explore new areas rather than obsessing
    over already-understood regions.

    Args:
        latent_dim:     Dimensionality of the latent space.
        beta_init:      Initial curiosity scale.
        beta_min:       Minimum curiosity scale (floor).
        decay_rate:     Multiplicative decay per update step.
        history_len:    Window for running mean prediction error.
    """

    def __init__(
        self,
        latent_dim: int = 512,
        beta_init: float = 1.0,
        beta_min: float = 0.01,
        decay_rate: float = 0.9999,
        history_len: int = 1000,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.beta_min = beta_min
        self.decay_rate = decay_rate
        self._beta = beta_init
        self._error_history: deque = deque(maxlen=history_len)
        self._step = 0

    @property
    def beta(self) -> float:
        return self._beta

    def compute_reward(
        self,
        predicted_next: torch.Tensor,
        actual_next: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute per-sample curiosity reward from prediction error.

        Args:
            predicted_next: (B, D) world model prediction of next latent.
            actual_next:    (B, D) actual observed next latent.

        Returns:
            (B, 1) curiosity reward tensor.
        """
        error = torch.norm(predicted_next - actual_next, p=2, dim=-1, keepdim=True)
        reward = self._beta * error

        # Track mean error for diagnostics
        mean_err = error.mean().item()
        self._error_history.append(mean_err)

        return reward

    def step(self) -> None:
        """Decay beta. Call once per training update."""
        self._beta = max(self.beta_min, self._beta * self.decay_rate)
        self._step += 1

    def mean_recent_error(self) -> float:
        """Running mean prediction error over the history window."""
        if not self._error_history:
            return 0.0
        return sum(self._error_history) / len(self._error_history)

    def is_curious(self, threshold: float = 0.1) -> bool:
        """True if recent mean error is above threshold (agent is still learning)."""
        return self.mean_recent_error() > threshold

    def status(self) -> dict:
        return {
            "beta": self._beta,
            "mean_error": self.mean_recent_error(),
            "step": self._step,
        }


__all__ = ["CuriosityDrive"]
