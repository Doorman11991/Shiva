"""
hypothalamus/entropy_temperature.py — SAC entropy temperature regulation.

Body temperature is a hypothalamic setpoint — too hot or too cold and
the body triggers corrective responses. The SAC entropy coefficient α
is the cognitive equivalent: too low and the agent becomes deterministic
(overheated, rigid); too high and it becomes random (hypothermic, chaotic).

This module manages α as a homeostatic variable with a target entropy
setpoint, separate from the brainstem's raw optimizer update.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class EntropyTemperatureRegulator(nn.Module):
    """
    Homeostatic manager for the SAC entropy temperature coefficient α.

    Wraps the standard SAC auto-tuning (Haarnoja 2018b) with additional
    homeostatic constraints from the hypothalamus:
        - If engagement drive is low → raise α (explore more)
        - If safety drive is low → lower α (be more conservative)
        - If energy is low → lower α (reduce compute cost of exploration)

    The base SAC update is:
        L(α) = -α · (log π(a|s) + H_target)

    The homeostatic correction adds a soft penalty toward the drive-derived
    target entropy, blended by a homeostasis_weight hyperparameter.

    Args:
        action_dim:           Action space dimensionality.
        lr:                   Learning rate for α optimizer.
        homeostasis_weight:   How strongly drive signals override SAC target.
        alpha_min:            Hard floor on α (prevents total determinism).
        alpha_max:            Hard ceiling on α (prevents total randomness).
    """

    def __init__(
        self,
        action_dim: int,
        lr: float = 3e-4,
        homeostasis_weight: float = 0.1,
        alpha_min: float = 0.001,
        alpha_max: float = 1.0,
    ) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.homeostasis_weight = homeostasis_weight
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max

        # SAC standard: target entropy = -|A|
        self.base_target_entropy = float(-action_dim)

        self.log_alpha = nn.Parameter(torch.zeros(1))
        self.optimizer = torch.optim.Adam([self.log_alpha], lr=lr)

    @property
    def alpha(self) -> torch.Tensor:
        """Current α, clamped to [alpha_min, alpha_max]."""
        return torch.clamp(self.log_alpha.exp(), self.alpha_min, self.alpha_max).detach()

    def effective_target_entropy(
        self,
        engagement: float = 0.6,
        safety: float = 0.8,
        energy: float = 0.9,
    ) -> float:
        """
        Compute the drive-adjusted target entropy.

        Low engagement → more exploration (higher entropy target).
        Low safety → less exploration (lower entropy target).
        Low energy → less exploration (lower entropy target).
        """
        # Engagement deficit → want more exploration
        engagement_correction = (0.6 - engagement) * 2.0  # positive when low
        # Safety deficit → want less exploration
        safety_correction = -(0.8 - safety) * 3.0         # negative when low
        # Energy deficit → want less exploration
        energy_correction = -(0.9 - energy) * 1.0

        drive_correction = engagement_correction + safety_correction + energy_correction
        return self.base_target_entropy + self.homeostasis_weight * drive_correction

    def update(
        self,
        log_probs: torch.Tensor,
        engagement: float = 0.6,
        safety: float = 0.8,
        energy: float = 0.9,
    ) -> float:
        """
        One α update step.

        Args:
            log_probs:   (B, 1) log-probabilities from the current policy.
            engagement:  Current engagement drive level [0, 1].
            safety:      Current safety drive level [0, 1].
            energy:      Current energy level [0, 1].

        Returns:
            Scalar alpha loss value.
        """
        target_entropy = self.effective_target_entropy(engagement, safety, energy)
        alpha_loss = -(self.log_alpha * (log_probs.detach() + target_entropy)).mean()

        self.optimizer.zero_grad()
        alpha_loss.backward()
        self.optimizer.step()

        # Hard clamp after update
        with torch.no_grad():
            self.log_alpha.clamp_(
                min=torch.log(torch.tensor(self.alpha_min)),
                max=torch.log(torch.tensor(self.alpha_max)),
            )

        return alpha_loss.item()

    def status(self) -> dict:
        return {
            "alpha": float(self.alpha.item()),
            "log_alpha": float(self.log_alpha.item()),
            "base_target_entropy": self.base_target_entropy,
        }


__all__ = ["EntropyTemperatureRegulator"]
