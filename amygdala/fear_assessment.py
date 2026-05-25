"""
amygdala/fear_assessment.py — Risk and threat evaluation.

The amygdala is the brain's threat detector — it can trigger a fear
response faster than conscious thought, bypassing the cortex entirely.
This module evaluates the risk of proposed actions and can veto them
before execution via the fast-path signal to the cerebellum.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn


class FearAssessor(nn.Module):
    """
    Evaluates the risk/threat level of a proposed action given the current state.

    Risk model:
        risk = σ(W · [z_conscious ‖ action])

    A risk score above the veto_threshold triggers a fear_veto signal
    to the cerebellum, blocking the action before execution.

    The assessor also maintains a running estimate of "safe zones" in
    latent space — regions where actions have historically been safe.

    Args:
        d_model:          Latent dimensionality.
        action_dim:       Action space dimensionality.
        veto_threshold:   Risk score above which actions are vetoed.
        safe_zone_decay:  EMA decay for safe zone estimation.
    """

    def __init__(
        self,
        d_model: int,
        action_dim: int,
        veto_threshold: float = 0.8,
        safe_zone_decay: float = 0.99,
    ) -> None:
        super().__init__()
        self.veto_threshold = veto_threshold
        self.safe_zone_decay = safe_zone_decay

        # Risk network: maps (state, action) → risk score in [0, 1]
        self.risk_net = nn.Sequential(
            nn.Linear(d_model + action_dim, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, 1),
            nn.Sigmoid(),
        )

        # Safe zone: EMA of latent states where actions were safe
        self.register_buffer("safe_zone_centroid", torch.zeros(d_model))
        self.register_buffer("safe_zone_radius", torch.tensor(1.0))
        self._safe_zone_initialized = False

    def assess(
        self,
        z_conscious: torch.Tensor,
        action: torch.Tensor,
    ) -> Tuple[torch.Tensor, bool]:
        """
        Assess the risk of taking `action` from state `z_conscious`.

        Args:
            z_conscious: (B, D) conscious latent state.
            action:      (B, action_dim) proposed action.

        Returns:
            risk_score: (B, 1) risk in [0, 1].
            veto:       True if any sample exceeds veto_threshold.
        """
        sa = torch.cat([z_conscious, action], dim=-1)
        risk_score = self.risk_net(sa)
        veto = bool(risk_score.max().item() > self.veto_threshold)
        return risk_score, veto

    def update_safe_zone(
        self,
        z_conscious: torch.Tensor,
        was_safe: bool,
    ) -> None:
        """
        Update the safe zone estimate based on observed outcome.

        Args:
            z_conscious: (B, D) latent state of the episode.
            was_safe:    Whether the action taken was safe.
        """
        if not was_safe:
            return

        centroid = z_conscious.mean(dim=0).detach()
        if not self._safe_zone_initialized:
            self.safe_zone_centroid.copy_(centroid)
            self._safe_zone_initialized = True
        else:
            self.safe_zone_centroid.mul_(self.safe_zone_decay).add_(
                centroid * (1 - self.safe_zone_decay)
            )

        # Update radius as mean distance from centroid
        dists = torch.norm(z_conscious - self.safe_zone_centroid.unsqueeze(0), dim=-1)
        self.safe_zone_radius = dists.mean().detach()

    def in_safe_zone(self, z_conscious: torch.Tensor) -> torch.Tensor:
        """
        Returns a (B,) boolean tensor: True if each sample is within the safe zone.
        """
        if not self._safe_zone_initialized:
            return torch.ones(z_conscious.shape[0], dtype=torch.bool)
        dists = torch.norm(z_conscious - self.safe_zone_centroid.unsqueeze(0), dim=-1)
        return dists <= self.safe_zone_radius * 1.5  # 50% margin


__all__ = ["FearAssessor"]
