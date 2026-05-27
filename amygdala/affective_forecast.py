"""
amygdala/affective_forecast.py — Predict future valence of a trajectory.

The world model predicts future *states*. This module predicts future
*feelings* — how a trajectory will make the agent feel over time. This
lets the cerebrum pick actions not just by predicted reward, but by
anticipated emotional outcome.

    predicted_valence_trajectory = forecast(z_trajectory)

Use cases:
    - Avoid actions that lead to sustained negative valence (learned anxiety)
    - Prefer actions that lead to engagement (learned motivation)
    - Detect "emotional dead ends" (trajectories that flatten to neutral)
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn


class AffectiveForecaster(nn.Module):
    """
    Predicts the valence trajectory for a sequence of latent states.

    Architecture: small GRU over the trajectory, outputs per-step valence
    predictions in [-1, 1].

    Args:
        latent_dim:  Dimensionality of latent states.
        hidden_dim:  GRU hidden size.
    """

    def __init__(self, latent_dim: int = 512, hidden_dim: int = 128) -> None:
        super().__init__()
        self.gru = nn.GRU(latent_dim, hidden_dim, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Tanh(),
        )

    def forward(self, trajectory: torch.Tensor) -> torch.Tensor:
        """
        Predict valence at each step of a trajectory.

        Args:
            trajectory: (B, T, D) latent state sequence.

        Returns:
            (B, T, 1) predicted valence at each timestep.
        """
        # nn.GRU is not supported on DirectML (privateuseone) — run on CPU.
        # On CUDA/MPS/CPU the GRU runs natively on the device.
        if trajectory.device.type == "privateuseone":
            dev = trajectory.device
            h, _ = self.gru.cpu()(trajectory.to('cpu'))
            h = h.to(dev)
        else:
            h, _ = self.gru(trajectory)
        return self.head(h)               # (B, T, 1)

    def forecast_mean(self, trajectory: torch.Tensor) -> torch.Tensor:
        """
        Predict the average valence over a trajectory.

        Returns:
            (B, 1) mean predicted valence.
        """
        per_step = self.forward(trajectory)
        return per_step.mean(dim=1)

    def forecast_final(self, trajectory: torch.Tensor) -> torch.Tensor:
        """
        Predict the valence at the END of the trajectory.

        Returns:
            (B, 1) final-step valence prediction.
        """
        per_step = self.forward(trajectory)
        return per_step[:, -1, :]

    def compare_trajectories(
        self,
        traj_a: torch.Tensor,
        traj_b: torch.Tensor,
    ) -> Tuple[float, float]:
        """
        Compare two trajectories by predicted emotional outcome.

        Returns:
            (mean_valence_a, mean_valence_b) — higher is emotionally preferred.
        """
        va = float(self.forecast_mean(traj_a).mean().item())
        vb = float(self.forecast_mean(traj_b).mean().item())
        return va, vb

    def emotional_dead_end(
        self,
        trajectory: torch.Tensor,
        threshold: float = 0.05,
    ) -> bool:
        """
        Detect if a trajectory leads to emotional flatness (stagnation).

        A dead end is where predicted valence variance across time is
        near zero AND mean valence is near zero. The agent isn't feeling
        anything — neither good nor bad. Often worse than negative valence
        (which at least indicates engagement).

        Returns:
            True if the trajectory is an emotional dead end.
        """
        per_step = self.forward(trajectory)  # (B, T, 1)
        var = per_step.var(dim=1).mean().item()
        mean = per_step.mean().abs().item()
        return var < threshold and mean < threshold


class AffectiveForecasterTrainer:
    """
    Trains the forecaster on observed (trajectory, actual_valence) pairs.

    Call `update()` with real episodes and their actual valences from the
    emotional core. The forecaster learns to predict future affect.
    """

    def __init__(self, forecaster: AffectiveForecaster, lr: float = 1e-4) -> None:
        self.forecaster = forecaster
        self.optimizer = torch.optim.Adam(forecaster.parameters(), lr=lr)
        self._step = 0

    def update(
        self,
        trajectory: torch.Tensor,
        actual_valences: torch.Tensor,
    ) -> float:
        """
        One training step.

        Args:
            trajectory:      (B, T, D) latent state sequences.
            actual_valences: (B, T, 1) actual valences observed.

        Returns:
            Scalar MSE loss.
        """
        self.optimizer.zero_grad()
        predicted = self.forecaster(trajectory)
        loss = nn.functional.mse_loss(predicted, actual_valences.detach())
        loss.backward()
        self.optimizer.step()
        self._step += 1
        return loss.item()


__all__ = ["AffectiveForecaster", "AffectiveForecasterTrainer"]
