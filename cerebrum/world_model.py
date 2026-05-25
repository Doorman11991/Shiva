"""
cerebrum/world_model.py — Predictive latent dynamics model.

The cerebrum's ability to imagine — to simulate consequences without
acting — is the foundation of planning, deliberation, and counterfactual
reasoning. This module implements a learned dynamics model that predicts
the next latent state given the current state and action.

    ẑ_{t+1} = WorldModel(z_t, a_t)

Trained on replay buffer transitions. Enables:
    - Mental simulation (rollouts in latent space)
    - Curiosity rewards (prediction error = novelty)
    - Counterfactual reasoning ("what if I had done X?")
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class LatentDynamicsModel(nn.Module):
    """
    Predicts the next latent state given current state and action.

        ẑ_{t+1} = f(z_t, a_t)

    Architecture: 3-layer MLP with residual connection.
    The residual connection biases the model toward predicting small
    changes (most transitions are smooth), which improves early training.

    Args:
        latent_dim:  Dimensionality of the latent state space.
        action_dim:  Dimensionality of the action space.
        hidden_dim:  Hidden layer width.
    """

    def __init__(
        self,
        latent_dim: int = 512,
        action_dim: int = 4,
        hidden_dim: int = 512,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.action_dim = action_dim

        self.net = nn.Sequential(
            nn.Linear(latent_dim + action_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        # Zero-init last layer: start by predicting no change (residual)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, z: torch.Tensor, action: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Predict next latent state.

        Args:
            z:      (B, D) current latent state.
            action: (B, A) action taken. If None, predicts passive dynamics.

        Returns:
            (B, D) predicted next latent state.
        """
        if action is None:
            action = torch.zeros(z.shape[0], self.action_dim, device=z.device)
        za = torch.cat([z, action], dim=-1)
        delta = self.net(za)
        return z + delta  # residual: predict change, not absolute state

    def simulate(
        self,
        z_start: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        """
        Roll out a trajectory in latent space.

        Args:
            z_start: (B, D) starting latent state.
            actions: (B, T, A) action sequence.

        Returns:
            (B, T+1, D) predicted trajectory including start state.
        """
        B, T, A = actions.shape
        trajectory = [z_start]
        z = z_start
        for t in range(T):
            z = self.forward(z, actions[:, t, :])
            trajectory.append(z)
        return torch.stack(trajectory, dim=1)  # (B, T+1, D)

    def prediction_error(
        self,
        z_current: torch.Tensor,
        action: torch.Tensor,
        z_next_actual: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute prediction error (used as curiosity reward).

        Returns:
            (B, 1) L2 prediction error per sample.
        """
        z_predicted = self.forward(z_current, action)
        return torch.norm(z_predicted - z_next_actual, p=2, dim=-1, keepdim=True)


class WorldModelTrainer:
    """
    Trains the LatentDynamicsModel on replay buffer transitions.

    Args:
        model:  The LatentDynamicsModel to train.
        lr:     Adam learning rate.
    """

    def __init__(self, model: LatentDynamicsModel, lr: float = 3e-4) -> None:
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self._step = 0

    def update(
        self,
        z_current: torch.Tensor,
        actions: torch.Tensor,
        z_next: torch.Tensor,
    ) -> float:
        """
        One gradient step on a batch of transitions.

        IMPORTANT: Inputs are always .detach()'d before use. This ensures
        the world model optimizer NEVER sends gradients back through the
        backbone (which lives upstream and is owned by the policy optimizer).
        Without this stop-gradient, two optimizers would fight over the
        backbone's representation — the world model wanting features for
        prediction and the policy wanting features for action selection.

        Args:
            z_current: (B, D) current latent states.
            actions:   (B, A) actions taken.
            z_next:    (B, D) actual next latent states.

        Returns:
            Scalar MSE loss.
        """
        # Stop-gradient: detach inputs so backbone is frozen from WM's perspective.
        z_current = z_current.detach()
        actions = actions.detach()
        z_next = z_next.detach()

        self.optimizer.zero_grad()
        z_pred = self.model(z_current, actions)
        loss = F.mse_loss(z_pred, z_next)
        loss.backward()
        self.optimizer.step()
        self._step += 1
        return loss.item()


__all__ = ["LatentDynamicsModel", "WorldModelTrainer"]
