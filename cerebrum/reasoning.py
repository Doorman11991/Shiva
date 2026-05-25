"""
cerebrum/reasoning.py — Explicit multi-step reasoning and planning.

The prefrontal cortex (part of the cerebrum) handles deliberate,
sequential reasoning — planning ahead, considering alternatives,
and backtracking when a plan fails. This module implements chain-of-
thought reasoning in latent space: a sequence of latent "thought steps"
that refine the conscious latent before action selection.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class LatentThoughtStep(nn.Module):
    """
    One step of latent-space reasoning.

    Applies a learned transformation to the current thought vector,
    conditioned on the goal and working memory context.

        z_{t+1} = z_t + gate · f(z_t, z_goal, z_wm)

    The residual connection ensures each step makes a small, directed
    refinement rather than a large jump.
    """

    def __init__(self, latent_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim * 3, latent_dim * 2),
            nn.GELU(),
            nn.Linear(latent_dim * 2, latent_dim),
        )
        self.gate = nn.Sequential(
            nn.Linear(latent_dim, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        z: torch.Tensor,
        z_goal: torch.Tensor,
        z_wm: torch.Tensor,
    ) -> torch.Tensor:
        combined = torch.cat([z, z_goal, z_wm], dim=-1)
        delta = self.net(combined)
        gate = self.gate(z)
        return z + gate * delta


class ReasoningChain(nn.Module):
    """
    Multi-step latent reasoning chain.

    Applies N thought steps to refine the conscious latent before
    action selection. Each step is conditioned on the current goal
    and working memory context.

    This is "System 2 thinking" — slow, deliberate, activated when
    the meta-cognition monitor detects low confidence.

    Args:
        latent_dim:  Latent dimensionality.
        n_steps:     Number of reasoning steps.
        dropout:     Dropout between steps.
    """

    def __init__(
        self,
        latent_dim: int = 512,
        n_steps: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.n_steps = n_steps
        self.steps = nn.ModuleList([
            LatentThoughtStep(latent_dim) for _ in range(n_steps)
        ])
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(latent_dim)

    def forward(
        self,
        z_conscious: torch.Tensor,
        z_goal: Optional[torch.Tensor] = None,
        z_wm: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Run the reasoning chain.

        Args:
            z_conscious: (B, D) starting conscious latent.
            z_goal:      (B, D) or (D,) goal latent. Zeros if no active goal.
            z_wm:        (B, D) working memory context. Zeros if empty.

        Returns:
            (z_refined, thought_trace) where thought_trace is the list of
            intermediate latents for interpretability.
        """
        B, D = z_conscious.shape
        device = z_conscious.device

        if z_goal is None:
            z_goal = torch.zeros(B, D, device=device)
        elif z_goal.dim() == 1:
            z_goal = z_goal.unsqueeze(0).expand(B, -1)

        if z_wm is None:
            z_wm = torch.zeros(B, D, device=device)
        elif z_wm.dim() == 1:
            z_wm = z_wm.unsqueeze(0).expand(B, -1)

        z = z_conscious
        trace = [z]

        for step in self.steps:
            z = self.dropout(step(z, z_goal, z_wm))
            z = self.norm(z)
            trace.append(z)

        return z, trace


class PlanEvaluator(nn.Module):
    """
    Evaluates the expected value of a latent plan trajectory.

    Used during deliberation to select the best candidate action
    from a set of world model rollouts.

    Args:
        latent_dim: Latent dimensionality.
    """

    def __init__(self, latent_dim: int = 512) -> None:
        super().__init__()
        self.value_net = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.GELU(),
            nn.Linear(256, 1),
        )

    def evaluate_trajectory(self, trajectory: torch.Tensor) -> torch.Tensor:
        """
        Evaluate a predicted trajectory.

        Args:
            trajectory: (B, T, D) predicted latent trajectory.

        Returns:
            (B, 1) expected value of the trajectory.
        """
        # Mean-pool over time, then evaluate
        z_mean = trajectory.mean(dim=1)
        return self.value_net(z_mean)

    def select_best_action(
        self,
        z_current: torch.Tensor,
        candidate_actions: torch.Tensor,
        world_model: nn.Module,
    ) -> Tuple[torch.Tensor, int]:
        """
        Select the best action from candidates using world model rollouts.

        Args:
            z_current:         (1, D) current latent state.
            candidate_actions: (K, A) K candidate actions.
            world_model:       LatentDynamicsModel for rollouts.

        Returns:
            (best_action, best_idx)
        """
        K = candidate_actions.shape[0]
        values = []

        for k in range(K):
            action = candidate_actions[k].unsqueeze(0)  # (1, A)
            z_next = world_model(z_current, action)
            value = self.value_net(z_next)
            values.append(value.item())

        best_idx = int(torch.tensor(values).argmax().item())
        return candidate_actions[best_idx], best_idx


__all__ = ["ReasoningChain", "LatentThoughtStep", "PlanEvaluator"]
