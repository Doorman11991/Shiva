"""
cerebrum/planner.py — Tree-search planner (MCTS-lite).

When meta-cognition flags low confidence, the brain shouldn't just do one
3-step reasoning chain — it should consider MULTIPLE possible actions,
simulate each through the world model, and pick the best.

This is a lightweight Monte Carlo Tree Search:
    1. Sample K candidate actions from the policy.
    2. For each, roll out H steps through the world model.
    3. Score each trajectory with the PlanEvaluator.
    4. Return the best action (highest trajectory value).

Not full MCTS (no backpropagation of visit counts) — just forward
simulation with branching. Enough to avoid obvious mistakes without
the computational cost of a full search tree.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn


class TreeSearchPlanner:
    """
    Branching forward search using the world model.

    Args:
        world_model:     LatentDynamicsModel for rollouts.
        plan_evaluator:  PlanEvaluator for scoring trajectories.
        action_dim:      Action space dimensionality.
        n_candidates:    Number of candidate actions to branch (K).
        horizon:         Rollout length per candidate (H).
        noise_scale:     Scale of Gaussian noise for action sampling.
    """

    def __init__(
        self,
        world_model: nn.Module,
        plan_evaluator: nn.Module,
        action_dim: int = 4,
        n_candidates: int = 8,
        horizon: int = 5,
        noise_scale: float = 0.5,
    ) -> None:
        self.world_model = world_model
        self.plan_evaluator = plan_evaluator
        self.action_dim = action_dim
        self.n_candidates = n_candidates
        self.horizon = horizon
        self.noise_scale = noise_scale
        self._search_count = 0

    @torch.no_grad()
    def search(
        self,
        z_current: torch.Tensor,
        policy_action: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, float]:
        """
        Run tree search and return the best action.

        Args:
            z_current:      (1, D) or (D,) current latent state.
            policy_action:  (1, A) optional policy's default action.
                            Included as one of the candidates so the search
                            can confirm or override the policy.

        Returns:
            (best_action, best_value) — the highest-scoring candidate.
        """
        if z_current.dim() == 1:
            z_current = z_current.unsqueeze(0)

        device = z_current.device
        D = z_current.shape[-1]

        # Generate K candidate actions
        candidates = torch.randn(self.n_candidates, self.action_dim, device=device) * self.noise_scale
        candidates = torch.tanh(candidates)  # squash to [-1, 1]

        # Include the policy's own action as a candidate (position 0)
        if policy_action is not None:
            pa = policy_action.detach().squeeze(0) if policy_action.dim() > 1 else policy_action.detach()
            candidates[0] = pa

        # Roll out each candidate
        values = []
        for k in range(self.n_candidates):
            action = candidates[k].unsqueeze(0)  # (1, A)
            trajectory = [z_current.squeeze(0)]
            z = z_current
            for h in range(self.horizon):
                a = action if h == 0 else torch.zeros(1, self.action_dim, device=device)
                z = self.world_model(z, a)
                trajectory.append(z.squeeze(0))

            traj_tensor = torch.stack(trajectory).unsqueeze(0)  # (1, H+1, D)
            value = float(self.plan_evaluator.evaluate_trajectory(traj_tensor).item())
            values.append(value)

        # Pick the best
        best_idx = int(torch.tensor(values).argmax().item())
        best_action = candidates[best_idx].unsqueeze(0)
        best_value = values[best_idx]

        self._search_count += 1
        return best_action, best_value

    @property
    def search_count(self) -> int:
        return self._search_count

    def status(self) -> dict:
        return {
            "search_count": self._search_count,
            "n_candidates": self.n_candidates,
            "horizon": self.horizon,
        }


__all__ = ["TreeSearchPlanner"]
