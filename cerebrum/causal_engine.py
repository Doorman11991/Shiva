"""
cerebrum/causal_engine.py — Causal reasoning and counterfactual planning.

The prefrontal cortex (part of the cerebrum) supports causal reasoning —
understanding not just *what* happened but *why*, and asking "what would
have happened if I had done X instead?" This module implements a lightweight
structural causal model over the latent space.

Biological analogy
~~~~~~~~~~~~~~~~~~
Humans don't just learn correlations — we build causal models of the world.
When something goes wrong, we trace back the cause. When planning, we
simulate interventions: "if I do A, then B will follow." This is the
computational equivalent.

Design
~~~~~~
Rather than a full symbolic causal graph (expensive, brittle), we use a
learned approach:

  1. CausalAttributor  — given (z_before, z_after, action), scores which
                         action dimensions caused the observed state change.
  2. CounterfactualEngine — given (z_current, action_taken, action_alt),
                            predicts what z would have been under the
                            alternative action using the world model.
  3. CausalGraph       — maintains a soft adjacency matrix over latent
                         dimensions, updated from observed transitions.
                         Lets the agent ask "does dimension i influence j?"
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# CausalAttributor
# ---------------------------------------------------------------------------

class CausalAttributor(nn.Module):
    """
    Attributes observed state changes to specific action dimensions.

    Given a transition (z_before → z_after) and the action taken, it
    scores how much each action dimension contributed to the change.

    This is used for:
        - Post-hoc explanation: "why did the state change this way?"
        - Credit assignment: which part of the action was responsible?
        - Debugging: detecting when actions have unintended side effects.

    Architecture:
        Input:  [z_before ‖ z_after ‖ action]  →  (D + D + A,)
        Output: attribution scores per action dim  →  (A,) in [0, 1]

    Args:
        latent_dim:  Dimensionality of the latent state space.
        action_dim:  Dimensionality of the action space.
    """

    def __init__(self, latent_dim: int = 512, action_dim: int = 4) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.action_dim = action_dim

        self.net = nn.Sequential(
            nn.Linear(latent_dim * 2 + action_dim, 256),
            nn.GELU(),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, action_dim),
            nn.Sigmoid(),   # attribution scores in [0, 1]
        )

    def forward(
        self,
        z_before: torch.Tensor,
        z_after: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute attribution scores.

        Args:
            z_before: (B, D) state before the action.
            z_after:  (B, D) state after the action.
            action:   (B, A) action taken.

        Returns:
            (B, A) attribution scores — how much each action dim
            contributed to the observed state change.
        """
        x = torch.cat([z_before, z_after, action], dim=-1)
        return self.net(x)

    def top_causes(
        self,
        z_before: torch.Tensor,
        z_after: torch.Tensor,
        action: torch.Tensor,
        k: int = 2,
    ) -> List[Tuple[int, float]]:
        """
        Return the top-k action dimensions most responsible for the change.

        Returns:
            List of (action_dim_index, attribution_score) sorted descending.
        """
        scores = self.forward(z_before, z_after, action)
        scores_mean = scores.mean(dim=0)  # average over batch
        top_vals, top_idx = scores_mean.topk(min(k, self.action_dim))
        return [(int(i), float(v)) for i, v in zip(top_idx.tolist(), top_vals.tolist())]


# ---------------------------------------------------------------------------
# CounterfactualEngine
# ---------------------------------------------------------------------------

class CounterfactualEngine(nn.Module):
    """
    Answers "what would have happened if I had done X instead?"

    Uses the world model to simulate the counterfactual trajectory and
    computes the regret: how much better/worse the alternative would have been.

    This enables:
        - Learning from near-misses: "I almost did the right thing."
        - Regret-based exploration: seek states where counterfactuals
          would have been much better (high regret = unexplored potential).
        - Policy improvement: use counterfactual Q-values as additional
          training signal.

    Args:
        latent_dim:  Latent dimensionality.
        action_dim:  Action space dimensionality.
    """

    def __init__(self, latent_dim: int = 512, action_dim: int = 4) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.action_dim = action_dim

        # Value estimator: how good is a given latent state?
        self.value_net = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.GELU(),
            nn.Linear(256, 1),
        )

    def counterfactual_state(
        self,
        z_current: torch.Tensor,
        action_taken: torch.Tensor,
        action_alt: torch.Tensor,
        world_model: nn.Module,
    ) -> torch.Tensor:
        """
        Predict the state that would have resulted from action_alt.

        Args:
            z_current:   (B, D) current latent state.
            action_taken: (B, A) action that was actually taken.
            action_alt:  (B, A) alternative action to evaluate.
            world_model: LatentDynamicsModel for rollouts.

        Returns:
            (B, D) predicted counterfactual next state.
        """
        return world_model(z_current, action_alt)

    def regret(
        self,
        z_current: torch.Tensor,
        action_taken: torch.Tensor,
        action_alt: torch.Tensor,
        world_model: nn.Module,
    ) -> torch.Tensor:
        """
        Compute regret: V(counterfactual) - V(actual).

        Positive regret means the alternative would have been better.
        Negative regret means the taken action was better.

        Returns:
            (B, 1) regret scalar per sample.
        """
        z_actual = world_model(z_current, action_taken)
        z_counter = world_model(z_current, action_alt)

        v_actual = self.value_net(z_actual)
        v_counter = self.value_net(z_counter)

        return v_counter - v_actual

    def best_alternative(
        self,
        z_current: torch.Tensor,
        action_taken: torch.Tensor,
        candidate_actions: torch.Tensor,
        world_model: nn.Module,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Find the best alternative action from a set of candidates.

        Args:
            z_current:         (1, D) current state.
            action_taken:      (1, A) action that was taken.
            candidate_actions: (K, A) alternative actions to evaluate.
            world_model:       LatentDynamicsModel.

        Returns:
            (best_action, regret_score) — the best alternative and its regret.
        """
        K = candidate_actions.shape[0]
        regrets = []
        for k in range(K):
            alt = candidate_actions[k].unsqueeze(0)
            r = self.regret(z_current, action_taken, alt, world_model)
            regrets.append(r.item())

        best_k = int(torch.tensor(regrets).argmax().item())
        best_regret = torch.tensor([[regrets[best_k]]])
        return candidate_actions[best_k], best_regret


# ---------------------------------------------------------------------------
# CausalGraph
# ---------------------------------------------------------------------------

class CausalGraph(nn.Module):
    """
    Soft causal adjacency matrix over latent dimensions.

    Maintains a learned matrix A where A[i, j] ∈ [0, 1] represents
    the estimated causal influence of latent dimension i on dimension j.

    Updated online from observed transitions using a simple heuristic:
        If changing dimension i in the action correlates with changes in
        dimension j of the next state, A[i, j] increases.

    This is a lightweight approximation — not a full NOTEARS/PC algorithm —
    but it gives the cerebrum a queryable causal structure without the
    computational cost of full causal discovery.

    Args:
        n_dims:      Number of dimensions to track (typically action_dim
                     for action→state causality, or a compressed latent dim).
        decay:       EMA decay for the adjacency matrix updates.
        threshold:   Minimum edge weight to report as a causal link.
    """

    def __init__(
        self,
        n_dims: int = 16,
        decay: float = 0.99,
        threshold: float = 0.3,
    ) -> None:
        super().__init__()
        self.n_dims = n_dims
        self.decay = decay
        self.threshold = threshold

        # Soft adjacency matrix — starts uniform (no prior causal knowledge)
        self.register_buffer(
            "adjacency",
            torch.ones(n_dims, n_dims) / n_dims,
        )
        self._update_count = 0

    def update(
        self,
        cause_activations: torch.Tensor,
        effect_activations: torch.Tensor,
    ) -> None:
        """
        Update the causal graph from observed (cause, effect) activations.

        Args:
            cause_activations:  (B, n_dims) — e.g. action dimensions.
            effect_activations: (B, n_dims) — e.g. state change dimensions.
        """
        # Outer product: how much does each cause co-vary with each effect?
        # Mean over batch to get a (n_dims, n_dims) co-activation matrix.
        cause_norm = F.normalize(cause_activations.abs(), p=1, dim=-1)   # (B, n_dims)
        effect_norm = F.normalize(effect_activations.abs(), p=1, dim=-1) # (B, n_dims)

        co_activation = (cause_norm.unsqueeze(2) * effect_norm.unsqueeze(1)).mean(dim=0)  # (n_dims, n_dims)

        # EMA update
        self.adjacency = self.decay * self.adjacency + (1 - self.decay) * co_activation
        self._update_count += 1

    def causes_of(self, effect_dim: int, top_k: int = 3) -> List[Tuple[int, float]]:
        """
        Return the top-k causal drivers of a given effect dimension.

        Args:
            effect_dim: Index of the effect dimension.
            top_k:      Number of top causes to return.

        Returns:
            List of (cause_dim_index, strength) sorted by strength descending.
        """
        col = self.adjacency[:, effect_dim]
        k = min(top_k, self.n_dims)
        vals, idx = col.topk(k)
        return [(int(i), float(v)) for i, v in zip(idx.tolist(), vals.tolist())]

    def effects_of(self, cause_dim: int, top_k: int = 3) -> List[Tuple[int, float]]:
        """
        Return the top-k effects driven by a given cause dimension.
        """
        row = self.adjacency[cause_dim, :]
        k = min(top_k, self.n_dims)
        vals, idx = row.topk(k)
        return [(int(i), float(v)) for i, v in zip(idx.tolist(), vals.tolist())]

    def strong_edges(self) -> List[Tuple[int, int, float]]:
        """Return all edges above the threshold as (cause, effect, weight)."""
        edges = []
        for i in range(self.n_dims):
            for j in range(self.n_dims):
                w = float(self.adjacency[i, j].item())
                if w >= self.threshold:
                    edges.append((i, j, w))
        return sorted(edges, key=lambda e: e[2], reverse=True)

    def is_causal(self, cause_dim: int, effect_dim: int) -> bool:
        """Quick check: is there a strong causal link from cause to effect?"""
        return float(self.adjacency[cause_dim, effect_dim].item()) >= self.threshold

    def status(self) -> Dict:
        return {
            "n_dims": self.n_dims,
            "update_count": self._update_count,
            "n_strong_edges": len(self.strong_edges()),
            "mean_edge_weight": float(self.adjacency.mean().item()),
        }


# ---------------------------------------------------------------------------
# CausalEngine — top-level orchestrator
# ---------------------------------------------------------------------------

class CausalEngine(nn.Module):
    """
    Orchestrates causal attribution, counterfactual reasoning, and the
    causal graph. The cerebrum's "why" module.

    Exposes a clean interface for the rest of the brain:
        - attribute(z_before, z_after, action)  → what caused this?
        - regret(z, a_taken, a_alt, wm)         → was there a better choice?
        - best_alt(z, a_taken, candidates, wm)  → what should I have done?
        - update_graph(causes, effects)          → learn causal structure

    Args:
        latent_dim:  Latent dimensionality.
        action_dim:  Action space dimensionality.
        graph_dims:  Number of dimensions tracked in the causal graph.
                     Defaults to action_dim (action→state causality).
    """

    def __init__(
        self,
        latent_dim: int = 512,
        action_dim: int = 4,
        graph_dims: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.action_dim = action_dim

        self.attributor = CausalAttributor(latent_dim, action_dim)
        self.counterfactual = CounterfactualEngine(latent_dim, action_dim)
        self.graph = CausalGraph(n_dims=graph_dims or action_dim)

    def attribute(
        self,
        z_before: torch.Tensor,
        z_after: torch.Tensor,
        action: torch.Tensor,
        top_k: int = 2,
    ) -> List[Tuple[int, float]]:
        """What caused the observed state change? Returns top-k action dims."""
        return self.attributor.top_causes(z_before, z_after, action, k=top_k)

    def regret(
        self,
        z_current: torch.Tensor,
        action_taken: torch.Tensor,
        action_alt: torch.Tensor,
        world_model: nn.Module,
    ) -> float:
        """Was there a better choice? Returns scalar regret (positive = yes)."""
        r = self.counterfactual.regret(z_current, action_taken, action_alt, world_model)
        return float(r.mean().item())

    def best_alternative(
        self,
        z_current: torch.Tensor,
        action_taken: torch.Tensor,
        candidate_actions: torch.Tensor,
        world_model: nn.Module,
    ) -> Tuple[torch.Tensor, float]:
        """Find the best alternative action. Returns (action, regret_score)."""
        best, regret_t = self.counterfactual.best_alternative(
            z_current, action_taken, candidate_actions, world_model
        )
        return best, float(regret_t.item())

    def update_graph(
        self,
        action: torch.Tensor,
        state_delta: torch.Tensor,
    ) -> None:
        """
        Update the causal graph from an observed (action, state_change) pair.

        Args:
            action:       (B, action_dim) actions taken.
            state_delta:  (B, action_dim) or (B, n_dims) observed state changes.
                          If latent_dim > graph_dims, caller should project first.
        """
        # Clamp to graph dims if needed
        n = self.graph.n_dims
        a = action[:, :n] if action.shape[-1] > n else action
        d = state_delta[:, :n] if state_delta.shape[-1] > n else state_delta
        self.graph.update(a, d)

    def status(self) -> Dict:
        return {
            "graph": self.graph.status(),
        }


__all__ = [
    "CausalEngine",
    "CausalAttributor",
    "CounterfactualEngine",
    "CausalGraph",
]
