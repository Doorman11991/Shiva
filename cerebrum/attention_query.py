"""
cerebrum/attention_query.py — Top-down attention query builder.

Biological role
~~~~~~~~~~~~~~~
The cortex sends "expectation" signals back down to the thalamus that
bias what the thalamus passes through. This is corticothalamic feedback,
and it's the reason you notice the thing you're looking for. Without it,
attention is purely stimulus-driven.

Computational role
~~~~~~~~~~~~~~~~~~
The cerebrum maintains a small AttentionQueryBuilder that consumes:
    - the current top goal's target latent (what we want to find)
    - working memory's attended context (what we're currently thinking about)
    - the narrative self-model (who we are)

and produces a single (D,) query vector. The thalamus's AttentionBottleneck
takes this query and biases its salience gate toward tokens that are
cosine-similar to the query.

The query is built each tick *after* cerebrum state updates, then cached
for the *next* tick's thalamus pass. This causal lag mirrors biology —
top-down attention is always shaped by what the cortex was attending to
just before, not what it would attend to in the future.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionQueryBuilder(nn.Module):
    """
    Builds a top-down attention query for the thalamus from cerebrum state.

    Architecture:
        Concatenates [goal_latent, wm_context, self_model] (zeros where
        any component is missing), runs through a small MLP, and L2-
        normalises the output. The result is a single (D,) query vector
        that can be passed to AttentionBottleneck.forward(top_down_query=...).

    Why not just use the goal latent directly?
        Three-way fusion lets the brain attend to features that are
        simultaneously goal-relevant *and* consistent with current
        thinking *and* aligned with identity. A single component would
        miss that intersection.

    Args:
        d_model: Latent dimensionality.
        gate_init: Initial gate value for blending [0,1]. 0.5 starts neutral.
    """

    def __init__(self, d_model: int = 512, gate_init: float = 0.5) -> None:
        super().__init__()
        self.d_model = d_model

        # Three-way fusion projection
        self.fusion = nn.Sequential(
            nn.Linear(d_model * 3, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        # Output gate scalar — controls how strongly top-down attention
        # influences the thalamus. Starts at 0.5 (moderate influence).
        self.gate = nn.Parameter(torch.tensor(float(gate_init)))

        # Cache the most recent query for the next tick
        self._cached_query: Optional[torch.Tensor] = None

    def build(
        self,
        goal_latent: Optional[torch.Tensor] = None,
        wm_context: Optional[torch.Tensor] = None,
        self_model: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Build a fresh attention query from cerebrum components.

        All inputs are (D,) vectors. Missing components default to zeros.
        Returns a (D,) L2-normalised query vector.
        """
        device = (
            goal_latent.device if goal_latent is not None
            else wm_context.device if wm_context is not None
            else self_model.device if self_model is not None
            else torch.device("cpu")
        )

        z = torch.zeros(self.d_model, device=device)
        g = goal_latent.to(device).flatten() if goal_latent is not None else z
        w = wm_context.to(device).flatten() if wm_context is not None else z
        s = self_model.to(device).flatten() if self_model is not None else z

        # Truncate or pad each component to d_model
        g = self._fit(g, device)
        w = self._fit(w, device)
        s = self._fit(s, device)

        combined = torch.cat([g, w, s], dim=-1).unsqueeze(0)  # (1, 3D)
        query = self.fusion(combined).squeeze(0)              # (D,)
        query = F.normalize(query, p=2, dim=0)

        self._cached_query = query.detach()
        return query

    def _fit(self, x: torch.Tensor, device: torch.device) -> torch.Tensor:
        """Pad or truncate a vector to length d_model."""
        if x.numel() >= self.d_model:
            return x[: self.d_model]
        out = torch.zeros(self.d_model, device=device)
        out[: x.numel()] = x
        return out

    def get_cached(self) -> Optional[torch.Tensor]:
        """Return the most recently built query (for next tick's thalamus pass)."""
        return self._cached_query

    def get_for_thalamus(self, batch_size: int) -> Optional[torch.Tensor]:
        """
        Format the cached query for AttentionBottleneck.forward.

        AttentionBottleneck expects (B, D). Scales by the gate parameter
        so the brain can learn how much top-down influence is appropriate.

        Returns None on the very first tick (no cached query yet).
        """
        if self._cached_query is None:
            return None
        gate = torch.sigmoid(self.gate)
        scaled = gate * self._cached_query
        return scaled.unsqueeze(0).expand(batch_size, -1)

    @property
    def gate_value(self) -> float:
        """Current sigmoid-gated top-down influence in [0, 1]."""
        return float(torch.sigmoid(self.gate).item())


__all__ = ["AttentionQueryBuilder"]
