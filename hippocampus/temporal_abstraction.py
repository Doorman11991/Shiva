"""
hippocampus/temporal_abstraction.py — Hierarchical time-scale compression.

The hippocampus compresses experience across multiple timescales:
individual moments → episodes → sessions → life narrative. This module
implements hierarchical temporal compression, creating summary
representations at multiple levels of abstraction.
"""

from __future__ import annotations

from collections import deque
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn


class TemporalLevel:
    """One level of the temporal hierarchy."""

    def __init__(self, name: str, window: int, latent_dim: int) -> None:
        self.name = name
        self.window = window
        self._buffer: deque = deque(maxlen=window)
        self._summary: Optional[torch.Tensor] = None

    def push(self, z: torch.Tensor) -> bool:
        """Push a latent vector. Returns True if a new summary was computed."""
        self._buffer.append(z.detach().to('cpu'))
        if len(self._buffer) == self.window:
            self._summary = torch.stack(list(self._buffer)).mean(dim=0)
            return True
        return False

    @property
    def summary(self) -> Optional[torch.Tensor]:
        return self._summary

    @property
    def is_ready(self) -> bool:
        return self._summary is not None


class TemporalAbstractor(nn.Module):
    """
    Hierarchical temporal compression across multiple timescales.

    Levels (configurable, defaults mirror biological timescales):
        moment   → window=1    (raw latent, no compression)
        episode  → window=16   (short sequence summary)
        session  → window=64   (medium-term summary)
        epoch    → window=256  (long-term summary)

    Each level computes a mean-pooled summary when its buffer fills.
    The summaries are concatenated and projected into a single
    "temporal context" vector for the cerebrum.

    Args:
        latent_dim:  Dimensionality of latent vectors.
        levels:      List of (name, window_size) pairs defining the hierarchy.
    """

    def __init__(
        self,
        latent_dim: int = 512,
        levels: Optional[List[Tuple[str, int]]] = None,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim

        if levels is None:
            levels = [
                ("episode", 16),
                ("session", 64),
                ("epoch", 256),
            ]

        self._levels = [TemporalLevel(name, window, latent_dim) for name, window in levels]
        n_levels = len(levels)

        # Project concatenated summaries into a single temporal context vector
        self.projection = nn.Linear(latent_dim * n_levels, latent_dim)

    def push(self, z: torch.Tensor) -> None:
        """Push a new latent vector through all levels."""
        for level in self._levels:
            level.push(z)

    def get_temporal_context(self) -> torch.Tensor:
        """
        Return a single temporal context vector combining all levels.

        Levels without a summary yet use zero vectors.
        """
        device = next(self.projection.parameters()).device
        summaries = []
        for level in self._levels:
            if level.is_ready:
                summaries.append(level.summary.to(device))
            else:
                summaries.append(torch.zeros(self.latent_dim, device=device))

        combined = torch.cat(summaries, dim=-1).unsqueeze(0)  # (1, D*n_levels)
        return self.projection(combined).squeeze(0)           # (D,)

    def level_summaries(self) -> Dict[str, Optional[torch.Tensor]]:
        """Return all level summaries for inspection."""
        return {level.name: level.summary for level in self._levels}

    @property
    def ready_levels(self) -> List[str]:
        return [level.name for level in self._levels if level.is_ready]


__all__ = ["TemporalAbstractor", "TemporalLevel"]
