"""
hippocampus/spatial_map.py — Latent-space cognitive map.

Hippocampal place cells create a spatial map of the environment.
This module creates the cognitive equivalent: a topology map of the
latent space, tracking which regions have been explored, how they
connect, and how to navigate between them.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class LatentPlaceCell(nn.Module):
    """
    A single "place cell" — fires when the agent is near a specific
    region of latent space.

    Args:
        centroid:   (D,) centre of this cell's receptive field.
        sigma:      Width of the Gaussian receptive field.
    """

    def __init__(self, centroid: torch.Tensor, sigma: float = 1.0) -> None:
        super().__init__()
        self.register_buffer("centroid", centroid.clone())
        self.sigma = sigma
        self.visit_count: int = 0

    def activation(self, z: torch.Tensor) -> torch.Tensor:
        """
        Compute activation for latent vector z.

        Args:
            z: (B, D) or (D,) latent vector.

        Returns:
            (B,) or scalar activation in [0, 1].
        """
        if z.dim() == 1:
            z = z.unsqueeze(0)
        dist_sq = ((z - self.centroid.unsqueeze(0)) ** 2).sum(dim=-1)
        return torch.exp(-dist_sq / (2 * self.sigma ** 2))


class CognitiveMap(nn.Module):
    """
    Maintains a topology of the latent space using place cells.

    As the agent explores, new place cells are created for novel regions.
    The map tracks:
        - Which regions have been visited (and how often)
        - Transition probabilities between regions
        - "Frontier" regions (visited few times = high curiosity value)

    Args:
        latent_dim:     Dimensionality of the latent space.
        max_cells:      Maximum number of place cells.
        novelty_thresh: Distance threshold for creating a new place cell.
        sigma:          Receptive field width for place cells.
    """

    def __init__(
        self,
        latent_dim: int = 512,
        max_cells: int = 512,
        novelty_thresh: float = 2.0,
        sigma: float = 1.0,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.max_cells = max_cells
        self.novelty_thresh = novelty_thresh
        self.sigma = sigma

        self._cells: List[LatentPlaceCell] = []
        self._transitions: Dict[Tuple[int, int], int] = {}
        self._last_cell_idx: Optional[int] = None

    def update(self, z: torch.Tensor) -> Tuple[int, float]:
        """
        Update the map with a new latent observation.

        Args:
            z: (D,) current latent vector.

        Returns:
            (cell_idx, novelty_score) — index of the nearest cell and
            how novel this observation is (0 = familiar, 1 = very novel).
        """
        z = z.detach()
        if z.dim() > 1:
            z = z.squeeze(0)

        if not self._cells:
            self._add_cell(z)
            return 0, 1.0

        # Find nearest cell
        activations = torch.stack([cell.activation(z) for cell in self._cells])
        nearest_idx = int(activations.argmax().item())
        nearest_activation = float(activations[nearest_idx].item())

        # Novelty = inverse of activation (low activation = far from known regions)
        novelty = 1.0 - nearest_activation

        # Create new cell if sufficiently novel and capacity allows
        if novelty > (1.0 - torch.exp(torch.tensor(-self.novelty_thresh ** 2 / 2)).item()):
            if len(self._cells) < self.max_cells:
                nearest_idx = self._add_cell(z)

        # Record transition
        if self._last_cell_idx is not None:
            key = (self._last_cell_idx, nearest_idx)
            self._transitions[key] = self._transitions.get(key, 0) + 1

        self._cells[nearest_idx].visit_count += 1
        self._last_cell_idx = nearest_idx

        return nearest_idx, novelty

    def _add_cell(self, centroid: torch.Tensor) -> int:
        cell = LatentPlaceCell(centroid, self.sigma)
        self._cells.append(cell)
        return len(self._cells) - 1

    def frontier_cells(self, top_k: int = 5) -> List[int]:
        """Return indices of the least-visited cells (exploration frontiers)."""
        if not self._cells:
            return []
        counts = [(i, cell.visit_count) for i, cell in enumerate(self._cells)]
        counts.sort(key=lambda x: x[1])
        return [idx for idx, _ in counts[:top_k]]

    def get_frontier_direction(self, z_current: torch.Tensor) -> Optional[torch.Tensor]:
        """
        Return a direction vector pointing toward the nearest frontier cell.
        Useful for curiosity-driven exploration.
        """
        frontiers = self.frontier_cells(top_k=3)
        if not frontiers:
            return None
        z = z_current.detach()
        if z.dim() > 1:
            z = z.squeeze(0)
        frontier_centroids = torch.stack([self._cells[i].centroid for i in frontiers])
        nearest_frontier = frontier_centroids[
            torch.norm(frontier_centroids - z.unsqueeze(0), dim=-1).argmin()
        ]
        direction = F.normalize(nearest_frontier - z, dim=0)
        return direction

    def stats(self) -> Dict:
        return {
            "n_cells": len(self._cells),
            "n_transitions": len(self._transitions),
            "total_visits": sum(c.visit_count for c in self._cells),
            "frontier_cells": self.frontier_cells(),
        }


__all__ = ["CognitiveMap", "LatentPlaceCell"]
