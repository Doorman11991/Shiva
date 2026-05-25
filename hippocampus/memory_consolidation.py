"""
hippocampus/memory_consolidation.py — Episodic to semantic transfer.

During sleep, the hippocampus replays experiences to the neocortex,
gradually transferring episodic memories into semantic knowledge (facts,
skills, world models). This module implements the computational equivalent:
replaying episodic memories to update the world model's weights.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class MemoryConsolidator(nn.Module):
    """
    Transfers episodic memory patterns into semantic knowledge (world model).

    Consolidation process:
        1. Sample a batch of significant episodes from the hippocampus.
        2. Run each episode through the world model as a prediction task.
        3. Backpropagate the prediction error to update world model weights.
        4. Optionally prune low-significance memories after consolidation.

    This is called during "idle" periods (low energy, no active task) —
    analogous to sleep consolidation in biological systems.

    Args:
        world_model:        The latent dynamics model to consolidate into.
        consolidation_lr:   Learning rate for consolidation updates.
        prune_threshold:    Significance below which memories are pruned
                            after consolidation. 0 disables pruning.
    """

    def __init__(
        self,
        world_model: nn.Module,
        consolidation_lr: float = 1e-4,
        prune_threshold: float = 0.05,
    ) -> None:
        super().__init__()
        self.world_model = world_model
        self.prune_threshold = prune_threshold
        self.optimizer = torch.optim.Adam(world_model.parameters(), lr=consolidation_lr)
        self._consolidation_steps = 0

    def consolidate(
        self,
        episode_batch: torch.Tensor,
        n_steps: int = 1,
    ) -> float:
        """
        Run consolidation on a batch of episodes.

        Args:
            episode_batch: (B, T, D) batch of episode state sequences.
            n_steps:       Number of gradient steps per call.

        Returns:
            Mean consolidation loss.
        """
        total_loss = 0.0
        B, T, D = episode_batch.shape

        for _ in range(n_steps):
            self.optimizer.zero_grad()

            # Predict each step from the previous step
            # Input: z_t, Target: z_{t+1}
            inputs = episode_batch[:, :-1, :]   # (B, T-1, D)
            targets = episode_batch[:, 1:, :]   # (B, T-1, D)

            # Flatten time into batch for efficiency
            inputs_flat = inputs.reshape(B * (T - 1), D)
            targets_flat = targets.reshape(B * (T - 1), D)

            predictions = self.world_model(inputs_flat)
            loss = F.mse_loss(predictions, targets_flat)

            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            self._consolidation_steps += 1

        return total_loss / n_steps

    @property
    def consolidation_steps(self) -> int:
        return self._consolidation_steps


__all__ = ["MemoryConsolidator"]
