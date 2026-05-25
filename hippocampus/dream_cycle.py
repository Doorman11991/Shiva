"""
hippocampus/dream_cycle.py — Offline generative replay.

During REM sleep, the hippocampus generates counterfactual variations
of real experiences — "what if" scenarios that help the brain generalise
beyond what it has directly observed. This module implements the same:
offline replay with noise injection and counterfactual generation.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class DreamCycle(nn.Module):
    """
    Offline generative replay with noise injection and counterfactual generation.

    Dream process:
        1. Sample significant episodes from episodic memory.
        2. Inject structured noise to create counterfactual variations.
        3. Run the policy on dream states to compute a reconstruction loss.
        4. Optionally generate "what if" trajectories by perturbing actions.

    The reconstruction loss encourages the policy to be consistent across
    small variations of real experiences — a form of data augmentation
    that improves generalisation.

    Args:
        noise_scale:     Standard deviation of Gaussian noise injected
                         into dream states.
        counterfactual_k: Number of counterfactual variations per episode.
    """

    def __init__(
        self,
        noise_scale: float = 0.05,
        counterfactual_k: int = 3,
    ) -> None:
        super().__init__()
        self.noise_scale = noise_scale
        self.counterfactual_k = counterfactual_k

    def generate_dream_states(
        self,
        real_states: torch.Tensor,
    ) -> torch.Tensor:
        """
        Generate counterfactual dream states by injecting noise.

        Args:
            real_states: (B, T, D) real episode states.

        Returns:
            (B * counterfactual_k, T, D) augmented dream states.
        """
        B, T, D = real_states.shape
        repeated = real_states.unsqueeze(1).expand(B, self.counterfactual_k, T, D)
        repeated = repeated.reshape(B * self.counterfactual_k, T, D)

        noise = torch.randn_like(repeated) * self.noise_scale
        return repeated + noise

    def compute_dream_loss(
        self,
        policy: nn.Module,
        dream_states: torch.Tensor,
        real_states: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute reconstruction loss between dream policy outputs and real targets.

        L_dream = MSE(policy(dream_final), real_trajectory)

        Args:
            policy:       The cerebrum policy module.
            dream_states: (B, T, D) dream state sequences.
            real_states:  (B, T, D) corresponding real state sequences.

        Returns:
            Scalar dream loss.
        """
        # Use the final dream state to predict the real trajectory
        dream_final = dream_states[:, -1, :].unsqueeze(1)  # (B, 1, D)
        outputs, _, _ = policy.get_action(dream_final)

        targets = real_states[:, 1:, :]  # (B, T-1, D)
        return F.mse_loss(outputs.unsqueeze(1).expand_as(targets), targets)

    def run(
        self,
        policy: nn.Module,
        episodic_memory: nn.Module,
        batch_size: int = 32,
    ) -> Optional[float]:
        """
        Full dream cycle: sample → augment → compute loss.

        Args:
            policy:          Cerebrum policy.
            episodic_memory: Hippocampus episodic memory bank.
            batch_size:      Number of real episodes to sample.

        Returns:
            Dream loss scalar, or None if insufficient memories.
        """
        real_states = episodic_memory.get_dream_batch(batch_size)
        if real_states is None:
            return None

        dream_states = self.generate_dream_states(real_states)
        # Subsample dream_states back to batch_size for loss computation
        dream_states_sub = dream_states[:batch_size]

        loss = self.compute_dream_loss(policy, dream_states_sub, real_states)
        return loss.item()


__all__ = ["DreamCycle"]
