"""
cerebrum/personality.py — Behavioral style and task conditioning.

Personality is the cerebrum's stable pattern of thought and behaviour
across contexts. This module encodes behavioral style as learnable
conditioning vectors — risk tolerance, exploration tendency, verbosity —
that modulate the policy's action selection.

Extracted from: core/chip_policy.py (task conditioning logic)
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Task vocabulary (stable — never reorder, only append)
# ---------------------------------------------------------------------------

TASK_VOCAB: Dict[str, int] = {
    "general":  0,
    "research": 1,
    "browse":   2,
    "code":     3,
    "voice":    4,
    "memory":   5,
    "training": 6,
}
NUM_TASKS = len(TASK_VOCAB)


def task_id_for(name: str) -> int:
    """Resolve a task label to its index. Unknown labels map to 'general'."""
    return TASK_VOCAB.get(name, TASK_VOCAB["general"])


# ---------------------------------------------------------------------------
# Personality traits
# ---------------------------------------------------------------------------

class PersonalityTraits(nn.Module):
    """
    Learnable personality trait vectors that condition the policy.

    Traits are continuous vectors in latent space. They are added to the
    conscious latent before action selection, biasing the policy toward
    behaviours consistent with the current personality configuration.

    Built-in traits:
        risk_tolerance:    High → explore risky actions; Low → conservative
        curiosity:         High → seek novelty; Low → exploit known strategies
        persistence:       High → retry failures; Low → abandon quickly
        social:            High → prefer collaborative actions

    Args:
        latent_dim:  Latent dimensionality.
        n_tasks:     Number of task types for task conditioning.
    """

    def __init__(self, latent_dim: int = 512, n_tasks: int = NUM_TASKS) -> None:
        super().__init__()
        self.latent_dim = latent_dim

        # Task-specific conditioning embeddings (zero-init = neutral on day 1)
        self.task_embed = nn.Embedding(n_tasks, latent_dim)
        nn.init.zeros_(self.task_embed.weight)

        # Personality trait scalars (learnable, initialised to neutral)
        self.risk_tolerance = nn.Parameter(torch.zeros(1))
        self.curiosity_bias = nn.Parameter(torch.zeros(1))
        self.persistence = nn.Parameter(torch.zeros(1))

        # Trait projection: maps trait scalars → latent bias vector
        self.trait_proj = nn.Sequential(
            nn.Linear(3, latent_dim // 4),
            nn.GELU(),
            nn.Linear(latent_dim // 4, latent_dim),
        )
        nn.init.zeros_(self.trait_proj[-1].weight)
        nn.init.zeros_(self.trait_proj[-1].bias)

    def get_task_token(
        self,
        task_id: torch.Tensor,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Get task conditioning token for prepending to the input sequence.

        Args:
            task_id:    Scalar or (B,) task index tensor.
            batch_size: Batch size B.
            device:     Target device.

        Returns:
            (B, 1, D) task token for sequence prepending.
        """
        task_id = task_id.to(device)
        if task_id.dim() == 0:
            task_id = task_id.expand(batch_size)
        return self.task_embed(task_id).unsqueeze(1)

    def get_personality_bias(self) -> torch.Tensor:
        """
        Compute the current personality bias vector.

        Returns:
            (D,) personality bias to add to the conscious latent.
        """
        traits = torch.cat([
            self.risk_tolerance,
            self.curiosity_bias,
            self.persistence,
        ])
        return self.trait_proj(traits.unsqueeze(0)).squeeze(0)

    def trait_summary(self) -> Dict[str, float]:
        return {
            "risk_tolerance": float(torch.tanh(self.risk_tolerance).item()),
            "curiosity_bias": float(torch.tanh(self.curiosity_bias).item()),
            "persistence": float(torch.tanh(self.persistence).item()),
        }


__all__ = ["PersonalityTraits", "TASK_VOCAB", "NUM_TASKS", "task_id_for"]
