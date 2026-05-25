"""
thalamus/attention_bottleneck.py — Information bottleneck and salience gating.

The thalamus doesn't relay everything — it filters. Only salient signals
reach the cortex. This module implements a learned information bottleneck
that compresses inputs to only what's task-relevant, with salience-gated
routing to downstream regions.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class SalienceGate(nn.Module):
    """
    Computes per-token salience scores for filtering.

    Tokens with low salience are suppressed before being forwarded
    to the cerebrum. This prevents the cerebrum from being overwhelmed
    by irrelevant information.

    Args:
        d_model:    Token dimensionality.
        threshold:  Salience threshold below which tokens are suppressed.
    """

    def __init__(self, d_model: int, threshold: float = 0.3) -> None:
        super().__init__()
        self.threshold = threshold
        self.salience_net = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        x: torch.Tensor,
        hard_gate: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply salience gating to a token sequence.

        Args:
            x:          (B, T, D) token sequence.
            hard_gate:  If True, zero out tokens below threshold.
                        If False, soft-multiply by salience score.

        Returns:
            (gated_x, salience_scores) where salience_scores is (B, T, 1).
        """
        salience = self.salience_net(x)  # (B, T, 1)

        if hard_gate:
            mask = (salience > self.threshold).float()
            return x * mask, salience
        else:
            return x * salience, salience


class AttentionBottleneck(nn.Module):
    """
    Information bottleneck that forces compression to task-relevant features.

    Architecture:
        1. Salience gate: suppress irrelevant tokens
        2. Compress: project to bottleneck dimension
        3. Reconstruct: project back to full dimension
        4. Top-k selection: keep only the K most salient tokens

    The bottleneck is trained end-to-end with the rest of the network.
    The compression forces the thalamus to discard noise and retain signal.

    Args:
        d_model:        Full token dimensionality.
        bottleneck_dim: Compressed dimensionality (default d_model // 4).
        top_k:          Number of tokens to forward to cerebrum.
                        None = forward all (no selection).
    """

    def __init__(
        self,
        d_model: int,
        bottleneck_dim: Optional[int] = None,
        top_k: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.bottleneck_dim = bottleneck_dim or d_model // 4
        self.top_k = top_k

        self.salience_gate = SalienceGate(d_model)

        self.compress = nn.Sequential(
            nn.Linear(d_model, self.bottleneck_dim),
            nn.LayerNorm(self.bottleneck_dim),
            nn.GELU(),
        )
        self.reconstruct = nn.Linear(self.bottleneck_dim, d_model)

    def forward(
        self,
        x: torch.Tensor,
        top_down_query: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply the attention bottleneck.

        Args:
            x:               (B, T, D) input token sequence from sensory encoder.
            top_down_query:  (B, D) optional top-down attention query from cerebrum.
                             When provided, biases salience toward query-relevant tokens.

        Returns:
            (filtered_x, salience_scores)
            filtered_x: (B, T', D) filtered token sequence (T' ≤ T).
        """
        # Apply top-down bias if provided
        if top_down_query is not None:
            # Compute relevance of each token to the query
            q = top_down_query.unsqueeze(1)  # (B, 1, D)
            relevance = F.cosine_similarity(x, q.expand_as(x), dim=-1).unsqueeze(-1)  # (B, T, 1)
            x = x + 0.1 * relevance * x  # soft top-down modulation

        gated_x, salience = self.salience_gate(x, hard_gate=False)

        # Compress and reconstruct (information bottleneck)
        compressed = self.compress(gated_x)
        reconstructed = self.reconstruct(compressed)

        # Optional top-k token selection
        if self.top_k is not None and x.shape[1] > self.top_k:
            scores = salience.squeeze(-1)  # (B, T)
            _, top_indices = scores.topk(self.top_k, dim=-1)
            top_indices_sorted = top_indices.sort(dim=-1).values
            reconstructed = torch.gather(
                reconstructed,
                1,
                top_indices_sorted.unsqueeze(-1).expand(-1, -1, self.d_model),
            )
            salience = torch.gather(salience, 1, top_indices_sorted.unsqueeze(-1))

        return reconstructed, salience


__all__ = ["AttentionBottleneck", "SalienceGate"]
