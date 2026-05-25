"""
amygdala/arousal_modulator.py — Attention gain control.

High arousal (fear, excitement) narrows attention to the most salient
features. Low arousal (calm, fatigue) broadens attention. The amygdala
modulates this gain signal and sends it to the thalamus to adjust
attention sensitivity in the transformer backbone.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ArousalModulator(nn.Module):
    """
    Computes an attention gain signal from the current arousal level.

    The gain is applied as a multiplicative bias to the transformer's
    attention scores in the thalamus:
        scores ← scores * gain_scale

    High arousal → gain > 1.0 (sharper, more focused attention)
    Low arousal  → gain < 1.0 (softer, more diffuse attention)

    Args:
        d_model:        Latent dimensionality.
        arousal_dim:    Dimensionality of the arousal input vector.
                        Defaults to 1 (scalar arousal from homeostasis).
    """

    def __init__(self, d_model: int, arousal_dim: int = 1) -> None:
        super().__init__()
        self.d_model = d_model

        # Maps arousal level → per-head gain scale
        self.gain_net = nn.Sequential(
            nn.Linear(arousal_dim, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )
        # Initialise to output 0 so gain starts at sigmoid(0) = 0.5 → scale = 1.0
        nn.init.zeros_(self.gain_net[-1].weight)
        nn.init.zeros_(self.gain_net[-1].bias)

    def forward(self, arousal: torch.Tensor) -> torch.Tensor:
        """
        Compute attention gain scale from arousal level.

        Args:
            arousal: (B, 1) or scalar arousal value in [0, 1].

        Returns:
            (B, 1) gain scale. Values > 1.0 sharpen attention,
            values < 1.0 soften it.
        """
        if arousal.dim() == 0:
            arousal = arousal.unsqueeze(0).unsqueeze(0)
        elif arousal.dim() == 1:
            arousal = arousal.unsqueeze(-1)

        # Map [0,1] arousal to gain: 0.5 arousal → gain 1.0 (neutral)
        raw = self.gain_net(arousal)
        # Sigmoid centred at 0 → range (0, 1), then scale to (0.5, 2.0)
        gain = 0.5 + 1.5 * torch.sigmoid(raw)
        return gain

    def get_valence_bias(
        self,
        arousal: torch.Tensor,
        valence: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the combined arousal+valence attention bias for the thalamus.

        This is the signal sent via "arousal_gain" on the SignalBus.

        Args:
            arousal: (B, 1) arousal level.
            valence: (B, 1) emotional valence.

        Returns:
            (B, 1) combined bias for transformer attention scores.
        """
        gain = self.forward(arousal)
        # Valence shifts the bias direction: positive valence → attend to
        # rewarding features; negative valence → attend to threat features
        return gain * valence


__all__ = ["ArousalModulator"]
