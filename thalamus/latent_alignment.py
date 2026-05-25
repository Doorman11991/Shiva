"""
thalamus/latent_alignment.py — Multi-modal sensory alignment.

The thalamus receives signals from all sensory modalities and routes them
into a unified representation. This module does the same computationally:
it projects modality-specific encoder outputs into a shared 512-D latent
space using contrastive learning, so the cerebrum sees a single coherent
stream regardless of input modality.

Moved from: core/latent_alignment.py
"""

from __future__ import annotations
from typing import Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from interfaces.base import IAlignmentLoss


# ---------------------------------------------------------------------------
# Concrete loss strategies
# ---------------------------------------------------------------------------

class InfoNCELoss(IAlignmentLoss):
    """
    Noise-Contrastive Estimation loss (SimCLR-style).

      L = -1/2 * [CE(sim(z_a,z_b)/τ, I) + CE(sim(z_b,z_a)/τ, I)]
    """

    def __init__(self, temperature: float = 0.07) -> None:
        self.temperature = temperature

    def compute(self, *embeddings: torch.Tensor) -> torch.Tensor:
        if len(embeddings) != 2:
            raise ValueError("InfoNCELoss expects exactly two embedding tensors.")
        z_a, z_b = embeddings
        z_a = F.normalize(z_a, p=2, dim=1)
        z_b = F.normalize(z_b, p=2, dim=1)
        logits = (z_a @ z_b.T) / self.temperature
        labels = torch.arange(z_a.shape[0], device=z_a.device)
        return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2


class TriModalAlignmentLoss(IAlignmentLoss):
    """
    Three-way symmetric alignment loss used for (view_A, view_B, emotion).

      L = (InfoNCE(z_a, z_b) + InfoNCE(z_a, z_e) + InfoNCE(z_b, z_e)) / 3
    """

    def __init__(self, temperature: float = 0.07) -> None:
        self._pairwise = InfoNCELoss(temperature)

    def compute(self, *embeddings: torch.Tensor) -> torch.Tensor:
        if len(embeddings) != 3:
            raise ValueError(
                "TriModalAlignmentLoss expects exactly three embedding tensors "
                "(z_a, z_b, z_emotion)."
            )
        z_a, z_b, z_emotion = embeddings
        return (
            self._pairwise.compute(z_a, z_b)
            + self._pairwise.compute(z_a, z_emotion)
            + self._pairwise.compute(z_b, z_emotion)
        ) / 3


# ---------------------------------------------------------------------------
# LatentAligner
# ---------------------------------------------------------------------------

class LatentAligner(nn.Module):
    """
    Projects modality-specific encoder outputs into a shared latent space via
    an information-bottleneck and aligns them using a pluggable loss strategy.

    This is the thalamic relay in computational form: all sensory streams
    (vision, audio, text, proprioception) are compressed and aligned into
    the same 512-D space before being forwarded to the cerebrum.

    Args:
        encoders:          ModuleDict mapping modality name → encoder network.
        d_model:           Shared latent dimensionality.
        backbone:          Optional shared backbone for supervised train_step.
                           Must expose a `forward_pass(x) → Tensor` method.
        default_loss:      IAlignmentLoss used in train_step when no emotion
                           IDs are supplied.
        emotional_loss:    IAlignmentLoss used when emotion IDs are supplied.
        lr:                AdamW learning rate.
    """

    EMOTION_VOCAB: Dict[str, int] = {"Angry": 0, "Sad": 1, "Happy": 2, "Calm": 3}

    def __init__(
        self,
        encoders: nn.ModuleDict,
        d_model: int = 512,
        backbone: Optional[nn.Module] = None,
        default_loss: Optional[IAlignmentLoss] = None,
        emotional_loss: Optional[IAlignmentLoss] = None,
        lr: float = 1e-4,
    ) -> None:
        super().__init__()
        self.aligners = encoders
        self.backbone = backbone

        # Information Bottleneck: compress then reconstruct to filter noise.
        self.bottleneck = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, d_model),
        )

        self.emotion_vocab = self.EMOTION_VOCAB
        self.emotion_embeddings = nn.Embedding(len(self.emotion_vocab), d_model)

        self._default_loss: IAlignmentLoss = default_loss or InfoNCELoss()
        self._emotional_loss: IAlignmentLoss = emotional_loss or TriModalAlignmentLoss()

        self.optimizer = optim.AdamW(self.parameters(), lr=lr, weight_decay=1e-2)

    def forward(self, x: torch.Tensor, modality: str) -> torch.Tensor:
        if modality not in self.aligners:
            raise KeyError(f"Modality '{modality}' not found in encoder registry.")
        z_raw = self.aligners[modality](x)
        return self.bottleneck(z_raw)

    def train_step(
        self,
        data_a: torch.Tensor,
        data_b: torch.Tensor,
        emotion_ids: Optional[torch.Tensor] = None,
    ) -> float:
        if self.backbone is None:
            raise AttributeError(
                "A backbone must be injected before calling train_step."
            )
        self.optimizer.zero_grad()
        z_a = self.backbone.forward_pass(data_a).mean(dim=1)
        z_b = self.backbone.forward_pass(data_b).mean(dim=1)

        if emotion_ids is not None:
            z_e = self.emotion_embeddings(emotion_ids)
            loss = self._emotional_loss.compute(z_a, z_b, z_e)
        else:
            loss = self._default_loss.compute(z_a, z_b)

        loss.backward()
        self.optimizer.step()
        return loss.item()
