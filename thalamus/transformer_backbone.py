"""
thalamus/transformer_backbone.py — Information routing backbone.

The thalamus is the brain's relay station: every sensory signal passes
through it before reaching the cortex. This transformer encoder is the
computational equivalent — it filters, positions, and routes latent
tokens before they reach the cerebrum for higher reasoning.

Moved from: core/transformer_architecture.py
"""

from __future__ import annotations

import math
from typing import Optional, Union

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# GateHyperNetwork — per-token residual gating
# ---------------------------------------------------------------------------


class GateHyperNetwork(nn.Module):
    """
    Per-token gating signal in (0, 1)^D from a small two-layer MLP.

    Output layer is zero-initialised so the gate begins at sigmoid(0) = 0.5
    and learns from data whether to open or close each residual path.

        x ← x + g ⊙ F(x)         where F is attention or FFN
    """

    def __init__(self, d_model: int, hidden_dim: Optional[int] = None) -> None:
        super().__init__()
        hidden_dim = hidden_dim or d_model // 2
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, d_model),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(x))


# ---------------------------------------------------------------------------
# Transformer encoder block
# ---------------------------------------------------------------------------


class TransformerEncoderBlock(nn.Module):
    """
    Single Transformer encoder layer — the thalamic relay unit.

      • Multi-head self-attention with learned positional embeddings
      • GELU FFN with 4× expansion
      • Dynamic per-token gating on both residual paths
      • Optional emotionally-modulated attention bias (from amygdala)
      • Dropout on both residual branches

    Emotional bias formulation (from amygdala arousal signal):
        scores ← scores + valence · emotional_gate

    Args:
        d_model:    Model dim (must divide num_heads).
        num_heads:  Parallel attention heads.
        max_seq_len: Maximum supported sequence length for positional
                    encoding. Inputs longer than this are truncated to
                    the last `max_seq_len` tokens.
        dropout:    Dropout probability applied to attention output and
                    FFN output. 0.0 disables.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        max_seq_len: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.max_seq_len = max_seq_len

        # ----- Attention projections -----------------------------------------
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        # ----- Feed-forward network ------------------------------------------
        ff_dim = d_model * 4
        self.ff_net = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, d_model),
        )

        # ----- Layer norms ---------------------------------------------------
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # ----- Dynamic gating ------------------------------------------------
        self.attn_gate_net = GateHyperNetwork(d_model)
        self.ff_gate_net = GateHyperNetwork(d_model)

        # ----- Emotional modulation (signal from amygdala) -------------------
        self.emotional_gate = nn.Parameter(torch.ones(1))

        # ----- Positional encoding (learned, truncated-normal init) ----------
        self.pos_embed = nn.Parameter(torch.zeros(1, max_seq_len, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # ----- Dropout -------------------------------------------------------
        self.attn_dropout = nn.Dropout(dropout)
        self.ff_dropout = nn.Dropout(dropout)

    # ------------------------------------------------------------------
    # Internal: multi-head attention
    # ------------------------------------------------------------------

    def _multi_head_attention(
        self,
        x: torch.Tensor,
        bias_shift: Union[float, torch.Tensor] = 0.0,
    ) -> torch.Tensor:
        B, T, _ = x.shape

        Q = self.q_proj(x).view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        K = self.k_proj(x).view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        V = self.v_proj(x).view(B, T, self.num_heads, self.d_k).transpose(1, 2)

        scores = (Q @ K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if isinstance(bias_shift, torch.Tensor) or bias_shift != 0.0:
            scores = scores + bias_shift

        attn_weights = torch.softmax(scores, dim=-1)
        context = (attn_weights @ V).transpose(1, 2).contiguous().view(B, T, self.d_model)
        return self.out_proj(context)

    def _compute_gates(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.attn_gate_net(x), self.ff_gate_net(x)

    def _add_positions(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        if T > self.max_seq_len:
            x = x[:, -self.max_seq_len:, :]
            T = self.max_seq_len
        return x + self.pos_embed[:, :T, :]

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        valence: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Thalamic relay forward pass.

        Args:
            x:       (B, T, D) input tensor.
            valence: optional (B,) or scalar valence from amygdala for
                     emotional attention bias.

        Returns:
            (B, T, D) encoded tensor.
        """
        x = self._add_positions(x)
        gate_attn, gate_ff = self._compute_gates(x)

        bias_shift: Union[float, torch.Tensor] = 0.0
        if valence is not None:
            bias_shift = valence * self.emotional_gate
            if isinstance(bias_shift, torch.Tensor):
                while bias_shift.dim() < 4:
                    bias_shift = bias_shift.unsqueeze(-1)

        attn_out = self.attn_dropout(self._multi_head_attention(x, bias_shift=bias_shift))
        x = self.norm1(x + gate_attn * attn_out)

        ff_out = self.ff_dropout(self.ff_net(x))
        x = self.norm2(x + gate_ff * ff_out)
        return x

    # Backwards-compatible alias used by cerebrum, cerebellum, parasite.
    def forward_pass(
        self,
        x: torch.Tensor,
        valence: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.forward(x, valence)


__all__ = ["GateHyperNetwork", "TransformerEncoderBlock"]
