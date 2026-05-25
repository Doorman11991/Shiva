"""
thalamus/sensory_encoder.py — Unified multi-modal input encoding.

The thalamus receives raw sensory signals from all modalities and
converts them into standardised latent tokens before routing them
to the cerebrum. This module provides a unified encoder interface
for text, numerical, and vector inputs.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Union

import torch
import torch.nn as nn


class VectorEncoder(nn.Module):
    """Encodes a raw vector input into a latent token."""

    def __init__(self, input_dim: int, d_model: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TextEncoder(nn.Module):
    """
    Text-modality encoder backed by the GraniteEmbedder singleton.

    Consumes raw strings (or lists of strings) and emits Chip-space
    latent vectors. The heavy granite model is shared process-wide so
    every region that wants a text embedding pays the load cost only once.
    """

    def __init__(self, d_model: int = 512) -> None:
        super().__init__()
        self.d_model = d_model
        self._embedder = None  # populated lazily on first call

    def _ensure(self):
        if self._embedder is None:
            from thalamus.granite_embedder import get_embedder
            self._embedder = get_embedder()
            if self._embedder.output_dim != self.d_model:
                raise RuntimeError(
                    f"GraniteEmbedder output {self._embedder.output_dim} "
                    f"does not match SensoryEncoder d_model {self.d_model}"
                )
        return self._embedder

    def forward(self, x: Union[str, Sequence[str]]) -> torch.Tensor:
        """
        Args:
            x: A string or list of strings.
        Returns:
            (B, D) tensor where B = 1 for a single string.
        """
        z = self._ensure().encode(x)
        if z.dim() == 1:
            z = z.unsqueeze(0)
        return z


class SensoryEncoder(nn.Module):
    """
    Unified encoder for all input modalities.

    Produces standardised (B, T, D) latent token sequences from
    heterogeneous inputs. Each modality has its own encoder head,
    but all outputs share the same dimensionality for the thalamus.

    Supported modalities (extensible):
        "text":     Raw strings — encoded by GraniteEmbedder (auto-registered).
        "vector":   Raw float vectors (proprioception, sensor readings).
        "sequence": Pre-tokenised sequences (already in latent space).

    Args:
        d_model:    Output latent dimensionality.
        modalities: Dict mapping modality name → input dimensionality.
                    E.g. {"vector": 64, "proprioception": 12}.
                    A "text" entry, if present, is replaced by the granite
                    backed encoder regardless of the dim you pass.
        enable_text: If True (default) auto-register a TextEncoder bound
                    to the granite embedder, even when "text" is missing
                    from the modalities dict.
    """

    def __init__(
        self,
        d_model: int = 512,
        modalities: Optional[Dict[str, int]] = None,
        enable_text: bool = True,
    ) -> None:
        super().__init__()
        self.d_model = d_model

        if modalities is None:
            modalities = {"vector": 512}

        encoders: Dict[str, nn.Module] = {}
        for name, input_dim in modalities.items():
            if name == "text":
                # Always use the granite-backed encoder for text.
                encoders[name] = TextEncoder(d_model=d_model)
            else:
                encoders[name] = VectorEncoder(input_dim, d_model)

        if enable_text and "text" not in encoders:
            encoders["text"] = TextEncoder(d_model=d_model)

        self.encoders = nn.ModuleDict(encoders)

        # Modality type embedding (so the thalamus knows which sense this is)
        self.modality_embed = nn.Embedding(len(self.encoders) + 1, d_model)
        self._modality_to_idx = {name: i for i, name in enumerate(self.encoders)}

    def encode(self, x, modality: str) -> torch.Tensor:
        """
        Encode a single modality input.

        Args:
            x:        Modality-specific input. For "text", a str or list of
                      str. For other modalities, a (B, input_dim) tensor.
            modality: Modality name (must be in self.encoders).

        Returns:
            (B, 1, D) latent token (unsqueezed for sequence compatibility).
        """
        if modality not in self.encoders:
            raise KeyError(f"Unknown modality '{modality}'. "
                           f"Available: {list(self.encoders.keys())}")

        z = self.encoders[modality](x)  # (B, D)

        # Add modality type embedding
        mod_idx = self._modality_to_idx.get(modality, 0)
        mod_embed = self.modality_embed(
            torch.tensor([mod_idx], device=z.device)
        )  # (1, D)
        z = z + mod_embed

        return z.unsqueeze(1)  # (B, 1, D)

    def encode_multi(
        self,
        inputs: Dict,
    ) -> torch.Tensor:
        """
        Encode multiple modalities and concatenate into a token sequence.

        Args:
            inputs: Dict mapping modality name → (B, input_dim) tensor.

        Returns:
            (B, n_modalities, D) multi-modal token sequence.
        """
        tokens = []
        for modality, x in inputs.items():
            tokens.append(self.encode(x, modality))
        return torch.cat(tokens, dim=1)  # (B, n_modalities, D)

    def register_modality(self, name: str, input_dim: int) -> None:
        """Dynamically register a new modality encoder."""
        self.encoders[name] = VectorEncoder(input_dim, self.d_model)
        new_idx = len(self._modality_to_idx)
        self._modality_to_idx[name] = new_idx


__all__ = ["SensoryEncoder", "VectorEncoder", "TextEncoder"]
