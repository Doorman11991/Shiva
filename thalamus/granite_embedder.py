"""
thalamus/granite_embedder.py — Local sensory text encoder (granite-125m-english).

Biological role
~~~~~~~~~~~~~~~
The thalamus is the brain's sensory relay. Before any signal reaches the
cortex, the thalamus filters, encodes and routes it. This module is the
text-sensory variant of that pathway: it converts raw natural language
into Chip's unified 512-D latent space using IBM Granite's small English
embedding model.

Why a dedicated module
~~~~~~~~~~~~~~~~~~~~~~
Loading a 125M-parameter transformer is expensive (memory, time). We don't
want every region that needs a text embedding to load its own copy. The
GraniteEmbedder uses a process-wide singleton with lazy loading:

    - First call to `get_embedder()` loads the model.
    - Every subsequent call reuses the same instance.
    - The 768→512 projection layer is part of the singleton, so its weights
      are stable across the lifetime of the process.

Design constraints
~~~~~~~~~~~~~~~~~~
    • Pure Python — only `transformers` and `torch`.
    • Auto device pick: CUDA → MPS → CPU (mirrors brainstem.device priorities,
      minus DirectML which the HF stack does not support natively).
    • Deterministic projection (seeded), so the same text always maps to the
      same Chip-space vector across processes when you save/load weights.
    • Mean pooling over the last hidden state with attention-mask weighting,
      so padding tokens don't dilute the embedding.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import List, Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

# IBM's official model id on the Hugging Face Hub.
_GRANITE_MODEL_ID = "ibm-granite/granite-embedding-125m-english"

# Granite outputs 768-D vectors; Chip's unified latent space is 512-D.
_GRANITE_HIDDEN_DIM = 768
_Chip_LATENT_DIM = 512

# Local checkpoint directory under the project root, if any. The user can drop
# a HF-format snapshot here to run fully offline. The GGUF blob in the parent
# folder is for llama.cpp and cannot be loaded by `transformers`.
_LOCAL_DIR_NAMES = (
    "granite-embedding-125m-english",
    "models/granite-embedding-125m-english",
    "../granite-embedding-125m-english",
)


# ---------------------------------------------------------------------------
# Singleton state (one shared embedder per process)
# ---------------------------------------------------------------------------

_singleton_lock = threading.Lock()
_singleton: Optional["GraniteEmbedder"] = None


# ---------------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------------

def _pick_device() -> torch.device:
    """
    Pick the best torch device for HF transformer inference.

    Priority: CUDA → MPS → CPU. (DirectML is intentionally skipped here —
    transformers' attention kernels target CUDA/CPU/MPS only.)
    """
    if torch.cuda.is_available():
        return torch.device(f"cuda:{torch.cuda.current_device()}")
    mps = getattr(getattr(torch, "backends", None), "mps", None)
    if mps is not None and mps.is_available() and mps.is_built():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# GraniteEmbedder
# ---------------------------------------------------------------------------

class GraniteEmbedder(nn.Module):
    """
    Granite-125m-English wrapped as a Chip sensory encoder.

    Pipeline:
        text → tokeniser → granite encoder → masked-mean-pool → L2 norm →
        Linear(768 → 512) → L2 norm  →  z ∈ ℝ^512

    The final L2 normalisation makes cosine similarity equal to a dot product
    and keeps the projected vectors on a unit hyper-sphere — the same surface
    the rest of Chip's latent space is calibrated against.

    Args:
        model_id_or_path:  HF model id, or local directory path. Defaults to
                           a local snapshot if found, otherwise the hub id.
        device:            Torch device override; auto-picked when None.
        max_seq_len:       Truncation length for the tokeniser.
        proj_seed:         Seed for the 768→512 projection initialisation.
                           Same seed → same projection across processes.
    """

    def __init__(
        self,
        model_id_or_path: Optional[str] = None,
        device: Optional[Union[str, torch.device]] = None,
        max_seq_len: int = 512,
        proj_seed: int = 0xC0FFEE,
    ) -> None:
        super().__init__()

        # Lazy heavy-import: only pull `transformers` when the embedder is
        # actually instantiated. Keeps cold-import time down for users who
        # only need the rest of the brain.
        from transformers import AutoModel, AutoTokenizer

        self.device_ = torch.device(device) if device is not None else _pick_device()
        self.max_seq_len = max_seq_len

        resolved = self._resolve_model_path(model_id_or_path)

        # Tokeniser & encoder. local_files_only auto-detected: if the path is
        # a directory we trust it; otherwise allow the hub fallback.
        self.tokenizer = AutoTokenizer.from_pretrained(resolved)
        self.encoder = AutoModel.from_pretrained(resolved).to(self.device_).eval()
        for p in self.encoder.parameters():
            p.requires_grad_(False)

        # Deterministic projection: 768 → 512.
        gen = torch.Generator().manual_seed(proj_seed)
        self.projection = nn.Linear(_GRANITE_HIDDEN_DIM, _Chip_LATENT_DIM)
        with torch.no_grad():
            nn.init.kaiming_uniform_(self.projection.weight, a=5 ** 0.5, generator=gen)
            nn.init.zeros_(self.projection.bias)
        self.projection = self.projection.to(self.device_)

        # Public metadata so the rest of the brain can introspect us.
        self.input_dim = _GRANITE_HIDDEN_DIM
        self.output_dim = _Chip_LATENT_DIM
        self.model_id_or_path = resolved

    # ------------------------------------------------------------------
    # Path resolution
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_model_path(override: Optional[str]) -> str:
        """
        Decide where to load the model from.

        Order:
            1. Caller's explicit override (path or hub id).
            2. A local HF-format snapshot in the project tree.
            3. The hub id as a last resort (network required).
        """
        if override:
            return override

        here = Path(__file__).resolve().parent
        for candidate in _LOCAL_DIR_NAMES:
            p = (here / candidate).resolve()
            if p.is_dir() and (p / "config.json").is_file():
                return str(p)

        return _GRANITE_MODEL_ID

    # ------------------------------------------------------------------
    # Internal: pooling
    # ------------------------------------------------------------------

    @staticmethod
    def _mean_pool(
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Attention-mask-weighted mean pooling over the last hidden state.

            pooled[b] = Σ_t mask[b,t] · h[b,t]  /  max(Σ_t mask[b,t], 1)

        This is the standard recipe for sentence-transformers-style models
        and avoids letting [PAD] tokens dilute the sentence vector.
        """
        mask = attention_mask.unsqueeze(-1).to(hidden_states.dtype)  # (B, T, 1)
        summed = (hidden_states * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1.0)
        return summed / counts

    # ------------------------------------------------------------------
    # Public: encoding
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def encode(
        self,
        text: Union[str, Sequence[str]],
        batch_size: int = 32,
    ) -> torch.Tensor:
        """
        Embed a string or list of strings into Chip's 512-D latent space.

        Args:
            text:       A single string or a list of strings.
            batch_size: Maximum number of strings per forward pass.

        Returns:
            torch.Tensor of shape:
                (D,)        if `text` is a single string
                (N, D)      if `text` is a list of N strings
            All vectors are L2-normalised.
        """
        single = isinstance(text, str)
        items: List[str] = [text] if single else list(text)
        if not items:
            return torch.empty(0, self.output_dim, device=self.device_)

        chunks: List[torch.Tensor] = []
        for start in range(0, len(items), batch_size):
            batch = items[start : start + batch_size]
            batch_enc = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_seq_len,
                return_tensors="pt",
            ).to(self.device_)

            outputs = self.encoder(**batch_enc)
            hidden = outputs.last_hidden_state                       # (B, T, 768)
            pooled = self._mean_pool(hidden, batch_enc["attention_mask"])
            pooled = F.normalize(pooled, p=2, dim=-1)                # unit 768-D
            # Cast to the projection's dtype — granite may load as bfloat16
            # while the projection is float32 (or vice versa on accelerators).
            pooled = pooled.to(self.projection.weight.dtype)
            projected = self.projection(pooled)                      # (B, 512)
            projected = F.normalize(projected, p=2, dim=-1)          # unit 512-D
            chunks.append(projected)

        z = torch.cat(chunks, dim=0)
        return z[0] if single else z

    # ------------------------------------------------------------------
    # Public: similarity helpers
    # ------------------------------------------------------------------

    def similarity(self, text1: Union[str, Sequence[str]], text2: Union[str, Sequence[str]]) -> torch.Tensor:
        """
        Cosine similarity between two texts (or aligned batches).

        Both inputs are L2-normalised inside `encode`, so cosine sim
        reduces to a plain dot product.

        Returns:
            scalar tensor when both inputs are strings, otherwise a 1-D
            tensor of pairwise similarities (one per row).
        """
        a = self.encode(text1)
        b = self.encode(text2)
        if a.dim() == 1 and b.dim() == 1:
            return torch.dot(a, b)
        if a.dim() == 1:
            a = a.unsqueeze(0).expand_as(b)
        if b.dim() == 1:
            b = b.unsqueeze(0).expand_as(a)
        return (a * b).sum(dim=-1)

    def most_similar(
        self,
        query: str,
        candidates: Sequence[str],
        top_k: int = 5,
    ) -> List[tuple]:
        """
        Return the top-k most similar candidates to `query` by cosine sim.

        Returns:
            A list of (candidate_text, similarity_score) tuples, sorted
            by score descending.
        """
        if not candidates:
            return []
        q = self.encode(query)
        c = self.encode(list(candidates))
        sims = (c @ q).cpu()
        k = min(top_k, len(candidates))
        scores, idx = sims.topk(k)
        return [(candidates[i], float(scores[j].item())) for j, i in enumerate(idx.tolist())]

    # ------------------------------------------------------------------
    # nn.Module surface — lets the embedder slot into any pipeline that
    # expects a ModuleDict["text"](batch_of_strings) style call.
    # ------------------------------------------------------------------

    def forward(self, text: Union[str, Sequence[str]]) -> torch.Tensor:
        return self.encode(text)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"GraniteEmbedder(model={self.model_id_or_path!r}, "
            f"device={self.device_}, in={self.input_dim}, out={self.output_dim})"
        )

    def info(self) -> dict:
        """A small dict of runtime info for the brainstem health monitor."""
        n_params = sum(p.numel() for p in self.encoder.parameters())
        return {
            "model": self.model_id_or_path,
            "device": str(self.device_),
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "encoder_params": int(n_params),
            "max_seq_len": self.max_seq_len,
        }


# ---------------------------------------------------------------------------
# Singleton accessors
# ---------------------------------------------------------------------------

def get_embedder(
    model_id_or_path: Optional[str] = None,
    device: Optional[Union[str, torch.device]] = None,
    max_seq_len: int = 512,
    proj_seed: int = 0xC0FFEE,
) -> GraniteEmbedder:
    """
    Return the process-wide GraniteEmbedder, creating it on first call.

    Subsequent calls ignore the constructor arguments — the first caller
    determines the configuration. Use `reset_embedder()` if you need to
    force a reconfiguration during testing.
    """
    global _singleton
    if _singleton is None:
        with _singleton_lock:
            if _singleton is None:
                _singleton = GraniteEmbedder(
                    model_id_or_path=model_id_or_path,
                    device=device,
                    max_seq_len=max_seq_len,
                    proj_seed=proj_seed,
                )
    return _singleton


def reset_embedder() -> None:
    """Drop the cached singleton (used by tests)."""
    global _singleton
    with _singleton_lock:
        _singleton = None


__all__ = ["GraniteEmbedder", "get_embedder", "reset_embedder"]
