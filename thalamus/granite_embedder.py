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
    - The 768->512 projection layer is part of the singleton, so its weights
      are stable across the lifetime of the process.

Async encoding
~~~~~~~~~~~~~~
Granite runs on a dedicated background thread. The tick loop submits an
encode job and immediately gets back the result from the *previous* tick
(1-tick lag). This removes the ~20ms granite forward pass from the
critical path entirely.

    tick N:   submit("new obs")  ->  returns result of tick N-1
    tick N+1: submit("next obs") ->  returns result of tick N

The lag is imperceptible for cognitive processing. If the background
thread hasn't finished yet (e.g. first tick), encode() blocks until
the result is ready.

Device selection
~~~~~~~~~~~~~~~~
Reads .chip_device (written by setup_device.py) to pick the right
backend. Falls back to the same CUDA -> MPS -> CPU priority chain
if the config file is absent.
"""

from __future__ import annotations

import concurrent.futures
import threading
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_GRANITE_MODEL_ID = "ibm-granite/granite-embedding-125m-english"
_GRANITE_HIDDEN_DIM = 768
_CHIP_LATENT_DIM = 512

_LOCAL_DIR_NAMES = (
    "granite-embedding-125m-english",
    "models/granite-embedding-125m-english",
    "../granite-embedding-125m-english",
)

# ---------------------------------------------------------------------------
# Singleton state
# ---------------------------------------------------------------------------

_singleton_lock = threading.Lock()
_singleton: Optional["GraniteEmbedder"] = None

# ---------------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------------

def _pick_device() -> torch.device:
    """
    Pick the best device for granite inference.

    Priority (from .chip_device config if present):
        directml -> cuda -> rocm -> mps -> ipex -> cpu

    Falls back to CUDA -> MPS -> CPU if no config file exists.
    """
    # Try reading the config written by setup_device.py
    config_path = Path(".chip_device")
    if config_path.exists():
        try:
            import json
            with open(config_path) as f:
                cfg = json.load(f)
            backend = cfg.get("backend", "cpu")
            verified = cfg.get("verified", False)
            if verified and backend != "cpu":
                return _backend_to_device(backend)
        except Exception:
            pass

    # Fallback: probe in priority order
    if torch.cuda.is_available():
        return torch.device(f"cuda:{torch.cuda.current_device()}")
    try:
        import torch_directml  # type: ignore
        if torch_directml.device_count() > 0:
            return torch.device(str(torch_directml.device()))
    except ImportError:
        pass
    mps = getattr(getattr(torch, "backends", None), "mps", None)
    if mps is not None and mps.is_available() and mps.is_built():
        return torch.device("mps")
    return torch.device("cpu")


def _backend_to_device(backend: str) -> torch.device:
    """Convert a backend string from .chip_device to a torch.device."""
    if backend == "cuda":
        if torch.cuda.is_available():
            return torch.device(f"cuda:{torch.cuda.current_device()}")
    if backend == "directml":
        try:
            import torch_directml  # type: ignore
            if torch_directml.device_count() > 0:
                return torch.device(str(torch_directml.device()))
        except ImportError:
            pass
    if backend == "rocm":
        # ROCm exposes through the CUDA API
        if torch.cuda.is_available():
            return torch.device(f"cuda:{torch.cuda.current_device()}")
    if backend == "mps":
        mps = getattr(getattr(torch, "backends", None), "mps", None)
        if mps is not None and mps.is_available():
            return torch.device("mps")
    if backend == "ipex":
        try:
            import intel_extension_for_pytorch  # type: ignore
            return torch.device("xpu")
        except ImportError:
            pass
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# GraniteEmbedder
# ---------------------------------------------------------------------------

class GraniteEmbedder(nn.Module):
    """
    Granite-125m-English wrapped as a Chip sensory encoder.

    Pipeline:
        text -> tokeniser -> granite encoder -> masked-mean-pool -> L2 norm ->
        Linear(768 -> 512) -> L2 norm  ->  z in R^512

    Async mode (default):
        encode() submits the job to a background thread and returns the
        result from the previous call. First call blocks until ready.
        This removes granite from the critical tick path.

    Sync mode (async_encode=False):
        encode() blocks until the result is ready. Used in tests and
        for inner_speech where the thought text is generated on the fly.

    Args:
        model_id_or_path:  HF model id, or local directory path.
        device:            Torch device override; auto-picked when None.
        max_seq_len:       Truncation length for the tokeniser.
        proj_seed:         Seed for the 768->512 projection initialisation.
        async_encode:      Run encode on a background thread (default True).
    """

    def __init__(
        self,
        model_id_or_path: Optional[str] = None,
        device: Optional[Union[str, torch.device]] = None,
        max_seq_len: int = 512,
        proj_seed: int = 0xC0FFEE,
        async_encode: bool = True,
    ) -> None:
        super().__init__()

        from transformers import AutoModel, AutoTokenizer

        self.device_ = torch.device(device) if device is not None else _pick_device()
        self.max_seq_len = max_seq_len
        self.async_encode = async_encode

        resolved = self._resolve_model_path(model_id_or_path)

        self.tokenizer = AutoTokenizer.from_pretrained(resolved)
        self.encoder = AutoModel.from_pretrained(resolved).to(self.device_).eval()
        for p in self.encoder.parameters():
            p.requires_grad_(False)

        # Deterministic projection: 768 -> 512.
        gen = torch.Generator().manual_seed(proj_seed)
        self.projection = nn.Linear(_GRANITE_HIDDEN_DIM, _CHIP_LATENT_DIM)
        with torch.no_grad():
            nn.init.kaiming_uniform_(self.projection.weight, a=5 ** 0.5, generator=gen)
            nn.init.zeros_(self.projection.bias)
        self.projection = self.projection.to(self.device_)

        # Per-call cache: same single string -> return immediately.
        self._cache_text: Optional[str] = None
        self._cache_vec: Optional[torch.Tensor] = None

        # Async encode state.
        # _executor: single-worker thread pool (one granite at a time).
        # _pending_future: the in-flight encode job.
        # _pending_text: the text submitted to that job.
        # _last_result: the most recently completed result.
        self._executor: Optional[concurrent.futures.ThreadPoolExecutor] = None
        self._pending_future: Optional[concurrent.futures.Future] = None
        self._pending_text: Optional[str] = None
        self._last_result: Optional[torch.Tensor] = None
        if async_encode:
            self._executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="granite_encode"
            )

        self.output_dim = _CHIP_LATENT_DIM
        self.model_id_or_path = resolved

    # ------------------------------------------------------------------
    # Path resolution
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_model_path(override: Optional[str]) -> str:
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
        mask = attention_mask.unsqueeze(-1).to(hidden_states.dtype)
        summed = (hidden_states * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1.0)
        return summed / counts

    # ------------------------------------------------------------------
    # Internal: synchronous forward pass
    # ------------------------------------------------------------------

    def _encode_sync(
        self,
        text: Union[str, Sequence[str]],
        batch_size: int = 32,
    ) -> torch.Tensor:
        """Blocking encode. Called directly or from the background thread."""
        single = isinstance(text, str)
        items: List[str] = [text] if single else list(text)
        if not items:
            return torch.empty(0, self.output_dim, device=self.device_)

        # DirectML does not support torch.inference_mode() — use no_grad instead.
        # On CUDA/CPU inference_mode is faster, so we pick based on device type.
        _no_grad_ctx = torch.no_grad if self.device_.type == "privateuseone" else torch.inference_mode

        chunks: List[torch.Tensor] = []
        with _no_grad_ctx():
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
                hidden = outputs.last_hidden_state
                pooled = self._mean_pool(hidden, batch_enc["attention_mask"])
                pooled = F.normalize(pooled, p=2, dim=-1)
                pooled = pooled.to(self.projection.weight.dtype)
                projected = self.projection(pooled)
                projected = F.normalize(projected, p=2, dim=-1)
                chunks.append(projected)

        z = torch.cat(chunks, dim=0)
        return z[0] if single else z

    # ------------------------------------------------------------------
    # Public: encoding
    # ------------------------------------------------------------------

    def encode(
        self,
        text: Union[str, Sequence[str]],
        batch_size: int = 32,
    ) -> torch.Tensor:
        """
        Embed text into Chip's 512-D latent space.

        In async mode (default):
            - Submits `text` to the background thread.
            - Returns the result from the *previous* call (1-tick lag).
            - First call blocks until the result is ready.
            - If the same string is submitted twice in a row, returns
              the cached result immediately without re-encoding.

        In sync mode:
            - Blocks until the result is ready (original behaviour).
            - Used for inner_speech and tests.

        Returns:
            (D,) tensor for a single string, (N, D) for a list.
        """
        single = isinstance(text, str)

        # Cache hit: same single string as last completed encode.
        if single and text == self._cache_text and self._cache_vec is not None:
            if self.async_encode:
                # Still submit to keep the pipeline warm for the next tick.
                self._submit_async(text, batch_size)
            return self._cache_vec

        if not self.async_encode:
            result = self._encode_sync(text, batch_size)
            if single:
                self._cache_text = text
                self._cache_vec = result
            return result

        # --- Async path ---
        # 1. Collect the result from the previous submission.
        #    If we have a last_result already (not the first call), don't block —
        #    just check if the future is done and grab it if so.
        if self._last_result is not None and self._pending_future is not None:
            if self._pending_future.done():
                self._collect_pending()
            # else: background still running, we'll use last_result below
        else:
            # First call or no pending — block until we have something.
            prev_result = self._collect_pending()

        # 2. Submit the new text to the background thread.
        self._submit_async(text, batch_size)

        # 3. Return the previous result.
        #    On the very first call _last_result is None, so block on the
        #    just-submitted future to get a real result.
        if self._last_result is None:
            self._last_result = self._collect_pending()

        if single and self._last_result is not None:
            self._cache_text = self._pending_text
            self._cache_vec = self._last_result

        return self._last_result  # type: ignore[return-value]

    def encode_now(
        self,
        text: Union[str, Sequence[str]],
        batch_size: int = 32,
    ) -> torch.Tensor:
        """
        Always-synchronous encode. Bypasses the async pipeline.

        Use this when you need the embedding for the *current* text
        immediately (e.g. inner_speech, consistency checks on new beliefs).
        """
        return self._encode_sync(text, batch_size)

    # ------------------------------------------------------------------
    # Async helpers
    # ------------------------------------------------------------------

    def _submit_async(self, text: Union[str, Sequence[str]], batch_size: int) -> None:
        """Submit an encode job to the background thread."""
        if self._executor is None:
            return
        self._pending_text = text if isinstance(text, str) else None
        self._pending_future = self._executor.submit(
            self._encode_sync, text, batch_size
        )

    def _collect_pending(self) -> Optional[torch.Tensor]:
        """Wait for the pending future and return its result."""
        if self._pending_future is None:
            return self._last_result
        try:
            # Use a short timeout — if the GPU is fast, 5s is plenty.
            # If it times out, return the last known result so the tick
            # doesn't block indefinitely.
            result = self._pending_future.result(timeout=5.0)
            self._last_result = result
            self._pending_future = None
            return result
        except concurrent.futures.TimeoutError:
            # Background encode still running — return last result and let
            # the next tick collect it.
            return self._last_result
        except Exception as e:
            print(f"[GraniteEmbedder] async encode failed: {e}")
            self._pending_future = None
            return self._last_result

    def warmup(self, text: str = "warmup") -> None:
        """
        Pre-submit a warmup encode so the first real tick doesn't block.
        Call this once after boot, before the first tick().
        """
        if self.async_encode and self._executor is not None:
            self._submit_async(text, 32)

    def shutdown(self) -> None:
        """Shut down the background thread pool cleanly."""
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None

    # ------------------------------------------------------------------
    # Public: similarity helpers
    # ------------------------------------------------------------------

    def similarity(
        self,
        text1: Union[str, Sequence[str]],
        text2: Union[str, Sequence[str]],
    ) -> torch.Tensor:
        a = self.encode_now(text1)
        b = self.encode_now(text2)
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
        if not candidates:
            return []
        q = self.encode_now(query)
        c = self.encode_now(list(candidates))
        sims = (c @ q).cpu()
        k = min(top_k, len(candidates))
        scores, idx = sims.topk(k)
        return [(candidates[i], float(scores[j].item())) for j, i in enumerate(idx.tolist())]

    # ------------------------------------------------------------------
    # nn.Module surface
    # ------------------------------------------------------------------

    def forward(self, text: Union[str, Sequence[str]]) -> torch.Tensor:
        return self.encode(text)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def info(self) -> Dict:
        n_params = sum(p.numel() for p in self.encoder.parameters())
        return {
            "model": self.model_id_or_path,
            "device": str(self.device_),
            "output_dim": self.output_dim,
            "encoder_params": int(n_params),
            "max_seq_len": self.max_seq_len,
            "async_encode": self.async_encode,
        }

    def __repr__(self) -> str:
        mode = "async" if self.async_encode else "sync"
        return (
            f"GraniteEmbedder(model={self.model_id_or_path!r}, "
            f"device={self.device_}, out={self.output_dim}, mode={mode})"
        )


# ---------------------------------------------------------------------------
# Singleton accessors
# ---------------------------------------------------------------------------

def get_embedder(
    model_id_or_path: Optional[str] = None,
    device: Optional[Union[str, torch.device]] = None,
    max_seq_len: int = 512,
    proj_seed: int = 0xC0FFEE,
    async_encode: bool = True,
) -> GraniteEmbedder:
    """
    Return the process-wide GraniteEmbedder, creating it on first call.

    Subsequent calls ignore constructor arguments. Use reset_embedder()
    to force reconfiguration (tests only).
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
                    async_encode=async_encode,
                )
    return _singleton


def reset_embedder() -> None:
    """Drop the cached singleton (used by tests)."""
    global _singleton
    with _singleton_lock:
        if _singleton is not None:
            _singleton.shutdown()
        _singleton = None


__all__ = ["GraniteEmbedder", "get_embedder", "reset_embedder"]
