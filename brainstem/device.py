"""
brainstem/device.py — Hardware device selection.

The brainstem controls the body's basic infrastructure — the nervous
system bus that connects brain to hardware. This module selects the
best available compute device (DirectML → CUDA → MPS → CPU) and
exposes it to all other regions.

Moved from: core/device.py
"""

from __future__ import annotations

import os
from typing import Optional

import torch


_VALID_PREFERENCES = {"auto", "directml", "cuda", "mps", "cpu"}


def _try_directml() -> Optional[str]:
    try:
        import torch_directml  # type: ignore
    except ImportError:
        return None
    try:
        if torch_directml.device_count() < 1:
            return None
        return str(torch_directml.device())
    except Exception:
        return None


def _try_cuda() -> Optional[str]:
    if torch.cuda.is_available():
        return f"cuda:{torch.cuda.current_device()}"
    return None


def _try_mps() -> Optional[str]:
    backends = getattr(torch, "backends", None)
    mps = getattr(backends, "mps", None) if backends is not None else None
    if mps is not None and mps.is_available() and mps.is_built():
        return "mps"
    return None


def pick_device(prefer: Optional[str] = None) -> str:
    """
    Select the best available compute device for Chip.

    Priority order:
        1. CHIP_DEVICE environment variable
        2. `prefer` argument
        3. .chip_device config file (written by setup_device.py)
        4. Live probe: CUDA -> DirectML -> MPS -> CPU

    Recognised values: "auto", "directml", "cuda", "mps", "cpu".
    """
    env = os.environ.get("CHIP_DEVICE")
    if env:
        prefer = env

    pref = (prefer or "auto").lower()
    if pref not in _VALID_PREFERENCES:
        return prefer  # type: ignore[return-value]

    if pref == "cpu":
        return "cpu"

    # If a specific backend is requested, try it first then fall through.
    if pref != "auto":
        chain_map = {
            "directml": [_try_directml, _try_cuda, _try_mps],
            "cuda":     [_try_cuda, _try_directml, _try_mps],
            "mps":      [_try_mps, _try_cuda, _try_directml],
        }
        for probe in chain_map.get(pref, []):
            result = probe()
            if result is not None:
                return result
        return "cpu"

    # Auto: read .chip_device config first for consistency with granite_embedder.
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".chip_device")
    if os.path.exists(config_path):
        try:
            import json as _json
            with open(config_path) as f:
                cfg = _json.load(f)
            backend = cfg.get("backend", "cpu")
            verified = cfg.get("verified", False)
            if verified:
                result = _backend_str_to_device(backend)
                if result is not None:
                    return result
        except Exception:
            pass

    # Live probe fallback: CUDA -> DirectML -> MPS -> CPU
    for probe in [_try_cuda, _try_directml, _try_mps]:
        result = probe()
        if result is not None:
            return result
    return "cpu"


def _backend_str_to_device(backend: str) -> Optional[str]:
    """Try to get a working device string for a backend name."""
    if backend == "cuda" or backend == "rocm":
        return _try_cuda()
    if backend == "directml":
        return _try_directml()
    if backend == "mps":
        return _try_mps()
    if backend == "cpu":
        return "cpu"
    return None


def describe_device(device: str) -> str:
    """Return a human-readable label for a device string."""
    dev = torch.device(device)
    kind = dev.type

    if kind == "privateuseone":
        try:
            import torch_directml  # type: ignore
            idx = dev.index if dev.index is not None else 0
            name = torch_directml.device_name(idx)
            return f"DirectML[{idx}] {name}"
        except Exception:
            return f"DirectML[{dev.index}]"

    if kind == "cuda":
        try:
            idx = dev.index if dev.index is not None else torch.cuda.current_device()
            name = torch.cuda.get_device_name(idx)
            return f"CUDA[{idx}] {name}"
        except Exception:
            return str(dev)

    if kind == "mps":
        return "Apple MPS"

    return "CPU"


__all__ = ["pick_device", "describe_device"]
