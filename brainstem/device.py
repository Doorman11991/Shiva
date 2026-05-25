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

    The CHIP_DEVICE environment variable, if set, overrides the `prefer`
    argument. Recognised values: "auto", "directml", "cuda", "mps", "cpu".
    """
    env = os.environ.get("CHIP_DEVICE")
    if env:
        prefer = env

    pref = (prefer or "auto").lower()
    if pref not in _VALID_PREFERENCES:
        return prefer  # type: ignore[return-value]

    chain = []
    if pref == "directml":
        chain = [_try_directml, _try_cuda, _try_mps]
    elif pref == "cuda":
        chain = [_try_cuda, _try_directml, _try_mps]
    elif pref == "mps":
        chain = [_try_mps, _try_cuda, _try_directml]
    elif pref == "cpu":
        return "cpu"
    else:
        chain = [_try_directml, _try_cuda, _try_mps]

    for probe in chain:
        result = probe()
        if result is not None:
            return result
    return "cpu"


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
