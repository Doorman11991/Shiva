"""
setup_device.py — First-run GPU detection and torch backend installer.

Run once before starting Chip:
    python setup_device.py

What it does:
    1. Detects your GPU vendor (NVIDIA, AMD, Intel, Apple, none).
    2. Installs the correct torch build for that hardware.
    3. Writes .chip_device so brain.py and granite_embedder.py
       both pick the right device on every subsequent run.

Supported backends:
    NVIDIA  -> torch + CUDA (pip install torch --index-url .../cu121)
    AMD     -> torch-directml on Windows (pip install torch-directml)
               torch + ROCm on Linux  (pip install torch --index-url .../rocm6.0)
    Intel   -> torch-directml on Windows (pip install torch-directml)
               Intel Extension for PyTorch on Linux (pip install intel-extension-for-pytorch)
    Apple   -> torch with MPS (already included in standard torch on macOS)
    None    -> CPU-only torch (already installed)
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple


# ---------------------------------------------------------------------------
# GPU detection
# ---------------------------------------------------------------------------

def _run(cmd: list[str]) -> str:
    """Run a command and return stdout, empty string on failure."""
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=10
        )
        return result.stdout.strip()
    except Exception:
        return ""


def detect_gpu() -> Dict:
    """
    Detect GPU vendor, model, and VRAM.

    Returns a dict with keys:
        vendor:  "nvidia" | "amd" | "intel" | "apple" | "none"
        name:    Human-readable GPU name
        vram_mb: VRAM in MB (0 if unknown)
        backend: Recommended torch backend string
        os:      "windows" | "linux" | "macos"
    """
    os_name = platform.system().lower()
    if "windows" in os_name:
        os_label = "windows"
    elif "darwin" in os_name:
        os_label = "macos"
    else:
        os_label = "linux"

    # --- Apple Silicon ---
    if os_label == "macos":
        chip = _run(["sysctl", "-n", "machdep.cpu.brand_string"])
        if "apple" in chip.lower() or not chip:
            return {"vendor": "apple", "name": chip or "Apple Silicon",
                    "vram_mb": 0, "backend": "mps", "os": os_label}

    # --- NVIDIA via nvidia-smi ---
    nvidia_out = _run(["nvidia-smi",
                       "--query-gpu=name,memory.total",
                       "--format=csv,noheader,nounits"])
    if nvidia_out:
        parts = nvidia_out.split(",")
        name = parts[0].strip() if parts else "NVIDIA GPU"
        vram = int(parts[1].strip()) if len(parts) > 1 else 0
        return {"vendor": "nvidia", "name": name,
                "vram_mb": vram, "backend": "cuda", "os": os_label}

    # --- AMD / Intel via WMI on Windows ---
    if os_label == "windows":
        wmi_out = _run([
            "powershell", "-NoProfile", "-Command",
            "Get-WmiObject Win32_VideoController | "
            "Select-Object Name,AdapterRAM | "
            "ConvertTo-Json"
        ])
        if wmi_out:
            try:
                import json as _json
                data = _json.loads(wmi_out)
                # WMI can return a single object or a list
                if isinstance(data, dict):
                    data = [data]
                for gpu in data:
                    name = gpu.get("Name", "") or ""
                    vram_bytes = gpu.get("AdapterRAM") or 0
                    vram_mb = int(vram_bytes) // (1024 * 1024)
                    name_lower = name.lower()
                    if "nvidia" in name_lower:
                        return {"vendor": "nvidia", "name": name,
                                "vram_mb": vram_mb, "backend": "cuda", "os": os_label}
                    if "amd" in name_lower or "radeon" in name_lower:
                        return {"vendor": "amd", "name": name,
                                "vram_mb": vram_mb, "backend": "directml", "os": os_label}
                    if "intel" in name_lower:
                        return {"vendor": "intel", "name": name,
                                "vram_mb": vram_mb, "backend": "directml", "os": os_label}
            except Exception:
                pass

    # --- AMD / Intel on Linux via lspci ---
    if os_label == "linux":
        lspci = _run(["lspci"])
        for line in lspci.splitlines():
            ll = line.lower()
            if "vga" in ll or "3d" in ll or "display" in ll:
                if "nvidia" in ll:
                    return {"vendor": "nvidia", "name": line.split(":")[-1].strip(),
                            "vram_mb": 0, "backend": "cuda", "os": os_label}
                if "amd" in ll or "radeon" in ll or "ati" in ll:
                    return {"vendor": "amd", "name": line.split(":")[-1].strip(),
                            "vram_mb": 0, "backend": "rocm", "os": os_label}
                if "intel" in ll:
                    return {"vendor": "intel", "name": line.split(":")[-1].strip(),
                            "vram_mb": 0, "backend": "ipex", "os": os_label}

    return {"vendor": "none", "name": "CPU only",
            "vram_mb": 0, "backend": "cpu", "os": os_label}


# ---------------------------------------------------------------------------
# Torch install commands per backend
# ---------------------------------------------------------------------------

def _torch_install_cmd(backend: str, os_label: str) -> Optional[list[str]]:
    """
    Return the pip install command for the given backend, or None if
    the current torch installation is already correct.
    """
    import torch
    current = torch.__version__

    if backend == "cuda":
        # Check if CUDA is already available
        if torch.cuda.is_available():
            return None  # already good
        return [
            sys.executable, "-m", "pip", "install", "--upgrade",
            "torch", "torchvision", "torchaudio",
            "--index-url", "https://download.pytorch.org/whl/cu121",
        ]

    if backend == "directml":
        try:
            import torch_directml  # type: ignore
            return None  # already installed
        except ImportError:
            pass
        # torch-directml only supports Python 3.10-3.12 as of 2026-05
        major, minor = sys.version_info[:2]
        if not (major == 3 and 10 <= minor <= 12):
            print(f"  [setup_device] torch-directml requires Python 3.10-3.12 "
                  f"(you have {major}.{minor}). Skipping DirectML install.")
            print(f"  Tip: create a Python 3.12 venv and re-run setup_device.py")
            return None  # signal no install, caller will fall back to cpu
        return [
            sys.executable, "-m", "pip", "install", "--upgrade",
            "torch-directml",
        ]

    if backend == "rocm" and os_label == "linux":
        if torch.cuda.is_available():  # ROCm exposes via CUDA API
            return None
        return [
            sys.executable, "-m", "pip", "install", "--upgrade",
            "torch", "torchvision", "torchaudio",
            "--index-url", "https://download.pytorch.org/whl/rocm6.0",
        ]

    if backend == "ipex" and os_label == "linux":
        try:
            import intel_extension_for_pytorch  # type: ignore
            return None
        except ImportError:
            pass
        return [
            sys.executable, "-m", "pip", "install", "--upgrade",
            "intel-extension-for-pytorch",
        ]

    if backend == "mps":
        # MPS is bundled with torch on macOS — no extra install needed.
        return None

    return None  # cpu — nothing to install


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def setup(force: bool = False) -> Dict:
    """
    Run GPU detection, install the right backend, write .chip_device.

    Args:
        force: Re-run even if .chip_device already exists.

    Returns:
        The GPU info dict.
    """
    config_path = Path(".chip_device")

    if config_path.exists() and not force:
        with open(config_path) as f:
            existing = json.load(f)
        print(f"[setup_device] Already configured: {existing['name']} "
              f"({existing['backend']}). Pass --force to re-detect.")
        return existing

    print("[setup_device] Detecting GPU...")
    gpu = detect_gpu()

    vendor_label = {
        "nvidia": "NVIDIA",
        "amd":    "AMD",
        "intel":  "Intel",
        "apple":  "Apple",
        "none":   "None",
    }.get(gpu["vendor"], gpu["vendor"])

    print(f"  Vendor : {vendor_label}")
    print(f"  Name   : {gpu['name']}")
    print(f"  VRAM   : {gpu['vram_mb']} MB" if gpu["vram_mb"] else "  VRAM   : unknown")
    print(f"  Backend: {gpu['backend']}")
    print(f"  OS     : {gpu['os']}")

    # Install the right torch backend
    cmd = _torch_install_cmd(gpu["backend"], gpu["os"])
    if cmd:
        print(f"\n[setup_device] Installing {gpu['backend']} backend...")
        print(f"  {' '.join(cmd)}")
        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"[setup_device] WARNING: install returned exit code {result.returncode}")
            print("  Falling back to CPU.")
            gpu["backend"] = "cpu"
    else:
        # cmd is None: either already installed, or install was skipped
        # (e.g. Python version incompatibility). Check if the backend works.
        if gpu["backend"] not in ("cpu", "mps"):
            if not _verify_backend(gpu["backend"]):
                print(f"[setup_device] {gpu['backend']} not available. Falling back to CPU.")
                gpu["backend"] = "cpu"
        print("[setup_device] Backend ready (no install needed).")

    # Verify the backend actually works after install
    gpu["verified"] = _verify_backend(gpu["backend"])
    if not gpu["verified"]:
        print(f"[setup_device] WARNING: {gpu['backend']} not functional after install. "
              "Falling back to CPU.")
        gpu["backend"] = "cpu"
        gpu["verified"] = True

    # Write config
    with open(config_path, "w") as f:
        json.dump(gpu, f, indent=2)
    print(f"\n[setup_device] Written to {config_path}")
    print(f"[setup_device] Chip will use: {gpu['backend']}")

    return gpu


def _verify_backend(backend: str) -> bool:
    """Quick sanity check that the backend can run a tensor op."""
    try:
        import torch
        if backend == "cuda":
            if not torch.cuda.is_available():
                return False
            t = torch.zeros(4, device="cuda")
            _ = t + 1
            return True
        if backend == "directml":
            import torch_directml  # type: ignore
            dev = torch_directml.device()
            t = torch.zeros(4, device=dev)
            _ = t + 1
            return True
        if backend == "rocm":
            if not torch.cuda.is_available():
                return False
            t = torch.zeros(4, device="cuda")
            _ = t + 1
            return True
        if backend == "mps":
            t = torch.zeros(4, device="mps")
            _ = t + 1
            return True
        if backend == "ipex":
            import intel_extension_for_pytorch  # type: ignore
            t = torch.zeros(4, device="xpu")
            _ = t + 1
            return True
        # cpu
        return True
    except Exception as e:
        print(f"  [verify] {backend} failed: {e}")
        return False


def read_config() -> Optional[Dict]:
    """Read .chip_device if it exists, return None otherwise."""
    p = Path(".chip_device")
    if p.exists():
        try:
            with open(p) as f:
                return json.load(f)
        except Exception:
            return None
    return None


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Detect GPU and install torch backend for Chip.")
    parser.add_argument("--force", action="store_true", help="Re-detect even if already configured.")
    args = parser.parse_args()
    setup(force=args.force)
