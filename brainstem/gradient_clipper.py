"""
brainstem/gradient_clipper.py — Gradient health monitor.

Like the brainstem's reflex arcs that prevent dangerous muscle contractions,
this module monitors gradient norms and clips them before they can cause
catastrophic weight updates. It also logs statistics for the health monitor.
"""

from __future__ import annotations

from typing import Dict, Iterable, Optional

import torch
import torch.nn as nn


class GradientClipper:
    """
    Monitors and clips gradient norms across model parameters.

    Tracks a rolling history of gradient norms so the health monitor
    can detect divergence trends before they become catastrophic.

    Args:
        max_norm:       Global gradient norm clip threshold.
        norm_type:      p-norm type (default 2.0).
        history_len:    Number of recent norms to retain for diagnostics.
        warn_threshold: Log a warning when norm exceeds this multiple of
                        the running mean. 0 disables warnings.
    """

    def __init__(
        self,
        max_norm: float = 1.0,
        norm_type: float = 2.0,
        history_len: int = 100,
        warn_threshold: float = 5.0,
    ) -> None:
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.warn_threshold = warn_threshold
        self._history: list[float] = []
        self._history_len = history_len
        self._clip_count: int = 0
        self._step: int = 0

    def clip(self, parameters: Iterable[nn.Parameter], label: str = "") -> float:
        """
        Clip gradients and return the pre-clip global norm.

        Args:
            parameters: Iterable of model parameters (e.g. model.parameters()).
            label:      Optional tag for logging (e.g. "actor", "critic").

        Returns:
            Pre-clip gradient norm as a float.
        """
        params = [p for p in parameters if p.grad is not None]
        if not params:
            return 0.0

        total_norm = torch.nn.utils.clip_grad_norm_(
            params, self.max_norm, norm_type=self.norm_type
        ).item()

        self._history.append(total_norm)
        if len(self._history) > self._history_len:
            self._history.pop(0)

        if total_norm > self.max_norm:
            self._clip_count += 1

        if self.warn_threshold > 0 and len(self._history) >= 10:
            mean_norm = sum(self._history[-10:]) / 10
            if total_norm > self.warn_threshold * mean_norm:
                print(
                    f"[GradientClipper] {label} norm spike: "
                    f"{total_norm:.3f} vs mean {mean_norm:.3f} "
                    f"(step {self._step})"
                )

        self._step += 1
        return float(total_norm)

    def stats(self) -> Dict[str, float]:
        """Return diagnostic statistics."""
        if not self._history:
            return {"mean_norm": 0.0, "max_norm_seen": 0.0, "clip_rate": 0.0}
        return {
            "mean_norm": sum(self._history) / len(self._history),
            "max_norm_seen": max(self._history),
            "clip_rate": self._clip_count / max(self._step, 1),
            "last_norm": self._history[-1],
        }

    def reset_counts(self) -> None:
        """Reset clip counter (e.g. at epoch boundaries)."""
        self._clip_count = 0
        self._step = 0


__all__ = ["GradientClipper"]
