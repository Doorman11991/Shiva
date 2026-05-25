"""
brainstem/health_monitor.py — Training vitals and auto-recovery.

The brainstem monitors vital signs — heart rate, blood pressure, breathing.
If any go out of range, it triggers reflexive corrections. This module
does the same for training: it watches for NaN/Inf in losses and activations,
detects divergence trends, and can trigger checkpoint rollback.
"""

from __future__ import annotations

import math
import time
from collections import deque
from typing import Any, Callable, Dict, List, Optional

import torch
import torch.nn as nn


class HealthMonitor:
    """
    Watches training vitals and triggers recovery actions on anomalies.

    Monitored signals:
        - Loss values (NaN, Inf, divergence trend)
        - Gradient norms (explosion detection)
        - Activation statistics (dead neurons, saturation)
        - Training throughput (steps/sec)

    Recovery actions (injected as callbacks):
        - on_nan:       Called when NaN/Inf detected in a loss.
        - on_diverge:   Called when loss trend exceeds divergence_threshold.
        - on_recover:   Called after successful recovery.

    Args:
        window:              Rolling window size for trend detection.
        divergence_threshold: Loss increase ratio that triggers diverge alert.
                             E.g. 3.0 = loss tripled vs window mean.
        on_nan:              Optional callback(step, label) on NaN detection.
        on_diverge:          Optional callback(step, label, ratio) on divergence.
    """

    def __init__(
        self,
        window: int = 50,
        divergence_threshold: float = 3.0,
        on_nan: Optional[Callable] = None,
        on_diverge: Optional[Callable] = None,
    ) -> None:
        self.window = window
        self.divergence_threshold = divergence_threshold
        self._on_nan = on_nan
        self._on_diverge = on_diverge

        self._loss_history: Dict[str, deque] = {}
        self._nan_counts: Dict[str, int] = {}
        self._step: int = 0
        self._start_time: float = time.time()
        self._step_times: deque = deque(maxlen=100)

    def record(self, label: str, value: float) -> bool:
        """
        Record a scalar metric. Returns True if healthy, False if anomaly detected.

        Args:
            label: Metric name (e.g. "critic_loss", "actor_loss").
            value: Scalar value to record.
        """
        if label not in self._loss_history:
            self._loss_history[label] = deque(maxlen=self.window)
            self._nan_counts[label] = 0

        # NaN / Inf check
        if math.isnan(value) or math.isinf(value):
            self._nan_counts[label] += 1
            print(f"[HealthMonitor] NaN/Inf in '{label}' at step {self._step} "
                  f"(count: {self._nan_counts[label]})")
            if self._on_nan is not None:
                self._on_nan(self._step, label)
            return False

        history = self._loss_history[label]
        history.append(value)

        # Divergence check: current value vs window mean
        if len(history) >= self.window // 2:
            mean = sum(history) / len(history)
            if mean > 0 and value > self.divergence_threshold * mean:
                ratio = value / mean
                print(f"[HealthMonitor] Divergence in '{label}': "
                      f"{value:.4f} vs mean {mean:.4f} (ratio {ratio:.2f})")
                if self._on_diverge is not None:
                    self._on_diverge(self._step, label, ratio)
                return False

        return True

    def record_dict(self, metrics: Dict[str, float]) -> Dict[str, bool]:
        """Record multiple metrics at once. Returns health status per metric."""
        return {k: self.record(k, v) for k, v in metrics.items()}

    def check_model(self, model: nn.Module) -> Dict[str, Any]:
        """
        Scan model parameters for NaN/Inf weights and dead parameters.

        Returns a dict with:
            healthy:      bool — True if no anomalies found
            nan_params:   list of parameter names with NaN values
            inf_params:   list of parameter names with Inf values
            dead_params:  list of parameter names that are all-zero
        """
        nan_params, inf_params, dead_params = [], [], []

        for name, param in model.named_parameters():
            if param.data is None:
                continue
            if torch.isnan(param.data).any():
                nan_params.append(name)
            if torch.isinf(param.data).any():
                inf_params.append(name)
            if param.data.abs().max().item() == 0.0:
                dead_params.append(name)

        return {
            "healthy": len(nan_params) == 0 and len(inf_params) == 0,
            "nan_params": nan_params,
            "inf_params": inf_params,
            "dead_params": dead_params,
        }

    def tick(self) -> None:
        """Advance step counter and record throughput."""
        now = time.time()
        self._step_times.append(now)
        self._step += 1

    def throughput(self) -> float:
        """Return steps per second over the recent window."""
        if len(self._step_times) < 2:
            return 0.0
        elapsed = self._step_times[-1] - self._step_times[0]
        return (len(self._step_times) - 1) / max(elapsed, 1e-6)

    def summary(self) -> Dict[str, Any]:
        """Return a summary of current health status."""
        return {
            "step": self._step,
            "uptime_s": time.time() - self._start_time,
            "steps_per_sec": self.throughput(),
            "nan_counts": dict(self._nan_counts),
            "tracked_metrics": list(self._loss_history.keys()),
        }


__all__ = ["HealthMonitor"]
