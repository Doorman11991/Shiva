"""
cerebellum/action_smoother.py — Temporal action smoothing.

The cerebellum produces smooth, coordinated movement by integrating
motor commands over time. Raw policy outputs can be jerky — this module
applies temporal smoothing to produce natural, continuous action sequences.
"""

from __future__ import annotations

from collections import deque
from typing import Optional

import torch
import torch.nn as nn


class ActionSmoother(nn.Module):
    """
    Applies temporal smoothing to raw policy action outputs.

    Smoothing methods:
        "ema":      Exponential moving average (fast, no memory overhead).
        "window":   Uniform average over a sliding window.
        "learned":  Learned convolution filter (adapts to action statistics).

    Args:
        action_dim:  Dimensionality of the action space.
        method:      Smoothing method: "ema", "window", or "learned".
        alpha:       EMA decay factor (only for "ema" method).
        window_size: Window size (only for "window" method).
    """

    def __init__(
        self,
        action_dim: int,
        method: str = "ema",
        alpha: float = 0.8,
        window_size: int = 5,
    ) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.method = method
        self.alpha = alpha
        self.window_size = window_size

        self._ema_state: Optional[torch.Tensor] = None
        self._window: deque = deque(maxlen=window_size)

        if method == "learned":
            # 1D convolution over time axis, per action dimension
            self.conv = nn.Conv1d(
                in_channels=action_dim,
                out_channels=action_dim,
                kernel_size=window_size,
                padding=window_size // 2,
                groups=action_dim,  # depthwise: each action dim independently
                bias=False,
            )
            # Initialise to uniform average
            nn.init.constant_(self.conv.weight, 1.0 / window_size)

    def smooth(self, action: torch.Tensor) -> torch.Tensor:
        """
        Apply smoothing to a single action tensor.

        Args:
            action: (B, action_dim) raw action from policy.

        Returns:
            (B, action_dim) smoothed action.
        """
        if self.method == "ema":
            return self._ema_smooth(action)
        elif self.method == "window":
            return self._window_smooth(action)
        elif self.method == "learned":
            return self._learned_smooth(action)
        return action

    def _ema_smooth(self, action: torch.Tensor) -> torch.Tensor:
        if self._ema_state is None:
            self._ema_state = action.detach().clone()
        self._ema_state = self.alpha * self._ema_state + (1 - self.alpha) * action.detach()
        return self.alpha * self._ema_state + (1 - self.alpha) * action

    def _window_smooth(self, action: torch.Tensor) -> torch.Tensor:
        self._window.append(action.detach())
        if len(self._window) == 0:
            return action
        return torch.stack(list(self._window)).mean(dim=0)

    def _learned_smooth(self, action: torch.Tensor) -> torch.Tensor:
        self._window.append(action)
        if len(self._window) < self.window_size:
            return action
        seq = torch.stack(list(self._window), dim=1)  # (B, T, action_dim)
        seq = seq.transpose(1, 2)                      # (B, action_dim, T)
        smoothed = self.conv(seq)                      # (B, action_dim, T)
        return smoothed[:, :, -1]                      # (B, action_dim) last step

    def reset(self) -> None:
        """Reset state on episode boundary."""
        self._ema_state = None
        self._window.clear()

    def forward(self, action: torch.Tensor) -> torch.Tensor:
        return self.smooth(action)


__all__ = ["ActionSmoother"]
