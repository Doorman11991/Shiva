"""
cerebrum/meta_cognition.py — Thinking about thinking.

Meta-cognition is the cerebrum's ability to monitor its own cognitive
processes — to know when it's confused, confident, or needs more
information. This module estimates confidence and triggers deliberation
(slow thinking) when confidence is low.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn


class ConfidenceEstimator(nn.Module):
    """
    Estimates how confident the policy is about its current action.

    Confidence signals:
        - Action entropy: low entropy = high confidence
        - Q-value spread: |Q1 - Q2| small = critics agree = confident
        - World model error: low prediction error = familiar territory

    Confidence score ∈ [0, 1]:
        0 = very uncertain (trigger deliberation)
        1 = very confident (act immediately)

    Args:
        d_model:    Latent dimensionality.
        action_dim: Action space dimensionality.
    """

    def __init__(self, d_model: int, action_dim: int) -> None:
        super().__init__()
        # Maps [z_conscious, entropy, q_spread, wm_error] → confidence
        self.net = nn.Sequential(
            nn.Linear(d_model + 3, 128),
            nn.GELU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        z_conscious: torch.Tensor,
        log_probs: torch.Tensor,
        q1: torch.Tensor,
        q2: torch.Tensor,
        wm_error: float = 0.0,
    ) -> torch.Tensor:
        """
        Compute confidence score.

        Args:
            z_conscious: (B, D) conscious latent.
            log_probs:   (B, 1) log-probabilities of sampled actions.
            q1, q2:      (B, 1) twin Q-values.
            wm_error:    Scalar world model prediction error.

        Returns:
            (B, 1) confidence score in [0, 1].
        """
        # Normalise signals to [0, 1] range
        entropy = -log_probs.detach()                          # higher = less confident
        entropy_norm = torch.sigmoid(entropy - 1.0)            # centre around typical values
        q_spread = (q1 - q2).abs().detach()
        q_spread_norm = torch.sigmoid(-q_spread)               # small spread = high confidence
        wm_err_t = torch.full_like(entropy_norm, wm_error)
        wm_err_norm = torch.sigmoid(-wm_err_t)                 # low error = high confidence

        features = torch.cat([z_conscious, entropy_norm, q_spread_norm, wm_err_norm], dim=-1)
        return self.net(features)


class MetaCognitionMonitor(nn.Module):
    """
    Monitors cognitive state and decides when to deliberate vs act.

    The monitor tracks:
        - Confidence history (rolling window)
        - Number of consecutive low-confidence steps
        - Whether deliberation is currently active

    When confidence drops below the threshold for too many consecutive
    steps, it signals the cerebrum to enter deliberation mode.

    Args:
        confidence_threshold: Below this, trigger deliberation.
        patience:             Consecutive low-confidence steps before
                              triggering deliberation.
        d_model:              Latent dimensionality.
        action_dim:           Action space dimensionality.
    """

    def __init__(
        self,
        d_model: int,
        action_dim: int,
        confidence_threshold: float = 0.4,
        patience: int = 3,
    ) -> None:
        super().__init__()
        self.confidence_threshold = confidence_threshold
        self.patience = patience
        self.estimator = ConfidenceEstimator(d_model, action_dim)

        self._low_conf_streak = 0
        self._deliberating = False
        self._confidence_history: list = []

        # Platt scaling calibration state
        self._platt_a: float = 1.0    # identity scaling on day 1
        self._platt_b: float = 0.0    # no offset on day 1
        self._calibration_data: list = []
        self._calibration_window: int = 500

    def assess(
        self,
        z_conscious: torch.Tensor,
        log_probs: torch.Tensor,
        q1: torch.Tensor,
        q2: torch.Tensor,
        wm_error: float = 0.0,
    ) -> Tuple[float, bool]:
        """
        Assess current confidence and decide whether to deliberate.

        Returns the *calibrated* confidence score (post Platt scaling).
        """
        raw_confidence = self.estimator(z_conscious, log_probs, q1, q2, wm_error)
        raw_scalar = float(raw_confidence.mean().item())

        # Apply Platt calibration: calibrated = sigmoid(a * raw + b)
        calibrated = self._calibrate(raw_scalar)

        self._confidence_history.append(calibrated)
        if len(self._confidence_history) > 100:
            self._confidence_history.pop(0)

        if calibrated < self.confidence_threshold:
            self._low_conf_streak += 1
        else:
            self._low_conf_streak = 0
            self._deliberating = False

        if self._low_conf_streak >= self.patience:
            self._deliberating = True

        return calibrated, self._deliberating

    # ------------------------------------------------------------------
    # Confidence calibration (Platt scaling)
    # ------------------------------------------------------------------

    def record_outcome(self, predicted_confidence: float, actual_success: bool) -> None:
        """
        Record a (predicted_confidence, actual_outcome) pair for calibration.

        Call this after the action is taken and the outcome is known:
            brain.meta.record_outcome(confidence, reward > 0)

        Once enough samples accumulate, the Platt scaling parameters
        (a, b) are refitted so that confidence score ≈ P(success | score).
        """
        self._calibration_data.append((predicted_confidence, float(actual_success)))
        if len(self._calibration_data) > self._calibration_window:
            self._calibration_data.pop(0)

        # Refit every 50 samples once we have enough data.
        if len(self._calibration_data) >= 30 and len(self._calibration_data) % 50 == 0:
            self._fit_platt()

    def _calibrate(self, raw: float) -> float:
        """Apply current Platt scaling: sigmoid(a * raw + b)."""
        import math
        x = self._platt_a * raw + self._platt_b
        return 1.0 / (1.0 + math.exp(-x))

    def _fit_platt(self) -> None:
        """
        Fit Platt scaling parameters (a, b) via simple logistic regression
        on the calibration buffer.

        Platt (1999): fit P(y=1|f) = 1 / (1 + exp(Af + B)) to the
        (predicted_score, actual_outcome) pairs. We use Newton-Raphson
        with 20 iterations — overkill for 2 parameters but reliable.
        """
        import math

        data = self._calibration_data
        if len(data) < 10:
            return

        scores = [d[0] for d in data]
        labels = [d[1] for d in data]
        n = len(data)

        # Target probabilities (Platt's smoothed targets)
        n_pos = sum(labels)
        n_neg = n - n_pos
        t_pos = (n_pos + 1) / (n_pos + 2) if n_pos > 0 else 0.5
        t_neg = 1.0 / (n_neg + 2) if n_neg > 0 else 0.5
        targets = [t_pos if y > 0.5 else t_neg for y in labels]

        # Newton-Raphson for 2 parameters
        a, b = self._platt_a, self._platt_b
        for _ in range(20):
            d1a, d2a, d1b, d2b, d1ab = 0.0, 0.0, 0.0, 0.0, 0.0
            for f, t in zip(scores, targets):
                fApB = a * f + b
                if fApB >= 0:
                    p = math.exp(-fApB) / (1.0 + math.exp(-fApB))
                else:
                    p = 1.0 / (1.0 + math.exp(fApB))
                q = 1.0 - p
                d = p * q
                h11 = f * f * d
                h22 = d
                h21 = f * d
                g1 = f * (p - t)
                g2 = p - t
                d1a += g1
                d2a += h11
                d1b += g2
                d2b += h22
                d1ab += h21

            # Avoid singularity
            det = d2a * d2b - d1ab * d1ab
            if abs(det) < 1e-10:
                break
            a -= (d2b * d1a - d1ab * d1b) / det
            b -= (d2a * d1b - d1ab * d1a) / det

        # Clamp to sane range — prevents explosion on degenerate data.
        self._platt_a = max(-50.0, min(50.0, a))
        self._platt_b = max(-50.0, min(50.0, b))

    @property
    def is_deliberating(self) -> bool:
        return self._deliberating

    def mean_confidence(self) -> float:
        if not self._confidence_history:
            return 0.5
        return sum(self._confidence_history) / len(self._confidence_history)

    def status(self) -> dict:
        return {
            "mean_confidence": self.mean_confidence(),
            "low_conf_streak": self._low_conf_streak,
            "deliberating": self._deliberating,
            "platt_a": self._platt_a,
            "platt_b": self._platt_b,
            "calibration_samples": len(self._calibration_data),
        }


__all__ = ["ConfidenceEstimator", "MetaCognitionMonitor"]
