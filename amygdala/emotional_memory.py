"""
amygdala/emotional_memory.py — Emotion-tagged memory encoding.

The amygdala tags memories with emotional significance at encoding time.
Traumatic or highly rewarding experiences are remembered more vividly
than neutral ones. This module biases hippocampal memory retrieval
toward emotionally significant episodes.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class EmotionalMemoryTagger(nn.Module):
    """
    Tags latent states with emotional significance scores at encoding time.

    The significance score is used by the hippocampus to weight replay
    sampling — high-significance memories are replayed more often.

    Significance formula:
        σ = α · |valence| + β · arousal + γ · surprise

    Where surprise = ||z_predicted - z_actual||₂ (from world model).

    Args:
        alpha:   Weight on valence magnitude.
        beta:    Weight on arousal level.
        gamma:   Weight on prediction surprise.
    """

    def __init__(
        self,
        alpha: float = 0.4,
        beta: float = 0.3,
        gamma: float = 0.3,
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def compute_significance(
        self,
        valence: torch.Tensor,
        arousal: torch.Tensor,
        surprise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute emotional significance score for a batch of experiences.

        Args:
            valence:  (B, 1) emotional valence in [-1, 1].
            arousal:  (B, 1) arousal level in [0, 1].
            surprise: (B, 1) optional prediction error from world model.

        Returns:
            (B, 1) significance score in [0, 1].
        """
        sig = self.alpha * valence.abs() + self.beta * arousal
        if surprise is not None:
            sig = sig + self.gamma * surprise
        return torch.clamp(sig, 0.0, 1.0)

    def tag_episode(
        self,
        states: torch.Tensor,
        valences: torch.Tensor,
        arousal: float,
        surprise: Optional[float] = None,
    ) -> Dict:
        """
        Create an emotion-tagged episode dict for the hippocampus.

        Args:
            states:   (T, D) state sequence.
            valences: (T, 1) valence sequence.
            arousal:  Scalar arousal level.
            surprise: Optional scalar prediction surprise.

        Returns:
            Dict with 'states', 'significance', 'emotional_tag'.
        """
        arousal_t = torch.tensor([[arousal]])
        surprise_t = torch.tensor([[surprise]]) if surprise is not None else None

        mean_valence = valences.mean().unsqueeze(0).unsqueeze(0)
        sig = self.compute_significance(mean_valence, arousal_t, surprise_t)

        return {
            "states": states,
            "significance": float(sig.item()),
            "emotional_tag": {
                "mean_valence": float(mean_valence.item()),
                "arousal": arousal,
                "surprise": surprise or 0.0,
            },
        }


class EmotionalRetrievalBias(nn.Module):
    """
    Biases memory retrieval queries toward emotionally congruent memories.

    Mood-congruent memory: humans recall memories that match their current
    mood more easily. This module implements the same bias: when the agent
    is in a high-arousal state, it preferentially retrieves high-arousal
    memories.

    Args:
        d_model:  Latent dimensionality.
    """

    def __init__(self, d_model: int) -> None:
        super().__init__()
        # Projects emotional state into query space
        self.emotion_to_query = nn.Linear(4, d_model)  # 4 = homeostasis dims

    def bias_query(
        self,
        base_query: torch.Tensor,
        homeostasis_vector: torch.Tensor,
        strength: float = 0.3,
    ) -> torch.Tensor:
        """
        Add an emotional bias to a memory retrieval query.

        Args:
            base_query:          (B, D) base retrieval query.
            homeostasis_vector:  (4,) current homeostatic state.
            strength:            How strongly to bias the query.

        Returns:
            (B, D) emotionally-biased query.
        """
        if homeostasis_vector.dim() == 1:
            homeostasis_vector = homeostasis_vector.unsqueeze(0)
        emotion_bias = self.emotion_to_query(homeostasis_vector)  # (1, D)
        return base_query + strength * emotion_bias


__all__ = ["EmotionalMemoryTagger", "EmotionalRetrievalBias"]
