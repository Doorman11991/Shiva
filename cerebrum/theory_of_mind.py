"""
cerebrum/theory_of_mind.py — Model other agents' beliefs and intentions.

In multi-agent scenarios (the swarm), understanding what other nodes
are likely thinking and planning is critical for coordination. This
module maintains a simple model of each peer node: their recent latent
trajectory, predicted next action, and inferred intent.

Theory of Mind (ToM) allows:
    - Predicting what another agent will do next
    - Detecting deception (action inconsistent with stated intent)
    - Coordinating without explicit communication
    - Anticipating how one's own actions affect others

Design: a small per-agent LSTM that, given the peer's observed latent
trajectory, predicts their next latent state and likely action direction.
"""

from __future__ import annotations

from collections import deque
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class AgentModel(nn.Module):
    """
    A model of one other agent's behaviour.

    Given the peer's recent latent trajectory (observed via the swarm's
    consensus broadcast), predicts their next state and action direction.

    Args:
        latent_dim: Dimensionality of latent states.
        action_dim: Action space dimensionality.
        hidden_dim: LSTM hidden size.
    """

    def __init__(
        self,
        latent_dim: int = 512,
        action_dim: int = 4,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.action_dim = action_dim

        self.lstm = nn.LSTM(latent_dim, hidden_dim, batch_first=True)
        self.state_head = nn.Linear(hidden_dim, latent_dim)
        self.action_head = nn.Linear(hidden_dim, action_dim)
        self.intent_head = nn.Linear(hidden_dim, latent_dim)

    def forward(
        self, trajectory: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Predict next state, action, and intent from observed trajectory.

        Args:
            trajectory: (1, T, D) peer's recent latent states.

        Returns:
            predicted_state: (1, D) expected next latent state.
            predicted_action: (1, A) expected next action direction.
            inferred_intent: (1, D) inferred goal latent.
        """
        h, _ = self.lstm(trajectory)
        last_h = h[:, -1, :]  # (1, hidden)

        pred_state = self.state_head(last_h)
        pred_action = torch.tanh(self.action_head(last_h))
        intent = self.intent_head(last_h)

        return pred_state, pred_action, intent


class TheoryOfMind(nn.Module):
    """
    Maintains mental models of all peer agents in the swarm.

    Args:
        latent_dim:     Latent dimensionality.
        action_dim:     Action space dimensionality.
        max_history:    Max trajectory length per peer.
    """

    def __init__(
        self,
        latent_dim: int = 512,
        action_dim: int = 4,
        max_history: int = 16,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.max_history = max_history

        self._peer_histories: Dict[str, deque] = {}
        self._peer_models: nn.ModuleDict = nn.ModuleDict()
        self._predictions: Dict[str, Dict] = {}

    def observe_peer(self, peer_id: str, latent: torch.Tensor) -> None:
        """
        Record an observation of a peer's current latent state.

        Args:
            peer_id: Identifier for the peer agent.
            latent:  (D,) the peer's current latent (from swarm broadcast).
        """
        if peer_id not in self._peer_histories:
            self._peer_histories[peer_id] = deque(maxlen=self.max_history)
            self._peer_models[peer_id] = AgentModel(
                self.latent_dim, self.action_dim
            )
        self._peer_histories[peer_id].append(latent.detach().cpu())

    @torch.no_grad()
    def predict_peer(self, peer_id: str) -> Optional[Dict]:
        """
        Predict a peer's next state, action, and intent.

        Returns:
            Dict with 'predicted_state', 'predicted_action', 'inferred_intent',
            or None if insufficient observations.
        """
        if peer_id not in self._peer_histories:
            return None
        history = list(self._peer_histories[peer_id])
        if len(history) < 3:
            return None

        trajectory = torch.stack(history).unsqueeze(0)  # (1, T, D)
        model = self._peer_models[peer_id]
        pred_state, pred_action, intent = model(trajectory)

        prediction = {
            "predicted_state": pred_state.squeeze(0),
            "predicted_action": pred_action.squeeze(0),
            "inferred_intent": intent.squeeze(0),
        }
        self._predictions[peer_id] = prediction
        return prediction

    def detect_deception(
        self,
        peer_id: str,
        actual_state: torch.Tensor,
        threshold: float = 0.5,
    ) -> Tuple[bool, float]:
        """
        Detect if a peer's actual behaviour deviates from predicted.

        Large prediction error may indicate the peer is being deceptive
        or unpredictable. Returns (is_deceptive, prediction_error).
        """
        if peer_id not in self._predictions:
            self.predict_peer(peer_id)
        if peer_id not in self._predictions:
            return False, 0.0

        predicted = self._predictions[peer_id]["predicted_state"]
        actual = actual_state.detach().cpu()
        error = float(F.mse_loss(predicted, actual).item())
        return error > threshold, error

    def peer_alignment(self, peer_id: str, my_intent: torch.Tensor) -> float:
        """
        How aligned is a peer's inferred intent with our own?

        Returns cosine similarity [-1, 1]. High = cooperative, low = adversarial.
        """
        pred = self.predict_peer(peer_id)
        if pred is None:
            return 0.0
        their_intent = pred["inferred_intent"]
        return float(F.cosine_similarity(
            my_intent.unsqueeze(0).cpu(),
            their_intent.unsqueeze(0),
        ).item())

    def status(self) -> Dict:
        return {
            "n_peers": len(self._peer_histories),
            "peer_ids": list(self._peer_histories.keys()),
            "observation_counts": {
                pid: len(hist) for pid, hist in self._peer_histories.items()
            },
        }


__all__ = ["TheoryOfMind", "AgentModel"]
