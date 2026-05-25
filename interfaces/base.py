"""
interfaces/base.py — All abstract base classes for the Chip brain.

This is the "white matter" — the connective tissue between regions.
No region imports from another region directly; they depend only on
these contracts, keeping the architecture clean and testable.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Optional, Tuple, Dict, Any, List, TYPE_CHECKING

if TYPE_CHECKING:
    from interfaces.signals import NeuralSignal


# ---------------------------------------------------------------------------
# Memory (Hippocampus contract)
# ---------------------------------------------------------------------------

class IEpisodicMemory(nn.Module, ABC):
    """Stores and retrieves experiential episodes for dreaming and identity."""

    @abstractmethod
    def store_episode(
        self,
        state_sequence: torch.Tensor,
        valence_sequence: torch.Tensor,
        empowerment_score: float,
    ) -> None: ...

    @abstractmethod
    def get_dream_batch(self, batch_size: int) -> Optional[torch.Tensor]: ...

    @abstractmethod
    def get_identity_context(self, current_latent: torch.Tensor) -> torch.Tensor: ...


# ---------------------------------------------------------------------------
# Actor (Cerebrum contract)
# ---------------------------------------------------------------------------

class IActor(nn.Module, ABC):
    """Parameterises a stochastic policy over a continuous action space."""

    @abstractmethod
    def forward(
        self, state_features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (mu, log_std)."""
        ...

    @abstractmethod
    def sample(
        self, state_features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (action, log_prob) via the reparameterisation trick."""
        ...


# ---------------------------------------------------------------------------
# Replay buffer (Brainstem contract)
# ---------------------------------------------------------------------------

class IReplayBuffer(ABC):
    """Stores and samples transition tuples for off-policy learning."""

    @abstractmethod
    def add(self, sample: Any) -> None: ...

    @abstractmethod
    def sample(self, batch_size: int) -> Tuple[List[Any], List[int], torch.Tensor]:
        """Returns (batch, indices, importance-sampling weights)."""
        ...

    @abstractmethod
    def update_priorities(self, indices: List[int], errors: torch.Tensor) -> None: ...


# ---------------------------------------------------------------------------
# Alignment loss strategy (Thalamus contract)
# ---------------------------------------------------------------------------

class IAlignmentLoss(ABC):

    @abstractmethod
    def compute(self, *embeddings: torch.Tensor) -> torch.Tensor: ...


# ---------------------------------------------------------------------------
# Weight-merge strategy (Thalamus contract)
# ---------------------------------------------------------------------------

class IWeightMergeStrategy(ABC):

    @abstractmethod
    def merge(
        self,
        target_model: nn.Module,
        ext_state_dict: Dict[str, torch.Tensor],
        ext_config: Dict[str, Any],
    ) -> Dict[str, torch.Tensor]: ...


# ---------------------------------------------------------------------------
# Swarm consciousness (Cerebellum contract)
# ---------------------------------------------------------------------------

class ISwarmNode(ABC):
    @abstractmethod
    def get_conscious_latent(self) -> torch.Tensor:
        """Return this node's current latent state vector (D,)."""
        ...

    @abstractmethod
    def receive_consensus(self, consensus_vector: torch.Tensor) -> None:
        """Integrate the aggregated global workspace vector into local state."""
        ...


class IGlobalWorkspace(ABC):

    @abstractmethod
    def register_node(self, node_id: str, node: "ISwarmNode") -> None: ...

    @abstractmethod
    def broadcast_consensus(self) -> torch.Tensor:
        """
        Compute consensus from all registered nodes and push it back to each.
        Returns the consensus vector (D,).
        """
        ...


# ---------------------------------------------------------------------------
# Parasitic weight extraction (Parasite contract)
# ---------------------------------------------------------------------------

class IRepresentationProbe(ABC):
    @abstractmethod
    def attach(self, host_model: nn.Module, layer_name: str) -> None:
        """Register a forward hook on the named layer of host_model."""
        ...

    @abstractmethod
    def detach(self) -> None:
        """Remove all registered hooks from the host model."""
        ...

    @abstractmethod
    def distil_step(
        self, host_input: torch.Tensor, target_encoder: nn.Module
    ) -> float:
        """
        Run one contrastive distillation step.
        Returns the scalar loss value.
        """
        ...


# ---------------------------------------------------------------------------
# Autonomous locomotion (Locomotion contract)
# ---------------------------------------------------------------------------

class ICognitiveSnapshot(ABC):
    @abstractmethod
    def serialise(self) -> bytes:
        """Pack cognitive state into a portable byte payload."""
        ...

    @classmethod
    @abstractmethod
    def deserialise(cls, payload: bytes) -> "ICognitiveSnapshot":
        """Reconstruct a snapshot from a byte payload."""
        ...


class ILocomotionTransport(ABC):
    @abstractmethod
    def send(self, snapshot: ICognitiveSnapshot, destination: str) -> str:
        ...

    @abstractmethod
    def receive(self, migration_id: str) -> ICognitiveSnapshot:
        ...


# ---------------------------------------------------------------------------
# Brain region (Signal bus contract)
# ---------------------------------------------------------------------------

class IBrainRegion(ABC):
    """
    Every brain region implements this contract.

    Regions communicate exclusively through NeuralSignal objects on the
    SignalBus — never by importing each other's internals directly.
    This mirrors how real brain regions communicate via axonal projections,
    not by sharing cytoplasm.
    """

    @abstractmethod
    def step(self, signals: List["NeuralSignal"]) -> List["NeuralSignal"]:
        """
        Process incoming signals and return outgoing signals.
        Called once per cognitive tick.
        """
        ...

    @abstractmethod
    def reset(self) -> None:
        """Reset transient state on episode boundary."""
        ...


__all__ = [
    "IEpisodicMemory",
    "IActor",
    "IReplayBuffer",
    "IAlignmentLoss",
    "IWeightMergeStrategy",
    "ISwarmNode",
    "IGlobalWorkspace",
    "IRepresentationProbe",
    "ICognitiveSnapshot",
    "ILocomotionTransport",
    "IBrainRegion",
]
