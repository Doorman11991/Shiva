import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Optional, Tuple, Dict, Any, List


# ---------------------------------------------------------------------------
# Memory
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
# Actor
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
# Replay buffer
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
# Alignment loss strategy 
# ---------------------------------------------------------------------------

class IAlignmentLoss(ABC):

    @abstractmethod
    def compute(self, *embeddings: torch.Tensor) -> torch.Tensor: ...


# ---------------------------------------------------------------------------
# Weight-merge strategy  (pre-existing, now actually used via DIP)
# ---------------------------------------------------------------------------

class IWeightMergeStrategy(ABC):

    @abstractmethod
    def merge(
        self,
        target_model: nn.Module,
        ext_state_dict: Dict[str, torch.Tensor],
        ext_config: Dict[str, Any],
    ) -> Dict[str, torch.Tensor]: ...
