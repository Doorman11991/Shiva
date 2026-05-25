"""
cerebrum/working_memory.py — Active thought buffer.

Working memory is the cerebrum's scratchpad — a small set of actively
maintained thoughts that can be manipulated and combined. Miller's Law
(7±2 items) reflects a fundamental capacity limit. This module implements
a fixed-capacity attention-gated buffer with exponential decay.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class MemorySlot:
    """One slot in working memory."""
    content: torch.Tensor       # (D,) latent vector
    salience: float             # 0.0 (low) to 1.0 (high)
    age: int = 0                # ticks since last refresh
    source_tag: str = ""        # which region wrote this slot


class WorkingMemory(nn.Module):
    """
    Fixed-capacity attention-gated working memory buffer.

    Capacity: K slots (default 7, per Miller's Law).
    Eviction: lowest-salience slot is evicted when buffer is full.
    Decay: salience decreases by decay_rate per tick.
    Access: soft attention over all slots given a query vector.

    Args:
        latent_dim:   Dimensionality of slot content vectors.
        capacity:     Number of slots (default 7).
        decay_rate:   Salience decay per tick (default 0.05).
        min_salience: Slots below this are eligible for eviction.
    """

    def __init__(
        self,
        latent_dim: int = 512,
        capacity: int = 7,
        decay_rate: float = 0.05,
        min_salience: float = 0.1,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.capacity = capacity
        self.decay_rate = decay_rate
        self.min_salience = min_salience

        self._slots: List[MemorySlot] = []

        # Attention: query → attention weights over slots
        self.query_proj = nn.Linear(latent_dim, latent_dim)
        self.key_proj = nn.Linear(latent_dim, latent_dim)

    def write(
        self,
        content: torch.Tensor,
        salience: float,
        source_tag: str = "",
    ) -> int:
        """
        Write a new item to working memory.

        If at capacity, evicts the lowest-salience slot.
        Returns the index of the written slot.
        """
        content = content.detach()
        if content.dim() > 1:
            content = content.squeeze(0)

        if len(self._slots) >= self.capacity:
            # Evict lowest-salience slot
            evict_idx = min(range(len(self._slots)), key=lambda i: self._slots[i].salience)
            self._slots[evict_idx] = MemorySlot(content, salience, 0, source_tag)
            return evict_idx
        else:
            self._slots.append(MemorySlot(content, salience, 0, source_tag))
            return len(self._slots) - 1

    def attend(self, query: torch.Tensor) -> torch.Tensor:
        """
        Soft attention over all slots given a query vector.

        Args:
            query: (B, D) or (D,) query vector.

        Returns:
            (B, D) or (D,) attended working memory context.
        """
        if not self._slots:
            return torch.zeros_like(query)

        batched = query.dim() > 1
        if not batched:
            query = query.unsqueeze(0)

        B, D = query.shape
        slot_contents = torch.stack([s.content for s in self._slots]).to(query.device)  # (K, D)
        slot_saliences = torch.tensor(
            [s.salience for s in self._slots], device=query.device
        )  # (K,)

        q = self.query_proj(query)                          # (B, D)
        k = self.key_proj(slot_contents)                    # (K, D)

        scores = (q @ k.T) / (D ** 0.5)                    # (B, K)
        # Weight by salience before softmax
        scores = scores + torch.log(slot_saliences + 1e-6).unsqueeze(0)
        weights = F.softmax(scores, dim=-1)                 # (B, K)

        context = weights @ slot_contents                   # (B, D)
        return context if batched else context.squeeze(0)

    def read_all(self) -> Optional[torch.Tensor]:
        """Return all slot contents as a (K, D) tensor, or None if empty."""
        if not self._slots:
            return None
        return torch.stack([s.content for s in self._slots])

    def decay_step(self) -> None:
        """Decay all slot saliences. Call once per cognitive tick."""
        for slot in self._slots:
            slot.salience = max(0.0, slot.salience - self.decay_rate)
            slot.age += 1
        # Remove slots that have decayed below minimum
        self._slots = [s for s in self._slots if s.salience >= self.min_salience]

    def refresh(self, slot_idx: int, new_salience: float) -> None:
        """Refresh a slot's salience (e.g. when it's accessed again)."""
        if 0 <= slot_idx < len(self._slots):
            self._slots[slot_idx].salience = min(1.0, new_salience)
            self._slots[slot_idx].age = 0

    def reset(self) -> None:
        """Clear all slots (episode boundary)."""
        self._slots.clear()

    @property
    def utilisation(self) -> float:
        """Fraction of capacity currently used."""
        return len(self._slots) / self.capacity

    def status(self) -> dict:
        return {
            "n_slots": len(self._slots),
            "capacity": self.capacity,
            "utilisation": self.utilisation,
            "slots": [
                {"source": s.source_tag, "salience": s.salience, "age": s.age}
                for s in self._slots
            ],
        }


__all__ = ["WorkingMemory", "MemorySlot"]
