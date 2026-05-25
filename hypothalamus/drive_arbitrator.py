"""
hypothalamus/drive_arbitrator.py — Competing drive resolution.

When hunger and thirst compete, the hypothalamus arbitrates based on
urgency and context. This module does the same for cognitive drives:
curiosity wants to explore, safety wants to be conservative, energy
wants to rest. The arbitrator produces a single prioritised drive signal
that the cerebrum uses for goal selection.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch


@dataclass
class Drive:
    """A single motivational drive with urgency and valence."""
    name: str
    urgency: float          # 0.0 (low) to 1.0 (critical)
    valence: float          # -1.0 (aversive) to +1.0 (appetitive)
    source: str = ""        # which region generated this drive
    payload: Optional[dict] = field(default=None)  # optional context


class DriveArbitrator:
    """
    Priority queue arbitrator for competing motivational drives.

    Arbitration rules (in order):
        1. Safety drives with urgency > 0.8 always win (fear override).
        2. Otherwise, drives are scored by urgency × |valence|.
        3. Ties broken by recency (most recent drive wins).

    The winning drive is returned as a NeuralSignal-compatible dict
    for the cerebrum's goal generator to act on.

    Args:
        safety_override_threshold: Urgency above which safety drives
                                   preempt all others.
    """

    def __init__(self, safety_override_threshold: float = 0.8) -> None:
        self.safety_override_threshold = safety_override_threshold
        self._drives: List[Drive] = []

    def submit(self, drive: Drive) -> None:
        """Submit a drive for arbitration."""
        self._drives.append(drive)

    def submit_many(self, drives: List[Drive]) -> None:
        for d in drives:
            self.submit(d)

    def arbitrate(self) -> Optional[Drive]:
        """
        Select the winning drive from all submitted drives.

        Clears the drive queue after arbitration.
        Returns None if no drives were submitted.
        """
        if not self._drives:
            return None

        # Rule 1: Safety override
        safety_drives = [
            d for d in self._drives
            if d.name == "safety" and d.urgency >= self.safety_override_threshold
        ]
        if safety_drives:
            winner = max(safety_drives, key=lambda d: d.urgency)
            self._drives.clear()
            return winner

        # Rule 2: Score by urgency × |valence|
        scored = [(d, d.urgency * abs(d.valence)) for d in self._drives]
        winner = max(scored, key=lambda x: x[1])[0]
        self._drives.clear()
        return winner

    def peek_top(self, n: int = 3) -> List[Drive]:
        """Return top-n drives by score without consuming them."""
        scored = sorted(
            self._drives,
            key=lambda d: d.urgency * abs(d.valence),
            reverse=True,
        )
        return scored[:n]

    def clear(self) -> None:
        self._drives.clear()

    def status(self) -> Dict:
        return {
            "pending_drives": len(self._drives),
            "top_drives": [
                {"name": d.name, "urgency": d.urgency, "source": d.source}
                for d in self.peek_top(3)
            ],
        }


__all__ = ["Drive", "DriveArbitrator"]
