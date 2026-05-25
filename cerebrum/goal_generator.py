"""
cerebrum/goal_generator.py — Intrinsic goal proposal.

The cerebrum generates goals not just from external rewards but from
internal states — homeostatic deficits, curiosity, and long-horizon
planning. This module proposes candidate goals from drive signals and
ranks them for the policy to select from.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
import torch.nn as nn


@dataclass
class Goal:
    """A proposed goal with urgency, expected value, and a latent target."""
    name: str
    urgency: float                          # 0.0 to 1.0
    expected_value: float                   # estimated reward if achieved
    source_drive: str                       # which drive generated this goal
    target_latent: Optional[torch.Tensor] = field(default=None)  # (D,) target state
    horizon: int = 10                       # planning horizon in steps
    metadata: Dict = field(default_factory=dict)


class GoalGenerator(nn.Module):
    """
    Proposes intrinsic goals from homeostatic drives and curiosity signals.

    Goal sources:
        1. Homeostatic deficits → restore drive to setpoint
        2. Curiosity → explore frontier regions of latent space
        3. Skill gaps → practice skills with low success rates
        4. Narrative coherence → pursue goals consistent with self-model

    The generator produces a ranked list of candidate goals. The policy
    selects among them based on current context and emotional state.

    Args:
        latent_dim:     Latent dimensionality.
        max_goals:      Maximum number of candidate goals to maintain.
    """

    def __init__(self, latent_dim: int = 512, max_goals: int = 5) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.max_goals = max_goals

        # Maps drive vector → goal latent target
        self.drive_to_goal = nn.Sequential(
            nn.Linear(6, 128),   # 6 = HomeostaticRegulator.N_DIMS
            nn.GELU(),
            nn.Linear(128, latent_dim),
        )

        self._active_goals: List[Goal] = []

    def generate_from_drives(
        self,
        drive_errors: Dict[str, float],
        homeostasis_vector: torch.Tensor,
    ) -> List[Goal]:
        """
        Generate goals from homeostatic drive deficits.

        Args:
            drive_errors:       Dict of {drive_name: signed_error}.
                                Positive = deficit (need more).
            homeostasis_vector: (6,) current homeostatic state.

        Returns:
            List of Goal objects, sorted by urgency.
        """
        goals = []

        for drive_name, error in drive_errors.items():
            if abs(error) < 0.1:
                continue  # Not urgent enough

            urgency = min(abs(error), 1.0)
            target = self.drive_to_goal(homeostasis_vector.unsqueeze(0)).squeeze(0)

            goal = Goal(
                name=f"restore_{drive_name}",
                urgency=urgency,
                expected_value=urgency * 0.5,
                source_drive=drive_name,
                target_latent=target.detach(),
                metadata={"error": error},
            )
            goals.append(goal)

        return sorted(goals, key=lambda g: g.urgency, reverse=True)

    def generate_curiosity_goal(
        self,
        frontier_direction: Optional[torch.Tensor],
        curiosity_level: float,
    ) -> Optional[Goal]:
        """
        Generate an exploration goal toward the latent space frontier.

        Args:
            frontier_direction: (D,) unit vector toward unexplored region.
            curiosity_level:    Current curiosity drive level [0, 1].

        Returns:
            A curiosity Goal, or None if curiosity is low.
        """
        if curiosity_level < 0.2 or frontier_direction is None:
            return None

        return Goal(
            name="explore_frontier",
            urgency=curiosity_level,
            expected_value=curiosity_level * 0.8,
            source_drive="curiosity",
            target_latent=frontier_direction.detach(),
            horizon=20,
            metadata={"curiosity_level": curiosity_level},
        )

    def update_goals(self, new_goals: List[Goal]) -> None:
        """Merge new goals into the active goal list, keeping top-k."""
        self._active_goals.extend(new_goals)
        self._active_goals.sort(key=lambda g: g.urgency, reverse=True)
        self._active_goals = self._active_goals[:self.max_goals]

    def top_goal(self) -> Optional[Goal]:
        """Return the highest-urgency active goal."""
        return self._active_goals[0] if self._active_goals else None

    def complete_goal(self, name: str) -> None:
        """Mark a goal as completed and remove it."""
        self._active_goals = [g for g in self._active_goals if g.name != name]

    def status(self) -> Dict:
        return {
            "n_active_goals": len(self._active_goals),
            "top_goal": self._active_goals[0].name if self._active_goals else None,
            "goals": [
                {"name": g.name, "urgency": g.urgency, "source": g.source_drive}
                for g in self._active_goals
            ],
        }


__all__ = ["Goal", "GoalGenerator"]
