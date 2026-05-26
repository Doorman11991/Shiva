"""
cerebellum/skill_library.py — Cached motor programs.

The cerebellum stores learned motor programs — pre-compiled action
sequences for frequently-used skills. Once a skill is learned, it can
be executed without conscious planning, reducing cognitive load.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class Skill:
    """A cached action sequence with metadata."""

    def __init__(
        self,
        name: str,
        action_sequence: torch.Tensor,
        trigger_latent: torch.Tensor,
        success_rate: float = 0.0,
        use_count: int = 0,
    ) -> None:
        self.name = name
        self.action_sequence = action_sequence.detach().to('cpu')
        self.trigger_latent = trigger_latent.detach().to('cpu')
        self.success_rate = success_rate
        self.use_count = use_count

    def update_success(self, succeeded: bool) -> None:
        """EMA update of success rate."""
        self.use_count += 1
        self.success_rate = 0.9 * self.success_rate + 0.1 * float(succeeded)


class SkillLibrary(nn.Module):
    """
    Stores and retrieves cached motor programs (skills).

    Skills are retrieved by matching the current latent state to the
    skill's trigger latent via cosine similarity. High-similarity matches
    above a threshold trigger automatic skill execution.

    Args:
        latent_dim:         Latent dimensionality.
        max_skills:         Maximum number of stored skills.
        retrieval_threshold: Cosine similarity threshold for skill retrieval.
        min_success_rate:   Skills below this success rate are pruned.
    """

    def __init__(
        self,
        latent_dim: int = 512,
        max_skills: int = 128,
        retrieval_threshold: float = 0.85,
        min_success_rate: float = 0.3,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.max_skills = max_skills
        self.retrieval_threshold = retrieval_threshold
        self.min_success_rate = min_success_rate
        self._skills: Dict[str, Skill] = {}

    def store(
        self,
        name: str,
        action_sequence: torch.Tensor,
        trigger_latent: torch.Tensor,
    ) -> None:
        """
        Store a new skill or update an existing one.

        If at capacity, the least-used skill is evicted.
        """
        if len(self._skills) >= self.max_skills and name not in self._skills:
            # Evict least-used skill
            least_used = min(self._skills, key=lambda k: self._skills[k].use_count)
            del self._skills[least_used]

        self._skills[name] = Skill(name, action_sequence, trigger_latent)

    def retrieve(
        self,
        z_current: torch.Tensor,
    ) -> Optional[Tuple[str, torch.Tensor]]:
        """
        Retrieve the best matching skill for the current latent state.

        Args:
            z_current: (D,) current latent vector.

        Returns:
            (skill_name, action_sequence) if a match above threshold is found,
            else None.
        """
        if not self._skills:
            return None

        z = F.normalize(z_current.detach().to('cpu').flatten(), dim=0)
        best_name, best_sim = None, -1.0

        for name, skill in self._skills.items():
            trigger = F.normalize(skill.trigger_latent.flatten(), dim=0)
            sim = float(torch.dot(z, trigger).item())
            if sim > best_sim:
                best_sim = sim
                best_name = name

        if best_sim >= self.retrieval_threshold and best_name is not None:
            skill = self._skills[best_name]
            skill.use_count += 1
            return best_name, skill.action_sequence

        return None

    def prune(self) -> List[str]:
        """Remove skills below the minimum success rate. Returns pruned names."""
        to_prune = [
            name for name, skill in self._skills.items()
            if skill.use_count > 10 and skill.success_rate < self.min_success_rate
        ]
        for name in to_prune:
            del self._skills[name]
        return to_prune

    def update_skill_outcome(self, name: str, succeeded: bool) -> None:
        if name in self._skills:
            self._skills[name].update_success(succeeded)

    def stats(self) -> Dict:
        return {
            "n_skills": len(self._skills),
            "skill_names": list(self._skills.keys()),
            "mean_success_rate": (
                sum(s.success_rate for s in self._skills.values()) / len(self._skills)
                if self._skills else 0.0
            ),
        }


__all__ = ["Skill", "SkillLibrary"]
