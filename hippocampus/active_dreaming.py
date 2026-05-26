"""
hippocampus/active_dreaming.py — Counterfactual generation by decision perturbation.

Biological role
~~~~~~~~~~~~~~~
During REM sleep, the brain doesn't just replay memories — it *remixes*
them. Dreams combine elements from different episodes, insert impossible
scenarios, and explore "what if" paths not taken. This is thought to be
the mechanism by which the brain generalises beyond direct experience
and generates creative solutions.

How this differs from the existing DreamCycle
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The existing dream_cycle.py adds Gaussian noise to states. That creates
mild variations (slightly different states). Active dreaming is more
targeted: it identifies KEY DECISION POINTS in an episode (where the
policy was most uncertain or where the world model's prediction error
was highest) and perturbs the ACTION at that point, then rolls out the
consequences through the world model. This generates genuinely new
trajectories the agent has never experienced.

Design
~~~~~~
    1. Sample a significant episode from episodic memory.
    2. Identify decision points: ticks where Q-spread was large (the
       policy was torn between options) or prediction error spiked.
    3. At each decision point, sample K alternative actions.
    4. Roll out each alternative through the world model for H steps.
    5. Score each counterfactual trajectory with the plan evaluator.
    6. Store the best counterfactual as a new synthetic episode with
       boosted significance (high-value unrealised potential).
    7. Inject high-regret counterfactuals as training targets for the
       policy (hindsight learning).

This is where creativity emerges: the agent imagines paths not taken,
evaluates them, and learns from the best ones — even though they never
actually happened.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class CounterfactualTrajectory:
    """One imagined alternative trajectory."""
    decision_tick: int              # which tick in the episode was perturbed
    original_action: torch.Tensor   # action that was actually taken
    alternative_action: torch.Tensor  # the imagined alternative
    trajectory: torch.Tensor        # (H, D) rolled-out latent states
    estimated_value: float          # value estimate from PlanEvaluator
    regret: float                   # value(counterfactual) - value(actual)


class ActiveDreamer:
    """
    Generates counterfactual trajectories by perturbing key decisions.

    Args:
        world_model:        LatentDynamicsModel for rollouts.
        plan_evaluator:     PlanEvaluator for trajectory scoring.
        action_dim:         Action space dimensionality.
        horizon:            Rollout length per counterfactual.
        n_alternatives:     Number of alternative actions per decision point.
        min_significance:   Only dream about episodes above this significance.
        max_decision_points: Maximum decision points to perturb per episode.
    """

    def __init__(
        self,
        world_model: nn.Module,
        plan_evaluator: nn.Module,
        action_dim: int = 4,
        horizon: int = 5,
        n_alternatives: int = 4,
        min_significance: float = 0.3,
        max_decision_points: int = 3,
    ) -> None:
        self.world_model = world_model
        self.plan_evaluator = plan_evaluator
        self.action_dim = action_dim
        self.horizon = horizon
        self.n_alternatives = n_alternatives
        self.min_significance = min_significance
        self.max_decision_points = max_decision_points

        self._dream_count: int = 0
        self._total_counterfactuals: int = 0
        self._best_regret_seen: float = 0.0

    # ------------------------------------------------------------------
    # Identify decision points
    # ------------------------------------------------------------------

    def _find_decision_points(
        self,
        episode_states: torch.Tensor,
    ) -> List[int]:
        """
        Identify ticks where decisions matter most.

        Strategy: compute the local variance in the latent trajectory.
        High variance = the state was changing rapidly = a decision was
        being made that had large consequences.

        Args:
            episode_states: (T, D) state trajectory.

        Returns:
            List of tick indices (sorted by importance, descending).
        """
        T, D = episode_states.shape
        if T < 3:
            return [0]

        # Local variance: ||z_t - z_{t-1}||^2 at each tick
        diffs = (episode_states[1:] - episode_states[:-1]).pow(2).sum(dim=-1)  # (T-1,)

        # Top-k by local change magnitude
        k = min(self.max_decision_points, T - 1)
        _, top_idx = diffs.topk(k)
        return sorted(top_idx.tolist())

    # ------------------------------------------------------------------
    # Generate counterfactuals
    # ------------------------------------------------------------------

    @torch.no_grad()
    def dream_episode(
        self,
        episode_states: torch.Tensor,
    ) -> List[CounterfactualTrajectory]:
        """
        Generate counterfactual trajectories for one episode.
        """
        # Move episode to the world model's device (episodes are stored as CPU tensors).
        wm_device = next(self.world_model.parameters()).device
        episode_states = episode_states.to(wm_device)
        T, D = episode_states.shape
        decision_points = self._find_decision_points(episode_states)

        counterfactuals: List[CounterfactualTrajectory] = []

        for dp in decision_points:
            z_at_dp = episode_states[dp].unsqueeze(0)  # (1, D)
            device = episode_states.device

            if dp < T - 1:
                original_direction = episode_states[dp + 1] - episode_states[dp]
                original_action = original_direction[:self.action_dim]
            else:
                original_action = torch.zeros(self.action_dim, device=device)

            actual_future = episode_states[dp:dp + self.horizon + 1]
            if actual_future.shape[0] > 1:
                actual_value = float(
                    self.plan_evaluator.evaluate_trajectory(
                        actual_future.unsqueeze(0)
                    ).item()
                )
            else:
                actual_value = 0.0

            for k in range(self.n_alternatives):
                alt_action = torch.randn(1, self.action_dim, device=device) * 0.5
                trajectory = [z_at_dp.squeeze(0)]
                z = z_at_dp
                for h in range(self.horizon):
                    a = alt_action if h == 0 else torch.zeros(1, self.action_dim, device=device)
                    z = self.world_model(z, a)
                    trajectory.append(z.squeeze(0))

                traj_tensor = torch.stack(trajectory)  # (H+1, D)
                traj_value = float(
                    self.plan_evaluator.evaluate_trajectory(
                        traj_tensor.unsqueeze(0)
                    ).item()
                )
                regret = traj_value - actual_value

                cf = CounterfactualTrajectory(
                    decision_tick=dp,
                    original_action=original_action,
                    alternative_action=alt_action.squeeze(0),
                    trajectory=traj_tensor,
                    estimated_value=traj_value,
                    regret=regret,
                )
                counterfactuals.append(cf)

        # Filter to positive-regret counterfactuals (better alternatives only)
        good_cfs = [cf for cf in counterfactuals if cf.regret > 0]
        good_cfs.sort(key=lambda cf: cf.regret, reverse=True)

        self._total_counterfactuals += len(good_cfs)
        if good_cfs:
            self._best_regret_seen = max(self._best_regret_seen, good_cfs[0].regret)

        return good_cfs

    # ------------------------------------------------------------------
    # Full dream cycle: sample + generate + store
    # ------------------------------------------------------------------

    def run(
        self,
        episodic_memory,
        batch_size: int = 4,
    ) -> Dict:
        """
        Run a full active dreaming session.

        1. Sample significant episodes.
        2. Generate counterfactuals for each.
        3. Store the best counterfactual trajectories back into memory
           as synthetic episodes with boosted significance.

        Args:
            episodic_memory: The hippocampus EpisodicMemory instance.
            batch_size:      Number of episodes to dream about.

        Returns:
            Summary dict of the dreaming session.
        """
        dream_batch = episodic_memory.get_dream_batch(batch_size)
        if dream_batch is None:
            return {"status": "insufficient_memories", "n_counterfactuals": 0}

        total_cfs = 0
        stored = 0

        for i in range(dream_batch.shape[0]):
            episode = dream_batch[i]  # (T, D)
            counterfactuals = self.dream_episode(episode)

            # Store top counterfactual as a synthetic episode
            if counterfactuals:
                best = counterfactuals[0]
                # Synthetic episodes get boosted significance via high
                # empowerment score (unrealised potential = empowering)
                synthetic_valence = torch.full(
                    (best.trajectory.shape[0], 1),
                    min(best.regret, 1.0),
                )
                episodic_memory.store_episode(
                    state_sequence=best.trajectory,
                    valence_sequence=synthetic_valence,
                    empowerment_score=min(best.regret * 2.0, 1.0),
                )
                stored += 1
                total_cfs += len(counterfactuals)

        self._dream_count += 1

        return {
            "status": "completed",
            "episodes_dreamed": batch_size,
            "n_counterfactuals": total_cfs,
            "n_stored": stored,
            "best_regret": self._best_regret_seen,
            "total_dreams": self._dream_count,
        }

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def status(self) -> Dict:
        return {
            "dream_count": self._dream_count,
            "total_counterfactuals": self._total_counterfactuals,
            "best_regret_seen": self._best_regret_seen,
        }


__all__ = ["ActiveDreamer", "CounterfactualTrajectory"]
