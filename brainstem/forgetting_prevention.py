"""
brainstem/forgetting_prevention.py — Elastic Weight Consolidation (EWC).

When the brain trains on a new task, gradient updates can overwrite
parameters that were critical for previous tasks. EWC prevents this by
penalizing changes to important parameters proportionally to their
Fisher information (how much past performance depended on them).

    L_total = L_task + lambda * sum_i F_i * (theta_i - theta_star_i)^2

Where:
    F_i         = Fisher information of parameter i (diagonal approximation)
    theta_star  = parameter snapshot from after the previous task
    lambda      = consolidation strength (higher = more conservative)

Call `consolidate()` at the end of each task/phase to snapshot the
current parameters and their importance. The `penalty()` method then
returns the EWC regularization term to add to the actor loss.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn


class EWC:
    """
    Elastic Weight Consolidation for catastrophic forgetting prevention.

    Args:
        model:              The model to protect (typically the policy).
        consolidation_strength: Lambda — how strongly to penalize drift.
                               Higher = more conservative / forgets less
                               but learns new tasks slower.
        fisher_samples:     Number of samples to estimate Fisher information.
        max_tasks:          Maximum number of past task snapshots to retain.
                            Older tasks' importance decays.
    """

    def __init__(
        self,
        model: nn.Module,
        consolidation_strength: float = 100.0,
        fisher_samples: int = 200,
        max_tasks: int = 5,
    ) -> None:
        self.model = model
        self.consolidation_strength = consolidation_strength
        self.fisher_samples = fisher_samples
        self.max_tasks = max_tasks

        # List of (parameter_snapshot, fisher_diagonal) tuples
        self._task_memories: List[Dict[str, torch.Tensor]] = []
        self._consolidated = False

    def consolidate(self, data_loader=None, loss_fn=None) -> None:
        """
        Snapshot current parameters and estimate Fisher information.

        Call this at the END of a task/training phase. After this call,
        `penalty()` will include a term protecting these parameters.

        If data_loader and loss_fn are provided, Fisher is estimated from
        the data. Otherwise, a simpler heuristic is used: parameter magnitude
        as a proxy for importance (larger params = more important).

        Args:
            data_loader: Optional iterable of input batches.
            loss_fn:     Optional callable(model, batch) -> scalar loss.
        """
        # Snapshot parameters
        param_snapshot = {
            name: param.detach().clone()
            for name, param in self.model.named_parameters()
            if param.requires_grad
        }

        # Estimate Fisher information
        if data_loader is not None and loss_fn is not None:
            fisher = self._compute_fisher(data_loader, loss_fn)
        else:
            # Heuristic: use gradient magnitude from recent training as proxy.
            # If no gradients available, use parameter magnitude.
            fisher = {}
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    if param.grad is not None:
                        fisher[name] = param.grad.detach().pow(2).clone()
                    else:
                        # Fallback: param magnitude as importance proxy
                        fisher[name] = param.detach().abs().clone()

        self._task_memories.append({
            "params": param_snapshot,
            "fisher": fisher,
        })

        # Trim old task memories
        if len(self._task_memories) > self.max_tasks:
            self._task_memories.pop(0)

        self._consolidated = True

    def _compute_fisher(self, data_loader, loss_fn) -> Dict[str, torch.Tensor]:
        """Estimate diagonal Fisher information from data."""
        fisher = {
            name: torch.zeros_like(param)
            for name, param in self.model.named_parameters()
            if param.requires_grad
        }

        self.model.eval()
        n_samples = 0
        for batch in data_loader:
            if n_samples >= self.fisher_samples:
                break
            self.model.zero_grad()
            loss = loss_fn(self.model, batch)
            loss.backward()

            for name, param in self.model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    fisher[name] += param.grad.detach().pow(2)
            n_samples += 1

        # Average
        for name in fisher:
            fisher[name] /= max(n_samples, 1)

        self.model.train()
        return fisher

    def penalty(self) -> torch.Tensor:
        """
        Compute the EWC penalty term.

        Returns a scalar tensor to add to the training loss:
            L_ewc = lambda/2 * sum_tasks sum_params F_i * (theta_i - theta_star_i)^2

        Returns 0 if no tasks have been consolidated yet.
        """
        if not self._task_memories:
            return torch.tensor(0.0)

        total_penalty = torch.tensor(0.0)
        device = None

        for task_mem in self._task_memories:
            params_star = task_mem["params"]
            fisher = task_mem["fisher"]

            for name, param in self.model.named_parameters():
                if name not in params_star:
                    continue
                if device is None:
                    device = param.device
                    total_penalty = total_penalty.to(device)

                p_star = params_star[name].to(param.device)
                f = fisher[name].to(param.device)

                total_penalty = total_penalty + (f * (param - p_star).pow(2)).sum()

        return 0.5 * self.consolidation_strength * total_penalty

    @property
    def n_consolidated_tasks(self) -> int:
        return len(self._task_memories)

    @property
    def is_active(self) -> bool:
        return self._consolidated and len(self._task_memories) > 0

    def status(self) -> dict:
        return {
            "n_tasks": self.n_consolidated_tasks,
            "is_active": self.is_active,
            "consolidation_strength": self.consolidation_strength,
        }


__all__ = ["EWC"]
