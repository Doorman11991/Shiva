"""
cerebrum/chip_policy.py — Voluntary action selection.

The cerebrum is the seat of voluntary, conscious action. This module
implements the dual-actor SAC policy that selects actions based on the
conscious latent — a representation grounded in identity (hippocampus),
modulated by emotion (amygdala), and coordinated with the swarm
(cerebellum).

Moved from: core/chip_policy.py
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from interfaces.base import IActor, IEpisodicMemory


# ---------------------------------------------------------------------------
# Task vocabulary (personality / task conditioning)
# ---------------------------------------------------------------------------

TASK_VOCAB: dict[str, int] = {
    "general":  0,
    "research": 1,
    "browse":   2,
    "code":     3,
    "voice":    4,
    "memory":   5,
    "training": 6,
}
NUM_TASKS = len(TASK_VOCAB)


def task_id_for(name: str) -> int:
    """Resolve a task label to its index. Unknown labels map to 'general'."""
    return TASK_VOCAB.get(name, TASK_VOCAB["general"])


# ---------------------------------------------------------------------------
# Continuous actor
# ---------------------------------------------------------------------------

class ContinuousActor(IActor):
    """
    Gaussian policy: outputs (μ, log σ); samples via the reparameterisation
    trick with tanh squashing.
    """

    def __init__(self, d_model: int, action_dim: int) -> None:
        super().__init__()
        self.mu = nn.Linear(d_model, action_dim)
        self.log_std = nn.Linear(d_model, action_dim)

    def forward(self, state_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mu = self.mu(state_features)
        log_std = torch.clamp(self.log_std(state_features), -20, 2)
        return mu, log_std

    def sample(self, state_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mu, log_std = self.forward(state_features)
        std = torch.exp(log_std)
        dist = Normal(mu, std)
        x_t = dist.rsample()
        action = torch.tanh(x_t)
        log_prob = dist.log_prob(x_t) - torch.log(1 - action.pow(2) + 1e-6)
        return action, log_prob.sum(dim=-1, keepdim=True)


# ---------------------------------------------------------------------------
# Double Q-critic
# ---------------------------------------------------------------------------

class DoubleQCritic(nn.Module):
    """Twin Q-networks. Critic loss uses min(Q1, Q2) to reduce overestimation."""

    def __init__(self, d_model: int, action_dim: int) -> None:
        super().__init__()
        self.critic1 = nn.Sequential(
            nn.Linear(d_model + action_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )
        self.critic2 = nn.Sequential(
            nn.Linear(d_model + action_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )

    def forward(
        self, z_global: torch.Tensor, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        sa = torch.cat([z_global, action], dim=-1)
        return self.critic1(sa), self.critic2(sa)


# ---------------------------------------------------------------------------
# Continuous SAC policy
# ---------------------------------------------------------------------------

class ContinuousSACPolicy(nn.Module):
    """
    Dual-actor SAC with identity-grounded conscious latent and optional
    task conditioning.

    Conscious latent formation:
        z_task   = task_embed(task_id).unsqueeze(1)       # (B, 1, D)
        z_input  = concat([z_task, state], dim=1)         # (B, 1+T, D)
        z        = backbone.forward_pass(z_input)         # (B, 1+T, D)  [thalamus]
        z_global = z.mean(dim=1)                          # (B, D)
        z_id     = memory.get_identity_context(z_global)  # [hippocampus]
        z_c      = z_global + z_id
        g        = σ(gate(z_c))
        μ        = g·μ₁ + (1−g)·μ₂
        a        = tanh(x_t),  x_t ~ N(μ, exp(log σ))
    """

    def __init__(
        self,
        backbone: nn.Module,
        actor1: IActor,
        actor2: IActor,
        memory: IEpisodicMemory,
        critic: DoubleQCritic,
        d_model: int,
        swarm: Optional[nn.Module] = None,
        num_tasks: int = NUM_TASKS,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.actor1 = actor1
        self.actor2 = actor2
        self.memory = memory
        self.critic = critic
        self.swarm = swarm
        self.d_model = d_model

        self.task_embed = nn.Embedding(num_tasks, d_model)
        nn.init.zeros_(self.task_embed.weight)

        self.gate = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid(),
        )

    def _encode(
        self, state: torch.Tensor, task_id: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if task_id is not None:
            task_id = task_id.to(state.device)
            if task_id.dim() == 0:
                task_id = task_id.expand(state.size(0))
            task_token = self.task_embed(task_id).unsqueeze(1)
            state = torch.cat([task_token, state], dim=1)

        z = self.backbone.forward_pass(state)
        z_global = z.mean(dim=1)
        z_id = self.memory.get_identity_context(z_global)
        z_conscious = z_global + z_id

        if self.swarm is not None:
            self.swarm.update_node_latent("local_node", z_conscious.mean(dim=0))
            consensus, _ = self.swarm.step()
            z_conscious = z_conscious + consensus.unsqueeze(0)

        return z_conscious

    def get_action(
        self,
        state: torch.Tensor,
        task_id: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z_conscious = self._encode(state, task_id)
        g = self.gate(z_conscious)

        mu1, log_std1 = self.actor1.forward(z_conscious)
        mu2, log_std2 = self.actor2.forward(z_conscious)

        blended_mu = g * mu1 + (1 - g) * mu2
        blended_log_std = g * log_std1 + (1 - g) * log_std2

        std = torch.exp(blended_log_std)
        dist = Normal(blended_mu, std)
        x_t = dist.rsample()
        final_action = torch.tanh(x_t)
        final_log_prob = dist.log_prob(x_t) - torch.log(1 - final_action.pow(2) + 1e-6)
        return final_action, final_log_prob.sum(dim=-1, keepdim=True), g

    def evaluate_q(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        task_id: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        z_conscious = self._encode(state, task_id)
        return self.critic(z_conscious, action)


class DiscreteValencePolicy(nn.Module):
    """Legacy discrete policy with affective valence shifting."""

    def __init__(self, state_dim: int, action_dim: int) -> None:
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Linear(512, action_dim),
        )
        self.value_manifold = nn.Linear(state_dim, 1)

    def get_empowerment(self, action_probs: torch.Tensor) -> torch.Tensor:
        marginal = action_probs.mean(dim=0)
        mi = torch.sum(
            action_probs * torch.log(action_probs / (marginal + 1e-9) + 1e-9),
            dim=-1,
        )
        return mi.mean()

    def forward(
        self, state: torch.Tensor, valence: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.actor(state)
        action_probs = F.softmax(logits + valence, dim=-1)
        empowerment = self.get_empowerment(action_probs)
        return action_probs, empowerment


__all__ = [
    "ContinuousActor", "ContinuousSACPolicy", "DiscreteValencePolicy",
    "DoubleQCritic", "TASK_VOCAB", "NUM_TASKS", "task_id_for",
]
