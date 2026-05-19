from __future__ import annotations
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from core.interfaces import IActor, IEpisodicMemory


# ---------------------------------------------------------------------------
# Continuous actor
# ---------------------------------------------------------------------------

class ContinuousActor(IActor):
    """
    Gaussian policy: outputs (μ, log σ) and samples via the reparameterisation
    trick with a tanh squashing transformation.

    Log-prob correction for the squashing (original formulation preserved):
        log π(a|s) = log N(x|μ,σ) − Σ log(1 − tanh²(x) + ε)
    """

    def __init__(self, d_model: int, action_dim: int) -> None:
        super().__init__()
        self.mu = nn.Linear(d_model, action_dim)
        self.log_std = nn.Linear(d_model, action_dim)

    def forward(
        self, state_features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (mu, log_std) — the distribution parameters."""
        mu = self.mu(state_features)
        log_std = torch.clamp(self.log_std(state_features), -20, 2)
        return mu, log_std

    def sample(
        self, state_features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns (tanh(x_t), log_prob) via the reparameterisation trick.

          x_t ~ N(μ, σ)                            (reparameterised)
          a   = tanh(x_t)                           (squashed action)
          log π = log N(x_t|μ,σ) − Σ log(1−a²+ε)  (corrected log-prob)
        """
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
    """
    Twin Q-networks that jointly estimate action-value.

    Using two critics and taking their minimum reduces overestimation bias
    (Fujimoto et al., 2018).  Each critic is a two-layer MLP:
        Q(s,a) = W₂ · GELU(W₁ · [z ‖ a])
    """

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
        """Returns (Q1(s,a), Q2(s,a))."""
        sa = torch.cat([z_global, action], dim=-1)
        return self.critic1(sa), self.critic2(sa)


# ---------------------------------------------------------------------------
# Continuous SAC policy
# ---------------------------------------------------------------------------

class ContinuousSACPolicy(nn.Module):
    """
    Soft Actor-Critic policy with dual-actor blending, identity-grounded
    consciousness, and episodic memory integration.

    Architecture:
        z        = backbone.forward_pass(state)         # (B, T, D)
        z_global = mean_pool(z)                         # (B, D)
        z_id     = memory.get_identity_context(z_global)
        z_c      = z_global + z_id                      # conscious latent
        g        = σ(gate(z_c))                         # blending gate ∈ (0,1)
        μ        = g·μ₁ + (1−g)·μ₂                    # blended mean
        log σ    = g·log σ₁ + (1−g)·log σ₂             # blended log-std
        a        = tanh(x_t),   x_t ~ N(μ, exp(log σ))

    Args:
        backbone:  Feature extractor exposing `forward_pass(x) → Tensor`.
        actor1:    Primary IActor (injected).
        actor2:    Secondary IActor (injected).
        memory:    IEpisodicMemory for identity context (injected).
        critic:    DoubleQCritic for Q-value queries (injected).
        d_model:   Latent dimensionality.
    """

    def __init__(
        self,
        backbone: nn.Module,
        actor1: IActor,
        actor2: IActor,
        memory: IEpisodicMemory,
        critic: DoubleQCritic,
        d_model: int,
        swarm=None
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.actor1 = actor1
        self.actor2 = actor2
        self.memory = memory
        self.critic = critic
        self.swarm=swarm

        # Blending gate: maps the conscious latent to a scalar in (0,1).
        self.gate = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid(),
        )

    def get_action(
        self, state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample an action from the blended, identity-grounded policy.

        Returns:
            final_action:   Squashed action tensor (B, action_dim).
            final_log_prob: Log-probability under the blended policy (B, 1).
            g:              Blending gate value (B, 1) — interpretability hook.
        """
        z = self.backbone.forward_pass(state)          # (B, T, D)
        z_global = z.mean(dim=1)                        # (B, D)
        z_id = self.memory.get_identity_context(z_global)
        z_conscious = z_global + z_id
        if self.swarm is not None:
            self.swarm.update_node_latent(
                "local_node",
                z_conscious.mean(dim=0)
            )

            consensus,_=self.swarm.step()

            z_conscious=(
                z_conscious
                +
                consensus.unsqueeze(0)
            )
        g = self.gate(z_conscious)                      # (B, 1)

        mu1, log_std1 = self.actor1.forward(z_conscious)
        mu2, log_std2 = self.actor2.forward(z_conscious)

        blended_mu = g * mu1 + (1 - g) * mu2
        blended_log_std = g * log_std1 + (1 - g) * log_std2

        std = torch.exp(blended_log_std)
        dist = Normal(blended_mu, std)
        x_t = dist.rsample()
        final_action = torch.tanh(x_t)
        final_log_prob = dist.log_prob(x_t) - torch.log(
            1 - final_action.pow(2) + 1e-6
        )
        return final_action, final_log_prob.sum(dim=-1, keepdim=True), g

    def evaluate_q(
        self, state: torch.Tensor, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Evaluate Q-values for a (state, action) pair.

        The conscious latent is recomputed here so critics always see the
        same representation as the actors. Returns (Q1, Q2).
        """
        z = self.backbone.forward_pass(state).mean(dim=1)
        z_id = self.memory.get_identity_context(z)
        z_conscious = z + z_id
        return self.critic(z_conscious, action)


# ---------------------------------------------------------------------------
# Discrete valence policy
# ---------------------------------------------------------------------------

class DiscreteValencePolicy(nn.Module):
    """
    Discrete action policy modulated by affective valence, with an
    empowerment estimator based on mutual information.

    Empowerment formulation (original preserved):
        I(a; s') ≈ Σ_a π(a|s) · log[π(a|s) / marginal(a) + ε]

    where marginal(a) = mean over the batch of π(a|s).
    """

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
        """
        Estimates empowerment as mutual information between actions and states.

          I ≈ Σ π(a|s) · log[π(a|s) / (marginal(a) + ε) + ε]
        """
        marginal = action_probs.mean(dim=0)
        mi = torch.sum(
            action_probs * torch.log(action_probs / (marginal + 1e-9) + 1e-9),
            dim=-1,
        )
        return mi.mean()

    def forward(
        self, state: torch.Tensor, valence: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns (action_probabilities, empowerment_score).

        Valence shifts the logits before softmax, biasing action selection
        toward states the agent finds emotionally favourable.
        """
        logits = self.actor(state)
        action_probs = F.softmax(logits + valence, dim=-1)
        empowerment = self.get_empowerment(action_probs)
        return action_probs, empowerment
