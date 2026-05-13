import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from core.transformer_architecture import TransformerEncoderBlock
from core.episodic_memory import EpisodicMemory
class ContinuousActor(nn.Module):
    def __init__(self, d_model, action_dim):
        super().__init__()
        self.mu = nn.Linear(d_model, action_dim)
        self.log_std = nn.Linear(d_model, action_dim)
        
    def forward(self, state_features):
        mu = self.mu(state_features)
        log_std = self.log_std(state_features)
        log_std = torch.clamp(log_std, -20, 2)
        return mu, log_std

    def sample(self, state_features):
        mu, log_std = self.forward(state_features)
        std = torch.exp(log_std)
        normal_dist = Normal(mu, std)
        x_t = normal_dist.rsample() 
        action = torch.tanh(x_t)
        log_prob = normal_dist.log_prob(x_t)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return action, log_prob

class ContinuousSACPolicy(nn.Module):
    def __init__(self, d_model, action_dim, num_heads=8):
        super().__init__()
        self.backbone = TransformerEncoderBlock(d_model, num_heads)
        self.actor1 = ContinuousActor(d_model, action_dim) # e.g., Stability expert
        self.actor2 = ContinuousActor(d_model, action_dim) # e.g., Goal-reaching expert
        self.memory=EpisodicMemory()
        self.gate = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid()
        )
        self.critic1 = nn.Sequential(
            nn.Linear(d_model + action_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1)
        )
        self.critic2 = nn.Sequential(
            nn.Linear(d_model + action_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1)
        )

    def get_action(self, state):
        z = self.backbone.forward_pass(state)
        z_global = z.mean(dim=1) 
        
        identity_context=self.memory.get_identity_context(z_global)
        z_conscious=z_global+identity_context
        g = self.gate(z_conscious)
        mu1, log_std1 = self.actor1(z_conscious)
        mu2, log_std2 = self.actor2(z_conscious)
        blended_mu = g * mu1 + (1 - g) * mu2
        blended_log_std = g * log_std1 + (1 - g) * log_std2
        
        std = torch.exp(blended_log_std)
        normal_dist = Normal(blended_mu, std)
        x_t = normal_dist.rsample()
        final_action = torch.tanh(x_t)

        final_log_prob = normal_dist.log_prob(x_t)
        final_log_prob -= torch.log(1 - final_action.pow(2) + 1e-6)
        final_log_prob = final_log_prob.sum(dim=-1, keepdim=True)
        
        return final_action, final_log_prob, g

    def evaluate_q(self, state, action):
        z = self.backbone.forward_pass(state).mean(dim=1)
        sa_pair = torch.cat([z, action], dim=-1)
        return self.critic1(sa_pair), self.critic2(sa_pair)


class DiscreteValencePolicy(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Linear(512, action_dim)
        )
        self.value_manifold = nn.Linear(state_dim, 1)

    def get_empowerment(self, action_probs):
        marginal = action_probs.mean(dim=0)
        mi = torch.sum(action_probs * torch.log(action_probs / (marginal + 1e-9) + 1e-9), dim=-1)
        return mi.mean()

    def forward(self, state, valence):
        logits = self.actor(state)
        action_probs = F.softmax(logits + valence, dim=-1)
        
        empowerment = self.get_empowerment(action_probs)
        return action_probs, empowerment
