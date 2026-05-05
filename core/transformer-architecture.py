import torch
import torch.nn as nn
import math

class GateHyperNetwork(nn.Module):
    def __init__(self, d_model, hidden_dim=None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = d_model // 2
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, d_model)
        )
        # ReZero init: Start by predicting ~0.0 so the main block initially acts as an identity function.
        nn.init.zeros_(self.net[-1].weight) # type: ignore
        nn.init.zeros_(self.net[-1].bias) # type: ignore

    def forward(self, x):
        return torch.sigmoid(self.net(x))

class TransformerEncoderBlock(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        limit = math.sqrt(6 / (d_model + d_model))
        
        self.W_q = nn.Parameter(torch.distributions.Uniform(-limit, limit).sample((d_model, d_model)))
        self.b_q = nn.Parameter(torch.zeros(d_model))
        self.W_k = nn.Parameter(torch.distributions.Uniform(-limit, limit).sample((d_model, d_model)))
        self.b_k = nn.Parameter(torch.zeros(d_model))
        self.W_v = nn.Parameter(torch.distributions.Uniform(-limit, limit).sample((d_model, d_model)))
        self.b_v = nn.Parameter(torch.zeros(d_model))
        self.W_o = nn.Parameter(torch.distributions.Uniform(-limit, limit).sample((d_model, d_model)))
        self.b_o = nn.Parameter(torch.zeros(d_model))
        
        ff_dim = d_model * 4
        limit_ff = math.sqrt(2 / d_model)
        self.W_ff1 = nn.Parameter(torch.randn(d_model, ff_dim) * limit_ff)
        self.b_ff1 = nn.Parameter(torch.zeros(ff_dim))
        self.W_ff2 = nn.Parameter(torch.randn(ff_dim, d_model) * limit_ff)
        self.b_ff2 = nn.Parameter(torch.zeros(d_model))
        
        self.gamma1 = nn.Parameter(torch.ones(d_model))
        self.beta1 = nn.Parameter(torch.zeros(d_model))
        self.gamma2 = nn.Parameter(torch.ones(d_model))
        self.beta2 = nn.Parameter(torch.zeros(d_model))
        
        # Initialize HyperNetworks for dynamically predicting the gates
        self.attn_gate_net = GateHyperNetwork(d_model)
        self.ff_gate_net = GateHyperNetwork(d_model)

    def layer_norm(self, x, gamma, beta, eps=1e-6):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return gamma * (x - mean) / (std + eps) + beta

    def compute_multi_head_attention(self, x):
        batch_size, seq_len, _ = x.shape
        Q = (x @ self.W_q) + self.b_q
        K = (x @ self.W_k) + self.b_k
        V = (x @ self.W_v) + self.b_v
        Q = Q.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        scores = (Q @ K.transpose(-2, -1)) / math.sqrt(self.d_k)
        attn_weights = torch.softmax(scores, dim=-1)
        context = attn_weights @ V
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        return (context @ self.W_o) + self.b_o

    def compute_dynamic_gates(self, x):
        # Predicts the current gating weights dynamically based on the input features
        gate_attn = self.attn_gate_net(x)
        gate_ff = self.ff_gate_net(x)
        return gate_attn, gate_ff

    def forward_pass(self, x):
        gate_attn, gate_ff = self.compute_dynamic_gates(x)
        attn_out = self.compute_multi_head_attention(x)
        x = self.layer_norm(x + (gate_attn * attn_out), self.gamma1, self.beta1)
        ff_hidden = (x @ self.W_ff1) + self.b_ff1
        ff_activated = ff_hidden * torch.sigmoid(1.702 * ff_hidden) 
        ff_out = (ff_activated @ self.W_ff2) + self.b_ff2
        x = self.layer_norm(x + (gate_ff * ff_out), self.gamma2, self.beta2)
        return x