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
        nn.init.zeros_(self.net[-1].weight) # type: ignore
        nn.init.zeros_(self.net[-1].bias) # type: ignore

    def forward(self, x):
        return torch.sigmoid(self.net(x))

class TransformerEncoderBlock(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_k = d_model // num_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        ff_dim = d_model * 4
        self.ff_net = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(), # GELU is a common choice, can be replaced by the custom sigmoid activation if needed
            nn.Linear(ff_dim, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.attn_gate_net = GateHyperNetwork(d_model)
        self.ff_gate_net = GateHyperNetwork(d_model)

        self.emotional_gate = nn.Parameter(torch.ones(1))

    def compute_multi_head_attention(self, x, bias_shift: float | torch.Tensor = 0.0):
        batch_size, seq_len, _ = x.shape
        Q, K, V = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        Q = Q.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        scores = (Q @ K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if isinstance(bias_shift, torch.Tensor) or bias_shift != 0.0:
            scores = scores + bias_shift
            
        attn_weights = torch.softmax(scores, dim=-1)
        context = attn_weights @ V
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        return self.out_proj(context)

    def compute_dynamic_gates(self, x):
        gate_attn = self.attn_gate_net(x)
        gate_ff = self.ff_gate_net(x)
        return gate_attn, gate_ff

    def forward_pass(self, x, valence=None):
        gate_attn, gate_ff = self.compute_dynamic_gates(x)
        
        bias_shift = 0.0
        if valence is not None:
            bias_shift = valence * self.emotional_gate
            if isinstance(bias_shift, torch.Tensor):
                while bias_shift.dim() < 4:
                    bias_shift = bias_shift.unsqueeze(-1)
                    
        attn_out = self.compute_multi_head_attention(x, bias_shift=bias_shift)
        x = x + (gate_attn * attn_out)
        x = self.norm1(x)
        ff_out = self.ff_net(x)
        x = x + (gate_ff * ff_out)
        x = self.norm2(x)
        return x