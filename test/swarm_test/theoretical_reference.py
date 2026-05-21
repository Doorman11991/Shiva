import torch
import torch.nn as nn
import torch.nn.functional as F
import math
#Automated tests as packages
class ReferenceCrossAttentionWorkspace(nn.Module):
    """
    A standalone reference implementation of a Global Workspace using cross-attention.
    Validates that a central query can selectively route salient information 
    from local processors instead of executing a blind arithmetic average.
    """
    def __init__(self, latent_dim: int, num_heads: int = 8):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_heads = num_heads
        self.d_k = latent_dim // num_heads
        
        self.q_proj = nn.Parameter(torch.randn(1, latent_dim))
        self.k_proj = nn.Linear(latent_dim, latent_dim, bias=False)
        self.v_proj = nn.Linear(latent_dim, latent_dim, bias=False)
        self.out_proj = nn.Linear(latent_dim, latent_dim)

    def aggregate(self, node_latents: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        N, D = node_latents.shape
        H, d_k = self.num_heads, self.d_k
        
        K = self.k_proj(node_latents).view(N, H, d_k).transpose(0, 1)
        V = self.v_proj(node_latents).view(N, H, d_k).transpose(0, 1)
        Q = self.q_proj.view(1, H, d_k).transpose(0, 1)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
        weights = F.softmax(scores, dim=-1)
        
        out = torch.matmul(weights, V).transpose(0, 1).contiguous().view(1, D).squeeze(0)
        return self.out_proj(out), weights

def compute_reference_diversity_loss(latents: torch.Tensor) -> torch.Tensor:
    """
    An independent baseline calculation of contrastive diversity loss.
    L = -mean_{i != j} ||z_i - z_j||_2 over L2-normalized vectors.
    """
    z_norm = F.normalize(latents, p=2, dim=1)
    N = z_norm.shape[0]
    if N < 2:
        return torch.tensor(0.0)
    
    diff = z_norm.unsqueeze(0) - z_norm.unsqueeze(1)
    dist_matrix = torch.norm(diff, p=2, dim=-1)
    
    mask = torch.triu(torch.ones(N, N), diagonal=1)
    total_pairs = mask.sum()
    
    mean_pairwise_distance = (dist_matrix * mask).sum() / total_pairs
    return -mean_pairwise_distance
