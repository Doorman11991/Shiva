import torch
# Updated to use relative import to avoid hyphenated directory syntax errors
from .theoretical_references import ReferenceCrossAttentionWorkspace, compute_reference_diversity_loss

def test_shared_workspace_selective_routing():
    latent_dim = 64
    workspace = ReferenceCrossAttentionWorkspace(latent_dim=latent_dim, num_heads=2)
    
    z0 = torch.zeros(latent_dim)
    z0[0:10] = 10.0
    z_flat = torch.ones(latent_dim) * 0.1
    stacked = torch.stack([z0, z_flat, z_flat, z_flat])
    
    consensus, weights = workspace.aggregate(stacked)
    blind_average = stacked.mean(dim=0)
    
    functional_deviation = torch.norm(consensus - blind_average, p=2).item()
    assert functional_deviation > 1e-3


def test_copycat_prevention_metrics():
    latent_dim = 64
    copycat_vector = torch.randn(latent_dim)
    copycat_stack = torch.stack([copycat_vector for _ in range(4)])
    
    copycat_loss = compute_reference_diversity_loss(copycat_stack)
    assert abs(copycat_loss.item()) < 1e-6
    
    z0 = torch.zeros(latent_dim); z0[0] = 1.0
    z1 = torch.zeros(latent_dim); z1[1] = 1.0
    orthogonal_stack = torch.stack([z0, z1])
    diverse_loss = compute_reference_diversity_loss(orthogonal_stack)
    
    assert diverse_loss.item() < copycat_loss.item()
