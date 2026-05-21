import torch
import torch.nn as nn
import torch.nn.functional as F

def compute_theoretical_infonce(z_a: torch.Tensor, z_b: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    """
    An independent functional baseline calculation of InfoNCE contrastive alignment loss.
    Symmetrically aligns representations across a shared hypersphere barrier.
    """
    # L2 normalize both activation sets onto a unit hypersphere
    z_a_norm = F.normalize(z_a, p=2, dim=1)
    z_b_norm = F.normalize(z_b, p=2, dim=1)
    
    # Calculate cosine similarity scaled by the temperature parameter
    logits_a_to_b = torch.matmul(z_a_norm, z_b_norm.T) / temperature
    labels = torch.arange(z_a.shape[0], device=z_a.device)
    
    # Symmetrical Cross-Entropy aggregation
    loss_a = F.cross_entropy(logits_a_to_b, labels)
    loss_b = F.cross_entropy(logits_a_to_b.T, labels)
    return (loss_a + loss_b) / 2

def compute_spatial_distance_matrix(embeddings: torch.Tensor) -> torch.Tensor:
    """
    Constructs a standalone Pairwise L2 Distance Matrix to capture 
    the invariant geometric shape (topology) of a hidden knowledge space.
    """
    # Norm-scaled distance computation via broadcasting: (N, N, D)
    diff = embeddings.unsqueeze(0) - embeddings.unsqueeze(1)
    return torch.norm(diff, p=2, dim=-1)

def calculate_matrix_correlation(matrix_a: torch.Tensor, matrix_b: torch.Tensor) -> float:
    """
    Calculates the Pearson Correlation Coefficient between the upper triangles of two 
    spatial distance matrices to mathematically evaluate topographic preservation.
    """
    # Isolate upper triangles to bypass zero-value diagonals and duplicate pairs
    n = matrix_a.shape[0]
    mask = torch.triu(torch.ones(n, n), diagonal=1).bool()
    
    vec_a = matrix_a[mask]
    vec_b = matrix_b[mask]
    
    # Standard Pearson formulation
    mean_a, mean_b = vec_a.mean(), vec_b.mean()
    dev_a, dev_b = vec_a - mean_a, vec_b - mean_b
    
    num = (dev_a * dev_b).sum()
    den = torch.sqrt((dev_a ** 2).sum() * (dev_b ** 2).sum())
    
    if den.item() == 0:
        return 0.0
    return (num / den).item()
