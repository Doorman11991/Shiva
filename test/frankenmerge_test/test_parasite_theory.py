import torch
import torch.optim as optim
from .theoretical_references import (
    compute_theoretical_infonce, 
    compute_spatial_distance_matrix, 
    calculate_matrix_correlation
)

def test_invisible_clone_optimization_theory():
    """
    Shorthand: The Invisible Clone Test
    Validates that minimizing InfoNCE forces an unaligned target matrix to mimic 
    the latent logic of a completely frozen third-party host without mutating the host.
    """
    torch.manual_seed(42)
    batch_size, host_dim, target_dim = 16, 64, 32
    
    # Simulate a closed, third-party "Host" activation landscape (Frozen parameters)
    synthetic_host_activations = torch.randn(batch_size, host_dim)
    original_host_snapshot = synthetic_host_activations.clone()
    
    # Define an unaligned, mutable Probe Network projection matrix
    projection_weight = torch.randn(host_dim, target_dim, requires_grad=True)
    optimizer = optim.Adam([projection_weight], lr=1e-1)
    
    # Run an isolated optimization sequence to verify representational capture
    initial_loss = compute_theoretical_infonce(
        torch.matmul(synthetic_host_activations, projection_weight), 
        synthetic_host_activations[:, :target_dim]
    ).item()
    
    for _ in range(20):
        optimizer.zero_grad()
        z_probe = torch.matmul(synthetic_host_activations, projection_weight)
        z_shiva_mock = synthetic_host_activations[:, :target_dim]
        
        loss = compute_theoretical_infonce(z_probe, z_shiva_mock)
        loss.backward()
        optimizer.step()
        
    final_loss = loss.item()
    
    # Assertions validating Black-Box constraints
    assert final_loss < initial_loss, "Theory Error: InfoNCE failed to drive representational convergence."
    assert torch.equal(synthetic_host_activations, original_host_snapshot), \
        "Security Breach: The host's parameters or internal states were altered during training!"


def test_spatial_alignment_and_topographic_fidelity():
    """
    Shorthand: The Spatial Alignment Test
    Mathematically proves that structural geometric shapes can be perfectly transferred
    by verifying that the pairwise relative distances match tightly across dimensional spaces.
    """
    torch.manual_seed(42)
    num_samples, dim = 8, 128
    
    # Generate an explicit geometrical configuration (e.g., clustered points)
    base_points = torch.randn(num_samples, dim)
    
    # Compute the reference distance matrix from the source layer
    host_spatial_matrix = compute_spatial_distance_matrix(base_points)
    
    # Scenario A: Highly Distorted / Scrambled Space
    distorted_points = torch.randn(num_samples, dim * 2)
    poor_spatial_matrix = compute_spatial_distance_matrix(distorted_points)
    poor_correlation = calculate_matrix_correlation(host_spatial_matrix, poor_spatial_matrix)
    
    # Scenario B: Scale & Rotation Transformed Space (Maintains topological shape)
    # Apply an orthogonal projection matrix (rotation) and uniform scaling factor
    orthogonal_matrix, _ = torch.linalg.qr(torch.randn(dim, dim))
    aligned_points = torch.matmul(base_points, orthogonal_matrix) * 3.5
    good_spatial_matrix = compute_spatial_distance_matrix(aligned_points)
    ideal_correlation = calculate_matrix_correlation(host_spatial_matrix, good_spatial_matrix)
    
    # Structural Verification
    assert ideal_correlation > 0.95, f"Structural Mapping Failed: Target space correlation is {ideal_correlation}"
    assert ideal_correlation > poor_correlation, "Metric Failure: Correlation metric failed to distinguish scrambled geometry."
