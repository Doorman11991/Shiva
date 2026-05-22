import torch
import unittest
from swarm.SwarmAlgorithmWorkspace import SwarmCoordinator
from .theoretical_reference import compute_reference_diversity_loss

class TestSwarmDifferentialOracles(unittest.TestCase):
    def setUp(self):
        self.latent_dim = 512
        self.n_nodes = 4
        self.num_heads = 8
        
        # Instantiate your developed codebase module
        self.production_system = SwarmCoordinator(
            latent_dim=self.latent_dim, 
            n_nodes=self.n_nodes, 
            num_heads=self.num_heads
        )

    def test_mathematical_equivalence(self):
        shared_input_latents = torch.randn(self.n_nodes, self.latent_dim)
        
        # Stream data across your active production framework components
        for i in range(self.n_nodes):
            self.production_system.update_node_latent(f"node_{i}", shared_input_latents[i])
        
        _, prod_div_loss = self.production_system.step()
        ref_div_loss = compute_reference_diversity_loss(shared_input_latents)
        self.assertTrue(
            torch.allclose(prod_div_loss, ref_div_loss, atol=1e-5),
            f"Telemetry drift detected: Production={prod_div_loss.item()} vs Reference={ref_div_loss.item()}"
        )

if __name__ == "__main__":
    unittest.main()
