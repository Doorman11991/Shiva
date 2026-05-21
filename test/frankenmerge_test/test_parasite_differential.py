import torch
import torch.nn as nn
import unittest
from parasite.ModelWeightParasiticExtraction import ParasiticExtractor, ProbeNetwork
from .theoretical_references import compute_theoretical_infonce

class DummyHostModel(nn.Module):
    """A completely frozen, mock third-party architecture to attach hooks onto."""
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.fc_layer = nn.Linear(in_dim, out_dim)
        for p in self.parameters():
            p.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc_layer(x)


class TestParasiteDifferentialOracles(unittest.TestCase):
    def setUp(self):
        self.batch_size = 4
        self.host_dim = 64
        self.target_dim = 32
        
        # Instantiate production components
        self.production_extractor = ParasiticExtractor(
            host_dim=self.host_dim,
            target_dim=self.target_dim,
            use_ema=False
        )
        self.host_model = DummyHostModel(self.host_dim, self.host_dim)

    def test_differential_loss_equivalence(self):
        """
        Feeds synthetic input arrays through both the production extractor and the 
        isolated theoretical model to check that losses match up to an absolute tolerance.
        """
        synthetic_input = torch.randn(self.batch_size, self.host_dim)
        mock_shiva_backbone_output = torch.randn(self.batch_size, self.target_dim)
        
        # 1. Capture production pipeline forward-activation tracking
        with self.production_extractor.probe_context(self.host_model, "fc_layer"):
            # Intercept activations via forward hook integration
            host_output = self.host_model(synthetic_input)
            
        # Extract intercepted state vector from buffer
        captured_activation = self.production_extractor._buffer.read()
        
        # Drive production-side forward projection
        prod_projected_latent = self.production_extractor.probe(captured_activation)
        
        # 2. Run the decoupled functional oracle
        reference_calculated_loss = compute_theoretical_infonce(
            prod_projected_latent, 
            mock_shiva_backbone_output,
            temperature=0.07
        )
        
        # Calculate loss using the production code interface
        # Mocking backbone execution pathway matching core loops
        class MockEncoder:
            def forward_pass(self, x): return mock_shiva_backbone_output.unsqueeze(1)
            
        prod_loss_value = self.production_extractor.compute_loss(synthetic_input, MockEncoder())
        
        # 3. Assert exact execution invariants match precisely
        self.assertTrue(
            torch.allclose(prod_loss_value, reference_calculated_loss, atol=1e-5),
            f"Fidelity mismatch found: Production Loss = {prod_loss_value.item()} vs Reference = {reference_calculated_loss.item()}"
        )

if __name__ == "__main__":
    unittest.main()
