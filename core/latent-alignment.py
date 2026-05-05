import torch
import torch.nn.functional as F
from torch import optim
from core.transformer_architecture import TransformerEncoderBlock

class LatentAligner:
    def __init__(self, d_model=512, num_heads=8, lr=1e-4, temperature=0.07):
        # Using your custom architecture for the shared backbone
        self.backbone = TransformerEncoderBlock(d_model, num_heads)
        self.temperature = temperature
        
        # Using Adam as requested
        self.optimizer = optim.Adam(self.backbone.parameters(), lr=lr, weight_decay=1e-2)

    def compute_infonce_loss(self, z_a, z_b):
        """
        z_a: Latent vectors from Domain A (e.g., Robotics) [batch_size, d_model]
        z_b: Latent vectors from Domain B (e.g., Language) [batch_size, d_model]
        """
        batch_size = z_a.shape[0]
        
        # 1. Normalize the vectors to the unit hypersphere
        z_a = F.normalize(z_a, p=2, dim=1)
        z_b = F.normalize(z_b, p=2, dim=1)
        
        # 2. Compute the cosine similarity matrix (logits)
        # Resulting shape: [batch_size, batch_size]
        logits = (z_a @ z_b.T) / self.temperature
        
        # 3. Target labels are the diagonal (each element matches its counterpart)
        labels = torch.arange(batch_size, device=z_a.device)
        
        # 4. Symmetric Cross-Entropy Loss
        loss_a_to_b = F.cross_entropy(logits, labels)
        loss_b_to_a = F.cross_entropy(logits.T, labels)
        
        return (loss_a_to_b + loss_b_to_a) / 2

    def train_step(self, data_a, data_b):
        self.optimizer.zero_grad()
        
        # Forward pass through your gated transformer
        z_a = self.backbone.forward_pass(data_a)
        z_b = self.backbone.forward_pass(data_b)
        
        # We take the mean across the sequence dimension for the global representation
        z_a_global = z_a.mean(dim=1)
        z_b_global = z_b.mean(dim=1)
        
        loss = self.compute_infonce_loss(z_a_global, z_b_global)
        loss.backward()
        self.optimizer.step()
        
        return loss.item()