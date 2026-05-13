import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from core.transformer_architecture import TransformerEncoderBlock

class LatentAligner(nn.Module):
    def __init__(self, input_dims=None, d_model=512, num_heads=8, lr=1e-4, temperature=0.07):
        super().__init__()
        if input_dims is None:
            input_dims = {}
        self.aligners = nn.ModuleDict({
            modality: nn.Linear(dim, d_model) for modality, dim in input_dims.items()
        })
        # Information Bottleneck constraint
        self.bottleneck = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, d_model)
        )
        self.backbone = TransformerEncoderBlock(d_model, num_heads)
        self.temperature = temperature
        self.emotion_vocab = {'Angry': 0, 'Sad': 1, 'Happy': 2, 'Calm': 3}
        self.num_emotions = len(self.emotion_vocab)
        self.emotion_embeddings = nn.Embedding(self.num_emotions, d_model)
        self.optimizer = optim.AdamW(self.parameters(), lr=lr, weight_decay=1e-2)

    def forward(self, x, modality):
        z_raw = self.aligners[modality](x)
        # Apply the Information Bottleneck to strip non-essential info
        z_aligned = self.bottleneck(z_raw)
        return z_aligned

    def recalibrate(self):
        for layer in self.aligners.values():
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight.data)
        
    def compute_infonce_loss(self, z_a, z_b):
        batch_size = z_a.shape[0]
        z_a = F.normalize(z_a, p=2, dim=1)
        z_b = F.normalize(z_b, p=2, dim=1)
        logits = (z_a @ z_b.T) / self.temperature
        labels = torch.arange(batch_size, device=z_a.device)
        loss_a_to_b = F.cross_entropy(logits, labels)
        loss_b_to_a = F.cross_entropy(logits.T, labels)
        
        return (loss_a_to_b + loss_b_to_a) / 2

    def compute_emotional_alignment_loss(self, z_a, z_b, z_emotion):
        loss_a_b = self.compute_infonce_loss(z_a, z_b)
        loss_a_emotion = self.compute_infonce_loss(z_a, z_emotion)
        loss_b_emotion = self.compute_infonce_loss(z_b, z_emotion)
        
        # This loss encourages all three vectors (data_a, data_b, emotion) to cluster together.
        return (loss_a_b + loss_a_emotion + loss_b_emotion) / 3

    def train_step(self, data_a, data_b, emotion_ids=None):
        self.optimizer.zero_grad()
        z_a = self.backbone.forward_pass(data_a)
        z_b = self.backbone.forward_pass(data_b)
        z_a_global = z_a.mean(dim=1)
        z_b_global = z_b.mean(dim=1)
        
        if emotion_ids is not None:
            z_emotion = self.emotion_embeddings(emotion_ids)
            loss = self.compute_emotional_alignment_loss(z_a_global, z_b_global, z_emotion)
        else:
            loss = self.compute_infonce_loss(z_a_global, z_b_global)

        loss.backward()
        self.optimizer.step()
        
        return loss.item()
