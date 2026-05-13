import torch
import torch.nn as nn
from collections import deque
import random

class EpisodicMemory(nn.Module):
    def __init__(self,latent_dim=512, capacity=10000,sequence_length=16):
        super().__init__()
        self.capacity=capacity
        self.sequence_length=sequence_length
        self.memory_bank=deque(maxlen=capacity)
        self.narrative_encoder=nn.GRU(latent_dim,latent_dim,batch_first=True)
        self.self_token=nn.Parameter(torch.randn(1,1,latent_dim))#Distinguishing enabling for the model so that it can understand itself from the world
    def store_episode(self,state_sequence,valence_sequence,empowerment_score):
        significance=torch.abs(valence_sequence.mean())+empowerment_score
        self.memory_bank.append({
            'states':state_sequence.detach(),
            'significance':significance.item()
            })
    def get_dream_batch(self,batch_size):
        if len(self.memory_bank) < batch_size:
            return None
        weights = [m['significance'] for m in self.memory_bank]
        samples = random.choices(self.memory_bank, weights=weights, k=batch_size)
        return torch.stack([s['states'] for s in samples])
    def get_identity_context(self, current_latent):
        _, h_n = self.narrative_encoder(current_latent.unsqueeze(1))
        identity_context = h_n[-1]
        return identity_context + self.self_token.squeeze(0)
