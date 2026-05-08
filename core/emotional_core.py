import torch
import torch.nn as nn
from typing import Tuple
from core.latent_alignment import LatentAligner

class EmotionalCore(nn.Module):
    def __init__(self, latent_aligner: LatentAligner, initial_mood: str = 'Calm', hidden_dim: int = 512):
        super().__init__()
        self.latent_aligner = latent_aligner
        self.emotion_vocab = self.latent_aligner.emotion_vocab
        
        if initial_mood not in self.emotion_vocab:
            raise ValueError(f"Initial mood '{initial_mood}' not found in emotion vocabulary: {list(self.emotion_vocab.keys())}")

        self._current_mood_name: str = initial_mood
        self._reason_for_change: str = "Initialization"

        self.internal_state = nn.Parameter(torch.tensor([0.5, 0.8, 1.0, 0.5]), requires_grad=False)
        self.target_state = torch.tensor([0.8, 1.0, 0.7, 0.6])

        self.valence_network = nn.Sequential(
            nn.Linear(hidden_dim + 4, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Tanh() 
        )

    def current_mood(self) -> Tuple[str, torch.Tensor]:
        mood_id = self.emotion_vocab[self._current_mood_name]
        mood_tensor = torch.tensor([mood_id], dtype=torch.long, device=self.latent_aligner.emotion_embeddings.weight.device)
        mood_vector = self.latent_aligner.emotion_embeddings(mood_tensor).squeeze(0).detach()
        return self._current_mood_name, mood_vector

    def mood_swing(self, new_mood: str, reason: str):
        if new_mood not in self.emotion_vocab:
            print(f"I am feeling a mood swing !!!")
            return
        
        self._current_mood_name = new_mood
        self._reason_for_change = reason

    def reason_for_mood_change(self) -> str:
        return self._reason_for_change

    def set_mood_angry(self, reason: str):
        self.mood_swing('Angry', reason)

    def set_mood_sad(self, reason: str):
        self.mood_swing('Sad', reason)

    def set_mood_happy(self, reason: str):
        self.mood_swing('Happy', reason)

    def set_mood_calm(self, reason: str):
        self.mood_swing('Calm', reason)

    def update_homeostasis(self, action_impact, environment_surprise):
        self.internal_state[0] += 0.05 * environment_surprise
        self.internal_state[1] -= 0.02 * action_impact
        self.internal_state.clamp_(0, 1)

    def get_valence(self, latent_state):
        if latent_state.dim() > 1:
            internal_state_expanded = self.internal_state.unsqueeze(0).expand(latent_state.size(0), -1)
        else:
            internal_state_expanded = self.internal_state
            
        combined = torch.cat([latent_state, internal_state_expanded], dim=-1)
        return self.valence_network(combined)

    def calculate_strain(self):
        target = self.target_state.to(self.internal_state.device)
        return torch.norm(self.internal_state - target)
    #This is the new updated code for the Shiva code. Yet to add better functionalities to it.
