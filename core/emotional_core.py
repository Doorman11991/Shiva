import torch
from typing import Tuple
from core.latent_alignment import LatentAligner

class EmotionalCore:
    def __init__(self, latent_aligner: LatentAligner, initial_mood: str = 'Calm'):
        self.latent_aligner = latent_aligner
        self.emotion_vocab = self.latent_aligner.emotion_vocab
        
        if initial_mood not in self.emotion_vocab:
            raise ValueError(f"Initial mood '{initial_mood}' not found in emotion vocabulary: {list(self.emotion_vocab.keys())}")

        self._current_mood_name: str = initial_mood
        self._reason_for_change: str = "Initialization"

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