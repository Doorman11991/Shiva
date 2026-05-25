"""
cerebrum/concept_grounding.py — Symbol to embedding binding.

The cerebrum bridges the gap between continuous latent representations
and discrete symbolic concepts. This enables compositional reasoning:
combining known concepts to understand novel situations, and explaining
latent states in human-interpretable terms.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConceptGrounder(nn.Module):
    """
    Binds abstract symbols to distributed latent embeddings.

    Two directions:
        ground(z)    → List[str]    "What concepts does this latent activate?"
        compose(concepts) → Tensor  "What latent represents these concepts?"

    The grounding is learned: the to_symbol projection is trained to
    predict which concepts are active given a latent vector, and the
    to_latent embedding table maps concepts back to latent space.

    Args:
        d_model:    Latent dimensionality.
        vocab:      Dict mapping concept name → index.
        top_k:      Number of top concepts to return from ground().
    """

    def __init__(
        self,
        d_model: int = 512,
        vocab: Optional[Dict[str, int]] = None,
        top_k: int = 5,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.top_k = top_k

        # Default concept vocabulary
        if vocab is None:
            vocab = {
                "novel": 0, "familiar": 1, "risky": 2, "safe": 3,
                "rewarding": 4, "costly": 5, "urgent": 6, "routine": 7,
                "curious": 8, "satisfied": 9, "confused": 10, "confident": 11,
                "social": 12, "solitary": 13, "creative": 14, "analytical": 15,
            }
        self.vocab = vocab
        self.vocab_size = len(vocab)
        self._idx_to_concept = {v: k for k, v in vocab.items()}

        # Latent → concept probabilities
        self.to_symbol = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, self.vocab_size),
        )

        # Concept index → latent embedding
        self.to_latent = nn.Embedding(self.vocab_size, d_model)

    def ground(self, z: torch.Tensor) -> List[str]:
        """
        Identify which concepts are active in a latent vector.

        Args:
            z: (D,) or (B, D) latent vector.

        Returns:
            List of top-k concept names (for single vector input).
        """
        if z.dim() == 1:
            z = z.unsqueeze(0)
        logits = self.to_symbol(z)
        probs = torch.softmax(logits, dim=-1)
        top_indices = probs[0].topk(self.top_k).indices.tolist()
        return [self._idx_to_concept[i] for i in top_indices]

    def ground_probs(self, z: torch.Tensor) -> Dict[str, float]:
        """Return concept activation probabilities as a dict."""
        if z.dim() == 1:
            z = z.unsqueeze(0)
        logits = self.to_symbol(z)
        probs = torch.softmax(logits, dim=-1)[0]
        return {self._idx_to_concept[i]: float(probs[i].item()) for i in range(self.vocab_size)}

    def compose(self, concepts: List[str]) -> torch.Tensor:
        """
        Compose a latent vector from a list of concept names.

        Args:
            concepts: List of concept names from the vocabulary.

        Returns:
            (D,) composed latent vector (mean of concept embeddings).
        """
        indices = []
        for c in concepts:
            if c in self.vocab:
                indices.append(self.vocab[c])
        if not indices:
            return torch.zeros(self.d_model)
        idx_tensor = torch.tensor(indices, dtype=torch.long)
        embeddings = self.to_latent(idx_tensor)
        return embeddings.mean(dim=0)

    def concept_similarity(self, concept_a: str, concept_b: str) -> float:
        """Cosine similarity between two concept embeddings."""
        if concept_a not in self.vocab or concept_b not in self.vocab:
            return 0.0
        ea = self.to_latent(torch.tensor([self.vocab[concept_a]]))
        eb = self.to_latent(torch.tensor([self.vocab[concept_b]]))
        return float(F.cosine_similarity(ea, eb).item())

    def add_concept(self, name: str) -> int:
        """Dynamically add a new concept to the vocabulary."""
        if name in self.vocab:
            return self.vocab[name]
        new_idx = self.vocab_size
        self.vocab[name] = new_idx
        self._idx_to_concept[new_idx] = name
        # Extend embedding table
        new_embed = nn.Embedding(new_idx + 1, self.d_model)
        with torch.no_grad():
            new_embed.weight[:new_idx] = self.to_latent.weight
        self.to_latent = new_embed
        self.vocab_size = new_idx + 1
        return new_idx


__all__ = ["ConceptGrounder"]
