"""
hippocampus/episodic_recall.py — Inference-time memory retrieval.

The hippocampus doesn't only feed the trainer's dream cycles — it should
also serve the cerebrum's *current* reasoning by retrieving relevant past
episodes given a query state.

Biological role
~~~~~~~~~~~~~~~
When a familiar stimulus appears, the hippocampus retrieves associated
memories that bias perception, decision, and emotion. "I've seen something
like this before" is the recall mechanism humans use constantly. This
module is the computational equivalent: cosine-similarity retrieval
over the episodic memory bank's stored endpoints.

Design
~~~~~~
Two strategies, selected via `mode`:

    "endpoint":  Retrieve by cosine similarity to each episode's *final*
                 latent (the resolution state of the episode). Fast and
                 captures outcomes well.

    "trajectory": Retrieve by cosine similarity to the *mean* latent of
                  each episode (the average state during it). Captures
                  the overall character of the episode rather than the
                  conclusion.

The retrieved episodes are injected into working memory as additional
slots tagged "hippocampus_recall", with salience proportional to
similarity × stored_significance.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F


class EpisodicRecall:
    """
    Retrieves the top-K most relevant past episodes for a query latent.

    Args:
        memory:          The EpisodicMemory bank to query.
        mode:            "endpoint" or "trajectory" (see module docstring).
        top_k:           Number of episodes to retrieve per call.
        min_similarity:  Cosine similarity threshold below which results
                         are discarded. Prevents irrelevant recall.
    """

    def __init__(
        self,
        memory,
        mode: str = "endpoint",
        top_k: int = 3,
        min_similarity: float = 0.3,
    ) -> None:
        if mode not in ("endpoint", "trajectory"):
            raise ValueError(f"Unknown recall mode: {mode}")
        self.memory = memory
        self.mode = mode
        self.top_k = top_k
        self.min_similarity = min_similarity

    # ------------------------------------------------------------------
    # Internal: cache episode signatures
    # ------------------------------------------------------------------

    def _episode_signature(self, episode: dict) -> torch.Tensor:
        """One vector per episode for similarity matching. Always returns CPU tensor."""
        states = episode["states"]  # (T, D)
        if self.mode == "endpoint":
            sig = states[-1]
        else:  # trajectory
            sig = states.mean(dim=0)
        # Ensure CPU regardless of where the snapshot was saved from
        return sig.to('cpu') if sig.device.type != 'cpu' else sig

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: torch.Tensor,
    ) -> List[Tuple[dict, float]]:
        """
        Retrieve top-K relevant episodes for the query.

        Args:
            query: (D,) or (1, D) latent query vector.

        Returns:
            List of (episode_dict, similarity_score) tuples sorted by
            similarity descending. Empty list if memory is empty or no
            episode passes the similarity threshold.
        """
        if not self.memory._bank:
            return []

        # Move query to CPU for recall computation — episodes are stored as CPU tensors.
        if query.device.type != 'cpu':
            q = query.detach().to(torch.device('cpu'))
        else:
            q = query.detach()
        if q.dim() > 1:
            q = q.squeeze(0)
        q = F.normalize(q.float(), p=2, dim=0)

        # Stack signatures for vectorised cosine similarity.
        # Use numpy on DirectML (torch matmul on CPU tensors can be intercepted
        # by the DirectML dispatcher). On CUDA/MPS/CPU use torch directly.
        sigs_list = [self._episode_signature(ep) for ep in self.memory._bank]
        sigs = torch.stack([F.normalize(s.float(), p=2, dim=0) for s in sigs_list])

        if query.device.type == "privateuseone":
            import numpy as np
            sims = (sigs.numpy() @ q.numpy()).tolist()
        else:
            sims = (sigs @ q).tolist()

        # Pair with episodes, filter, sort
        scored = [
            (ep, sim) for ep, sim in zip(self.memory._bank, sims)
            if sim >= self.min_similarity
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[: self.top_k]

    def retrieve_latents(
        self,
        query: torch.Tensor,
    ) -> List[Tuple[torch.Tensor, float, float]]:
        """
        Convenience: return retrieved episode endpoint latents directly,
        along with similarity and significance.

        Returns:
            List of (latent_vector, similarity, significance) tuples.
        """
        results = self.retrieve(query)
        return [
            (
                self._episode_signature(ep),
                sim,
                float(ep.get("significance", 0.0)),
            )
            for ep, sim in results
        ]

    def inject_into_working_memory(
        self,
        query: torch.Tensor,
        working_memory,
        salience_scale: float = 0.7,
    ) -> int:
        """
        Retrieve top-K episodes and write them into working memory.

        Each recalled episode becomes a working memory slot with salience
        = salience_scale × similarity × episode_significance.

        Args:
            query:           (D,) query latent.
            working_memory:  WorkingMemory instance.
            salience_scale:  Master scale on injected salience.

        Returns:
            Number of episodes successfully injected.
        """
        retrieved = self.retrieve_latents(query)
        injected = 0
        for latent, sim, sig in retrieved:
            salience = salience_scale * sim * (0.5 + 0.5 * min(sig, 1.0))
            working_memory.write(
                latent.to(query.device) if isinstance(query, torch.Tensor) else latent,
                salience=float(salience),
                source_tag="hippocampus_recall",
            )
            injected += 1
        return injected


__all__ = ["EpisodicRecall"]
