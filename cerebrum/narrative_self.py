"""
cerebrum/narrative_self.py — Autobiographical identity and continuity.

The cerebrum maintains a coherent sense of self across time — not just
"who am I now" but "who have I been" and "who am I becoming." This module
builds and maintains an autobiographical narrative from episodic memories,
preventing identity drift across context switches.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class BeliefVector:
    """A single core belief about the self."""

    def __init__(self, name: str, embedding: torch.Tensor, confidence: float = 0.5) -> None:
        self.name = name
        self.embedding = embedding.detach().to('cpu')
        self.confidence = confidence
        self.update_count = 0

    def update(self, new_evidence: torch.Tensor, learning_rate: float = 0.1) -> None:
        """
        Spherical linear interpolation (SLERP) update of belief embedding.

        Unlike EMA (which can collapse colinear vectors to a shorter version
        of the same direction), SLERP actually rotates the belief on the
        unit sphere toward the evidence by `learning_rate` fraction of the
        angle between them.
        """
        import torch.nn.functional as F

        a = F.normalize(self.embedding.float(), dim=0)
        b = F.normalize(new_evidence.detach().to('cpu').float(), dim=0)

        # Cosine of angle between a and b
        dot = torch.clamp(torch.dot(a, b), -1.0, 1.0)
        omega = torch.acos(dot)  # angle

        # If angle is near-zero (already aligned), just keep current
        if omega.abs() < 1e-6:
            return

        sin_omega = torch.sin(omega)
        # SLERP: interpolate learning_rate of the way from a toward b
        t = learning_rate
        self.embedding = (
            (torch.sin((1 - t) * omega) / sin_omega) * a
            + (torch.sin(t * omega) / sin_omega) * b
        )
        self.update_count += 1


class NarrativeSelf(nn.Module):
    """
    Maintains autobiographical coherence across time.

    The narrative self is built from:
        1. Core beliefs: slowly-updating self-model vectors
        2. Recent narrative: GRU encoding of recent episode endpoints
        3. Identity token: a persistent learned self-representation

    The self-model vector is used to:
        - Ground the conscious latent (added to z_global in the policy)
        - Evaluate goal coherence ("is this goal aligned with who I am?")
        - Detect identity drift (large change in self-model = context switch)

    Args:
        latent_dim:      Latent dimensionality.
        n_core_beliefs:  Number of core belief slots.
        narrative_len:   Length of recent narrative window.
    """

    def __init__(
        self,
        latent_dim: int = 512,
        n_core_beliefs: int = 8,
        narrative_len: int = 16,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.n_core_beliefs = n_core_beliefs
        self.narrative_len = narrative_len

        # Persistent identity token (the "I")
        self.identity_token = nn.Parameter(torch.randn(1, latent_dim))

        # Belief encoder: maps episode to belief update
        self.belief_encoder = nn.Sequential(
            nn.Linear(latent_dim, latent_dim // 2),
            nn.GELU(),
            nn.Linear(latent_dim // 2, latent_dim),
        )

        # Narrative GRU: encodes recent trajectory
        self.narrative_gru = nn.GRU(latent_dim, latent_dim, batch_first=True)

        # Self-model projection: combines beliefs + narrative → self vector
        self.self_proj = nn.Linear(latent_dim * 2, latent_dim)

        self._core_beliefs: List[BeliefVector] = []
        self._narrative_buffer: List[torch.Tensor] = []

    def update_narrative(
        self,
        episode_latent: torch.Tensor,
        outcome_valence: float,
    ) -> float:
        """
        Update the narrative with a new episode.

        Args:
            episode_latent: (D,) mean latent of the episode.
            outcome_valence: Emotional valence of the outcome.

        Returns:
            Narrative surprise (how much this episode changed the self-model).
        """
        z = episode_latent.detach().to('cpu')
        self._narrative_buffer.append(z)
        if len(self._narrative_buffer) > self.narrative_len:
            self._narrative_buffer.pop(0)

        # Update core beliefs if outcome was significant
        if abs(outcome_valence) > 0.3:
            dev = next(self.belief_encoder.parameters()).device
            belief_embedding = self.belief_encoder(z.to(dev).unsqueeze(0)).squeeze(0)
            if len(self._core_beliefs) < self.n_core_beliefs:
                self._core_beliefs.append(
                    BeliefVector(f"belief_{len(self._core_beliefs)}", belief_embedding)
                )
            else:
                # Update the least-confident belief
                least_confident = min(self._core_beliefs, key=lambda b: b.confidence)
                old_emb = least_confident.embedding.clone()
                least_confident.update(belief_embedding)
                surprise = float(F.mse_loss(least_confident.embedding, old_emb).item())
                return surprise

        return 0.0

    def get_self_model(self, device: str = "cpu") -> torch.Tensor:
        """
        Compute the current self-model vector.

        Returns:
            (D,) self-model vector combining identity token, beliefs, and narrative.
        """
        dev = torch.device(device)
        identity = self.identity_token.to(dev).squeeze(0)  # (D,)

        if not self._narrative_buffer:
            return identity

        # Encode recent narrative.
        # nn.GRU is not supported on DirectML — run on CPU, move result back.
        narrative_cpu = torch.stack(self._narrative_buffer).to('cpu').unsqueeze(0)  # (1, T, D)
        gru_cpu = self.narrative_gru.cpu()
        with torch.no_grad():
            _, h_n = gru_cpu(narrative_cpu)
        narrative_vec = h_n[-1].squeeze(0).to(dev)  # (D,)

        # Combine with identity
        combined = torch.cat([identity, narrative_vec], dim=-1).unsqueeze(0)  # (1, 2D)
        self_model = self.self_proj(combined).squeeze(0)  # (D,)
        return self_model + identity  # residual

    def goal_coherence(
        self,
        goal_latent: torch.Tensor,
        device: str = "cpu",
    ) -> float:
        """
        Measure how coherent a goal is with the current self-model.

        Returns cosine similarity in [-1, 1]. High = coherent with identity.
        """
        self_model = self.get_self_model(device)
        goal = goal_latent.to(self_model.device).flatten()
        return float(F.cosine_similarity(
            self_model.unsqueeze(0), goal.unsqueeze(0)
        ).item())

    # ------------------------------------------------------------------
    # Periodic semantic mood grounding via the granite embedder
    # ------------------------------------------------------------------

    def ground_mood_in_language(
        self,
        homeostasis_vector: torch.Tensor,
        mood_name: str,
        valence: float,
        step: int,
        grounding_interval: int = 100,
    ) -> Optional[torch.Tensor]:
        """
        Periodically surface the agent's internal state into natural language,
        encode it with the granite embedder, and anchor the self-model to it.

        This is the "subconscious affect" mechanism: every N steps the brain
        translates its drives and mood into a sentence, embeds that sentence,
        and uses it to gently bias the identity token. The effect bleeds into
        every subsequent action through the narrative self-model.

        Why periodic and not every tick:
            Granite is a 125M-parameter transformer — ~50-200ms per call on CPU.
            Running it every tick would bottleneck the brain. Instead we run it
            every `grounding_interval` steps (default 100), which gives a
            smooth, slowly-drifting subconscious influence without stalling
            the main loop.

        Args:
            homeostasis_vector: (6,) or (4,) current drive state.
            mood_name:          Current mood string (e.g. "Calm", "Angry").
            valence:            Current scalar valence in [-1, 1].
            step:               Current training step (used for interval check).
            grounding_interval: How many steps between grounding calls.

        Returns:
            The mood anchor tensor (D,) if grounding fired this step, else None.
        """
        if step % grounding_interval != 0:
            return None

        # Build a natural language description of the internal state.
        mood_text = self._describe_internal_state(
            homeostasis_vector, mood_name, valence
        )

        # Encode with the granite embedder (singleton — loads once per process).
        from thalamus.granite_embedder import get_embedder
        embedder = get_embedder()
        mood_anchor = embedder.encode(mood_text)  # (D,)

        # Anchor the identity token toward this mood embedding.
        # Small step size (0.05) so the identity drifts slowly, not jumps.
        with torch.no_grad():
            anchor = mood_anchor.to(self.identity_token.device)
            self.identity_token.data = (
                0.95 * self.identity_token.data
                + 0.05 * anchor.unsqueeze(0)
            )

        # Also store as a core belief so it persists across episodes.
        belief_name = f"mood_anchor_step{step}"
        self._core_beliefs.append(BeliefVector(belief_name, mood_anchor, confidence=0.6))
        if len(self._core_beliefs) > self.n_core_beliefs:
            # Evict the least confident belief
            self._core_beliefs.sort(key=lambda b: b.confidence, reverse=True)
            self._core_beliefs = self._core_beliefs[:self.n_core_beliefs]

        return mood_anchor

    @staticmethod
    def _describe_internal_state(
        homeostasis_vector: torch.Tensor,
        mood_name: str,
        valence: float,
    ) -> str:
        """
        Convert the internal state into a natural language sentence.

        This is the bridge between continuous latent drives and the
        symbolic language the granite embedder understands.
        """
        h = homeostasis_vector.detach().to('cpu').tolist()

        # Map drive dimensions to descriptive phrases.
        # Handles both 4-dim (legacy) and 6-dim (HomeostaticRegulator) vectors.
        dim_names_4 = ["arousal", "energy", "safety", "engagement"]
        dim_names_6 = ["arousal", "energy", "safety", "engagement", "curiosity", "coherence"]
        dim_names = dim_names_6 if len(h) >= 6 else dim_names_4

        # Find the most prominent drive (furthest from 0.5 neutral)
        deviations = [(abs(v - 0.5), name, v) for name, v in zip(dim_names, h)]
        deviations.sort(reverse=True)
        top_drive, top_val = deviations[0][1], deviations[0][2]

        # Valence descriptor
        if valence > 0.3:
            valence_desc = "positive and engaged"
        elif valence < -0.3:
            valence_desc = "uneasy and cautious"
        else:
            valence_desc = "neutral and observant"

        # Drive descriptor
        drive_desc_map = {
            "arousal":    ("highly stimulated", "calm and understimulated"),
            "energy":     ("energised and ready", "fatigued and depleted"),
            "safety":     ("secure and confident", "anxious and threatened"),
            "engagement": ("deeply engaged", "disengaged and bored"),
            "curiosity":  ("intensely curious", "incurious and settled"),
            "coherence":  ("clear and coherent", "confused and uncertain"),
        }
        high_desc, low_desc = drive_desc_map.get(top_drive, ("active", "inactive"))
        drive_desc = high_desc if top_val > 0.5 else low_desc

        return (
            f"I feel {mood_name.lower()} and {valence_desc}. "
            f"I am {drive_desc}."
        )

    def status(self) -> Dict:
        return {
            "n_core_beliefs": len(self._core_beliefs),
            "narrative_length": len(self._narrative_buffer),
            "belief_names": [b.name for b in self._core_beliefs],
        }


__all__ = ["NarrativeSelf", "BeliefVector"]
