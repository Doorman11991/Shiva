"""
cerebrum/self_consistency.py — Contradiction detection and belief revision.

Biological role
~~~~~~~~~~~~~~~
When humans encounter information that contradicts a held belief, we
experience cognitive dissonance — an uncomfortable state that motivates
us to either revise the belief or reject the evidence. This module
implements the computational version: detecting when new latent evidence
contradicts stored core beliefs, scoring the severity of the contradiction,
and deciding whether to revise the belief (low stakes) or signal a
narrative crisis (high stakes, high strain).

Integration
~~~~~~~~~~~
Reads from:
    - cerebrum/narrative_self.py   : core beliefs
    - hippocampus/episodic_memory  : recent episode latents
    - amygdala/emotional_core      : homeostasis (for strain assessment)

Writes to:
    - cerebrum/narrative_self.py   : belief updates on revision
    - SignalBus                    : "contradiction_detected" signal
    - HookRegistry                : "belief_revised" event

The consistency check runs:
    - Every tick (cheap cosine check against belief centroid)
    - Deep scan every N ticks (compares new evidence against each
      individual belief embedding)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# ContradictionEvent — one detected inconsistency
# ---------------------------------------------------------------------------

@dataclass
class ContradictionEvent:
    """A detected contradiction between new evidence and a stored belief."""
    tick: int
    belief_name: str
    belief_embedding: torch.Tensor
    evidence_embedding: torch.Tensor
    severity: float             # 0.0 = trivial, 1.0 = catastrophic
    cosine_distance: float      # 1 - cosine_similarity (higher = more contradictory)
    resolved: bool = False
    resolution: str = ""        # "revised", "rejected_evidence", "crisis"


# ---------------------------------------------------------------------------
# ConsistencyChecker
# ---------------------------------------------------------------------------

class ConsistencyChecker:
    """
    Detects contradictions between new evidence and core beliefs.

    Contradiction is defined as a large cosine distance between the
    new latent and a belief embedding, combined with the belief being
    in the *opposite* hemisphere (cosine similarity < 0). Simple
    orthogonality (sim ≈ 0) is "unrelated", not contradictory.

    Severity is scored by:
        severity = |negative_cosine_sim| × belief_confidence × stakes

    where stakes = homeostatic strain (high strain = high stakes).

    Args:
        contradiction_threshold: Cosine similarity below this triggers.
                                 For synthetic latents: -0.2 (must be actively opposed).
                                 For granite-encoded text: 0.4 (must be dissimilar —
                                 since all text clusters in a narrow positive cone).
                                 Adjust based on your embedding model.
        deep_scan_every:         Run full belief scan every N ticks.
        revision_threshold:      Severity below which auto-revision happens.
                                 Above this → narrative crisis signal.
    """

    def __init__(
        self,
        contradiction_threshold: float = 0.4,
        deep_scan_every: int = 10,
        revision_threshold: float = 0.6,
    ) -> None:
        self.contradiction_threshold = contradiction_threshold
        self.deep_scan_every = deep_scan_every
        self.revision_threshold = revision_threshold

        self._events: List[ContradictionEvent] = []
        self._last_deep_scan: int = -10**9
        self._revision_count: int = 0
        self._crisis_count: int = 0

    # ------------------------------------------------------------------
    # Quick check: new evidence vs belief centroid
    # ------------------------------------------------------------------

    def quick_check(
        self,
        evidence: torch.Tensor,
        beliefs: List,
    ) -> Optional[ContradictionEvent]:
        """
        Fast check: is the new evidence contradictory to *any* belief?

        This runs every tick and does O(N_beliefs) cosine comparisons —
        cheap because N_beliefs is small (typically 4-8).

        Args:
            evidence: (D,) latent vector of the new observation.
            beliefs:  List of BeliefVector objects from NarrativeSelf.

        Returns:
            The worst ContradictionEvent found, or None if no contradiction.
        """
        if not beliefs:
            return None

        ev = evidence.detach().cpu()
        if ev.dim() > 1:
            ev = ev.squeeze(0)
        ev_norm = F.normalize(ev, dim=0)

        worst: Optional[ContradictionEvent] = None
        worst_severity = 0.0

        for belief in beliefs:
            b_norm = F.normalize(belief.embedding.float(), dim=0)
            sim = float(torch.dot(ev_norm, b_norm).item())

            if sim < self.contradiction_threshold:
                # This is a contradiction: similarity is below the threshold.
                # For granite text (where all embeddings live in a positive cone),
                # threshold ~0.4 means "semantically dissimilar enough to conflict."
                # For synthetic latents with threshold < 0, this catches true opposition.
                cos_dist = 1.0 - sim
                raw_severity = (1.0 - sim) * belief.confidence  # further = more severe
                if raw_severity > worst_severity:
                    worst_severity = raw_severity
                    worst = ContradictionEvent(
                        tick=0,  # set by caller
                        belief_name=belief.name,
                        belief_embedding=belief.embedding.clone(),
                        evidence_embedding=ev.clone(),
                        severity=raw_severity,
                        cosine_distance=cos_dist,
                    )

        return worst

    # ------------------------------------------------------------------
    # Deep scan: periodic full analysis
    # ------------------------------------------------------------------

    def deep_scan(
        self,
        tick: int,
        recent_latents: List[torch.Tensor],
        beliefs: List,
        homeostasis_strain: float = 0.0,
    ) -> List[ContradictionEvent]:
        """
        Full scan of recent evidence against all beliefs. Runs every
        `deep_scan_every` ticks.

        Args:
            tick:               Current tick.
            recent_latents:     List of recent observation latents.
            beliefs:            List of BeliefVector objects.
            homeostasis_strain: Current homeostatic strain (scales severity).

        Returns:
            List of all detected contradictions (may be empty).
        """
        if tick - self._last_deep_scan < self.deep_scan_every:
            return []
        self._last_deep_scan = tick

        events = []
        for latent in recent_latents:
            event = self.quick_check(latent, beliefs)
            if event is not None:
                # Scale severity by homeostatic strain (high strain = high stakes)
                event.severity *= (0.5 + 0.5 * min(homeostasis_strain, 1.0))
                event.tick = tick
                events.append(event)

        self._events.extend(events)
        return events

    # ------------------------------------------------------------------
    # Resolution: revise or escalate
    # ------------------------------------------------------------------

    def resolve(
        self,
        event: ContradictionEvent,
        narrative_self,
    ) -> str:
        """
        Resolve a detected contradiction.

        Low severity (< revision_threshold):
            Revise the belief toward the new evidence (small EMA update).
            This is normal learning — beliefs should update with data.

        High severity (≥ revision_threshold):
            Signal a "narrative crisis" — the contradiction is too large
            for a quiet update. The brain should deliberate, possibly
            triggering inner speech and a goal to investigate.

        Args:
            event:          The contradiction to resolve.
            narrative_self: The NarrativeSelf module (for belief updates).

        Returns:
            Resolution label: "revised", "crisis", or "rejected_evidence".
        """
        if event.severity < self.revision_threshold:
            # Low-stakes: quietly revise the belief
            for belief in narrative_self._core_beliefs:
                if belief.name == event.belief_name:
                    belief.update(event.evidence_embedding, learning_rate=0.15)
                    break
            event.resolved = True
            event.resolution = "revised"
            self._revision_count += 1
            return "revised"
        else:
            # High-stakes: narrative crisis
            event.resolved = True
            event.resolution = "crisis"
            self._crisis_count += 1
            return "crisis"

    # ------------------------------------------------------------------
    # History / diagnostics
    # ------------------------------------------------------------------

    def recent_events(self, n: int = 5) -> List[ContradictionEvent]:
        return list(self._events[-n:])

    def status(self) -> Dict:
        return {
            "total_contradictions": len(self._events),
            "revisions": self._revision_count,
            "crises": self._crisis_count,
            "recent": [
                {
                    "tick": e.tick,
                    "belief": e.belief_name,
                    "severity": round(e.severity, 3),
                    "resolution": e.resolution,
                }
                for e in self._events[-3:]
            ],
        }


__all__ = ["ConsistencyChecker", "ContradictionEvent"]
