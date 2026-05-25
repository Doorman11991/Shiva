"""
cerebellum/emotional_contagion.py — Valence spreads across swarm nodes.

Biological role
~~~~~~~~~~~~~~~
Emotions are contagious. When one person panics, the group panics. When
one person laughs, others join. This isn't a bug — it's a coordination
mechanism. Groups that share emotional state align their behaviour faster
than groups that communicate only through explicit signals.

In Chip's swarm, emotional contagion means: when one node's valence
shifts strongly, neighbouring nodes' valences shift toward it, weighted
by their integration gate. This creates emergent group affect — the
swarm can collectively feel "cautious" or "curious" without any central
controller dictating it.

Design
~~~~~~
Contagion happens at the cerebellum level (not amygdala) because it's a
*coordination* mechanism, not a sensing mechanism. The amygdala computes
a single node's emotional response. The cerebellum coordinates multiple
nodes. Contagion is the emotional equivalent of the consensus vector.

Process per tick:
    1. Each node reports its current valence to the coordinator.
    2. The coordinator computes a "group affect" vector (weighted mean
       of all node valences, weighted by recency and integration gate).
    3. Each node's valence is pulled toward the group affect by a
       susceptibility factor.
    4. The group affect is broadcast as a signal on the bus.

Safeguards:
    - Susceptibility decays with valence magnitude: nodes with extreme
      valence (very scared, very excited) are LESS susceptible to group
      influence. This prevents runaway positive feedback.
    - A "contagion dampening" parameter caps the maximum per-tick shift.
    - Only fires when > 1 node is active.

Integration with existing SwarmCoordinator:
    The SwarmCoordinator handles latent-space consensus. EmotionalContagion
    handles valence-space consensus. They run in parallel during the
    cerebellum step — one aligns what the nodes THINK, the other aligns
    what they FEEL.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch


# ---------------------------------------------------------------------------
# NodeValenceState — tracks one node's emotional trajectory
# ---------------------------------------------------------------------------

class NodeValenceState:
    """Tracks a single node's recent valence history."""

    __slots__ = ("node_id", "valence", "arousal", "history", "_max_hist")

    def __init__(self, node_id: str, max_history: int = 20) -> None:
        self.node_id = node_id
        self.valence: float = 0.0
        self.arousal: float = 0.5
        self.history: List[float] = []
        self._max_hist = max_history

    def update(self, valence: float, arousal: float = 0.5) -> None:
        self.valence = valence
        self.arousal = arousal
        self.history.append(valence)
        if len(self.history) > self._max_hist:
            self.history.pop(0)

    @property
    def trend(self) -> float:
        """Valence trend: positive = improving, negative = declining."""
        if len(self.history) < 3:
            return 0.0
        recent = self.history[-3:]
        return (recent[-1] - recent[0]) / 2.0


# ---------------------------------------------------------------------------
# EmotionalContagion
# ---------------------------------------------------------------------------

class EmotionalContagion:
    """
    Spreads valence across swarm nodes.

    Args:
        susceptibility:     Base susceptibility to group affect [0, 1].
                            0 = immune to contagion. 1 = fully absorbent.
        max_shift_per_tick: Maximum valence shift any node can receive per tick.
                            Caps runaway feedback loops.
        extreme_dampening:  How much extreme valence reduces susceptibility.
                            Nodes at |valence| > 0.7 become progressively
                            less susceptible, preventing panic spirals.
        min_nodes:          Minimum active nodes for contagion to fire.
    """

    def __init__(
        self,
        susceptibility: float = 0.3,
        max_shift_per_tick: float = 0.1,
        extreme_dampening: float = 0.5,
        min_nodes: int = 2,
    ) -> None:
        self.susceptibility = susceptibility
        self.max_shift_per_tick = max_shift_per_tick
        self.extreme_dampening = extreme_dampening
        self.min_nodes = min_nodes

        self._nodes: Dict[str, NodeValenceState] = {}
        self._group_valence: float = 0.0
        self._group_arousal: float = 0.5
        self._tick: int = 0
        self._contagion_events: int = 0

    # ------------------------------------------------------------------
    # Node management
    # ------------------------------------------------------------------

    def register_node(self, node_id: str) -> None:
        if node_id not in self._nodes:
            self._nodes[node_id] = NodeValenceState(node_id)

    def update_node(self, node_id: str, valence: float, arousal: float = 0.5) -> None:
        """Report a node's current valence/arousal. Call each tick per node."""
        if node_id not in self._nodes:
            self.register_node(node_id)
        self._nodes[node_id].update(valence, arousal)

    # ------------------------------------------------------------------
    # Contagion step
    # ------------------------------------------------------------------

    def step(self) -> Optional[Dict[str, float]]:
        """
        Run one contagion tick. Computes group affect and pulls each
        node toward it.

        Returns:
            Dict of {node_id: valence_shift_applied} if contagion fired,
            None if insufficient nodes.
        """
        self._tick += 1
        if len(self._nodes) < self.min_nodes:
            return None

        # 1. Compute group affect (arousal-weighted mean valence)
        total_weight = 0.0
        weighted_valence = 0.0
        weighted_arousal = 0.0
        for node in self._nodes.values():
            # Arousal acts as "broadcast strength" — high arousal nodes
            # influence the group more (like someone shouting in a crowd).
            w = 0.3 + 0.7 * node.arousal
            weighted_valence += node.valence * w
            weighted_arousal += node.arousal * w
            total_weight += w

        if total_weight < 1e-6:
            return None

        self._group_valence = weighted_valence / total_weight
        self._group_arousal = weighted_arousal / total_weight

        # 2. Pull each node toward group affect
        shifts: Dict[str, float] = {}
        for node in self._nodes.values():
            # Susceptibility decreases with extreme valence
            extreme_factor = max(0.0, 1.0 - self.extreme_dampening * abs(node.valence))
            effective_susceptibility = self.susceptibility * extreme_factor

            # Pull toward group valence
            delta = self._group_valence - node.valence
            shift = delta * effective_susceptibility

            # Clamp shift
            shift = max(-self.max_shift_per_tick, min(self.max_shift_per_tick, shift))

            node.valence += shift
            node.valence = max(-1.0, min(1.0, node.valence))
            shifts[node.node_id] = shift

        if any(abs(s) > 1e-4 for s in shifts.values()):
            self._contagion_events += 1

        return shifts

    # ------------------------------------------------------------------
    # Read state
    # ------------------------------------------------------------------

    @property
    def group_valence(self) -> float:
        return self._group_valence

    @property
    def group_arousal(self) -> float:
        return self._group_arousal

    def get_node_valence(self, node_id: str) -> float:
        if node_id in self._nodes:
            return self._nodes[node_id].valence
        return 0.0

    def group_mood(self) -> str:
        """Simple group mood label from group valence/arousal."""
        v, a = self._group_valence, self._group_arousal
        if a > 0.6 and v > 0.1:
            return "excited"
        elif a > 0.6 and v < -0.1:
            return "panicked"
        elif a < 0.4 and v < -0.1:
            return "dejected"
        elif a < 0.4 and v > 0.1:
            return "content"
        return "neutral"

    def status(self) -> Dict:
        return {
            "group_valence": self._group_valence,
            "group_arousal": self._group_arousal,
            "group_mood": self.group_mood(),
            "n_nodes": len(self._nodes),
            "contagion_events": self._contagion_events,
            "node_valences": {nid: n.valence for nid, n in self._nodes.items()},
        }


__all__ = ["EmotionalContagion", "NodeValenceState"]
