"""
interfaces/signals.py — Neural signal bus.

The SignalBus is the central nervous system of the Chip brain.
Regions publish typed NeuralSignal objects and subscribe to signal types
they care about. This decouples regions completely — no region ever
imports another region's module.

Signal type conventions (mirrors real neural pathways):
    "sensory_tokens"        Thalamus → Cerebrum      filtered latent tokens
    "attention_query"       Cerebrum → Thalamus      top-down attention bias
    "fear_veto"             Amygdala → Cerebellum    emergency action block
    "valence_update"        Amygdala → Cerebrum      current emotional valence
    "arousal_gain"          Amygdala → Thalamus      attention sensitivity scale
    "drive_signal"          Hypothalamus → Cerebrum  curiosity / fatigue / urgency
    "homeostasis_update"    Hypothalamus → Amygdala  internal state vector
    "memory_store"          Any → Hippocampus        experience to encode
    "memory_retrieve"       Hippocampus → Cerebrum   recalled episode batch
    "health_stats"          Brainstem → Hypothalamus loss / gradient / NaN counts
    "action_raw"            Cerebrum → Cerebellum    unsmoothed action tensor
    "action_smooth"         Cerebellum → output      refined action tensor
    "consensus_vector"      Cerebellum → Cerebrum    swarm consensus
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import torch


# ---------------------------------------------------------------------------
# NeuralSignal
# ---------------------------------------------------------------------------

@dataclass
class NeuralSignal:
    """
    A typed message passed between brain regions.

    Fields:
        source:      Region name that emitted this signal (e.g. "amygdala").
        target:      Intended recipient region, or "*" for broadcast.
        signal_type: Semantic label (see module docstring for conventions).
        payload:     The data being transmitted. May be a Tensor, float,
                     dict, or any picklable Python object.
        priority:    0.0 (low) to 1.0 (high). High-priority signals (e.g.
                     fear_veto) are delivered before low-priority ones.
        timestamp:   Unix time of emission. Used for latency diagnostics.
        step:        Training step at emission time (set by SignalBus).
    """
    source: str
    target: str
    signal_type: str
    payload: Any
    priority: float = 0.5
    timestamp: float = field(default_factory=time.time)
    step: int = 0

    def is_broadcast(self) -> bool:
        return self.target == "*"

    def __repr__(self) -> str:
        payload_repr = (
            f"Tensor{tuple(self.payload.shape)}"
            if isinstance(self.payload, torch.Tensor)
            else repr(self.payload)
        )
        return (
            f"NeuralSignal({self.source!r} → {self.target!r} "
            f"[{self.signal_type}] p={self.priority:.2f} {payload_repr})"
        )


# ---------------------------------------------------------------------------
# SignalBus
# ---------------------------------------------------------------------------

class SignalBus:
    """
    Central nervous system bus — routes NeuralSignals between brain regions.

    Usage:
        bus = SignalBus()

        # Regions subscribe to signal types they want to receive:
        bus.subscribe("cerebrum", ["sensory_tokens", "drive_signal", "memory_retrieve"])
        bus.subscribe("cerebellum", ["action_raw", "fear_veto"])

        # Regions publish signals:
        bus.publish(NeuralSignal(
            source="thalamus",
            target="cerebrum",
            signal_type="sensory_tokens",
            payload=z_filtered,
            priority=0.8,
        ))

        # Regions poll their inbox each tick:
        incoming = bus.poll("cerebrum")   # List[NeuralSignal], sorted by priority desc

        # Advance the global step counter (called by the trainer each update):
        bus.tick()
    """

    def __init__(self) -> None:
        # inbox[region] = list of pending signals
        self._inbox: Dict[str, List[NeuralSignal]] = defaultdict(list)
        # subscriptions[region] = set of signal_types it wants
        self._subscriptions: Dict[str, set] = defaultdict(set)
        # optional hooks: signal_type → list of callbacks (for debugging)
        self._hooks: Dict[str, List[Callable[[NeuralSignal], None]]] = defaultdict(list)
        self._step: int = 0
        self._history: List[NeuralSignal] = []
        self._max_history: int = 1000

    # ------------------------------------------------------------------
    # Subscription management
    # ------------------------------------------------------------------

    def subscribe(self, region: str, signal_types: List[str]) -> None:
        """Register a region to receive the given signal types."""
        self._subscriptions[region].update(signal_types)

    def unsubscribe(self, region: str, signal_types: Optional[List[str]] = None) -> None:
        """Remove subscriptions. If signal_types is None, remove all."""
        if signal_types is None:
            self._subscriptions[region].clear()
        else:
            self._subscriptions[region].difference_update(signal_types)

    # ------------------------------------------------------------------
    # Publishing
    # ------------------------------------------------------------------

    def publish(self, signal: NeuralSignal) -> None:
        """
        Route a signal to its target region's inbox.

        If target is "*", the signal is delivered to every subscribed region
        that has registered interest in this signal_type.
        """
        signal.step = self._step

        # Fire debug hooks first.
        for cb in self._hooks.get(signal.signal_type, []):
            try:
                cb(signal)
            except Exception:
                pass

        # Store in history (ring buffer).
        self._history.append(signal)
        if len(self._history) > self._max_history:
            self._history.pop(0)

        if signal.is_broadcast():
            # Deliver to every region subscribed to this signal type.
            for region, types in self._subscriptions.items():
                if signal.signal_type in types:
                    self._inbox[region].append(signal)
        else:
            # Direct delivery — no subscription check needed.
            self._inbox[signal.target].append(signal)

    def publish_many(self, signals: List[NeuralSignal]) -> None:
        """Convenience: publish a list of signals in one call."""
        for s in signals:
            self.publish(s)

    # ------------------------------------------------------------------
    # Polling
    # ------------------------------------------------------------------

    def poll(self, region: str) -> List[NeuralSignal]:
        """
        Drain and return all pending signals for a region, sorted by
        priority descending (highest priority delivered first).

        Clears the inbox after reading — each signal is consumed once.
        """
        inbox = self._inbox.pop(region, [])
        return sorted(inbox, key=lambda s: s.priority, reverse=True)

    def peek(self, region: str) -> List[NeuralSignal]:
        """Read without consuming. Useful for diagnostics."""
        return list(self._inbox.get(region, []))

    # ------------------------------------------------------------------
    # Step management
    # ------------------------------------------------------------------

    def tick(self) -> None:
        """Advance the global step counter. Call once per training update."""
        self._step += 1

    @property
    def step(self) -> int:
        return self._step

    # ------------------------------------------------------------------
    # Debug hooks
    # ------------------------------------------------------------------

    def add_hook(self, signal_type: str, callback: Callable[[NeuralSignal], None]) -> None:
        """Register a callback fired whenever a signal of this type is published."""
        self._hooks[signal_type].append(callback)

    def remove_hooks(self, signal_type: str) -> None:
        self._hooks[signal_type].clear()

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def recent_history(self, n: int = 20) -> List[NeuralSignal]:
        """Return the N most recent signals published (all types, all regions)."""
        return self._history[-n:]

    def pending_counts(self) -> Dict[str, int]:
        """Return the number of pending signals per region."""
        return {r: len(msgs) for r, msgs in self._inbox.items()}

    def __repr__(self) -> str:
        return (
            f"SignalBus(step={self._step}, "
            f"regions={list(self._subscriptions.keys())}, "
            f"pending={self.pending_counts()})"
        )


__all__ = ["NeuralSignal", "SignalBus"]
