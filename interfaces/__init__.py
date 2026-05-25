"""
interfaces/ — White matter of the Chip brain.

All abstract base classes and shared signal types live here.
No region imports from another region directly; they communicate
through these contracts and the signal bus.
"""
from interfaces.base import (
    IEpisodicMemory,
    IActor,
    IReplayBuffer,
    IAlignmentLoss,
    IWeightMergeStrategy,
    ISwarmNode,
    IGlobalWorkspace,
    IRepresentationProbe,
    ICognitiveSnapshot,
    ILocomotionTransport,
    IBrainRegion,
)
from interfaces.signals import NeuralSignal, SignalBus

__all__ = [
    "IEpisodicMemory", "IActor", "IReplayBuffer", "IAlignmentLoss",
    "IWeightMergeStrategy", "ISwarmNode", "IGlobalWorkspace",
    "IRepresentationProbe", "ICognitiveSnapshot", "ILocomotionTransport",
    "IBrainRegion", "NeuralSignal", "SignalBus",
]
