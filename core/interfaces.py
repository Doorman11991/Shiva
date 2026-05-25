"""
core/interfaces.py — DEPRECATED: backward-compatibility shim.

All interfaces have moved to interfaces/base.py.
This file re-exports everything so existing code continues to work
during the migration period.
"""
# ruff: noqa: F401
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
