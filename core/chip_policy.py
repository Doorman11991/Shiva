"""
core/chip_policy.py — DEPRECATED: backward-compatibility shim.

ContinuousSACPolicy has moved to cerebrum/chip_policy.py.
"""
# ruff: noqa: F401
from cerebrum.chip_policy import (
    ContinuousActor,
    ContinuousSACPolicy,
    DiscreteValencePolicy,
    DoubleQCritic,
    TASK_VOCAB,
    NUM_TASKS,
    task_id_for,
)
