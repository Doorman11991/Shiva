"""
core/transformer_architecture.py — DEPRECATED: backward-compatibility shim.

TransformerEncoderBlock has moved to thalamus/transformer_backbone.py.
"""
# ruff: noqa: F401
from thalamus.transformer_backbone import GateHyperNetwork, TransformerEncoderBlock
