"""
core/merge_strategies.py — DEPRECATED: backward-compatibility shim.

RapidFrankenmergeStrategy has moved to thalamus/merge_strategies.py.
"""
# ruff: noqa: F401
from thalamus.merge_strategies import SVDDimensionFitter, AttentionHeadAverager, RapidFrankenmergeStrategy
