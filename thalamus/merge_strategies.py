"""
thalamus/merge_strategies.py — External signal integration.

The thalamus doesn't just relay internal signals — it also integrates
incoming signals from outside the brain (sensory organs, in biology;
pre-trained external models, in Chip). This module handles the
architecture-agnostic weight transfer from external models (e.g. LLMs)
into Chip's latent space via SVD fitting and attention head averaging.

Moved from: core/merge_strategies.py
"""

from __future__ import annotations
from typing import Any, Dict
import torch
import torch.linalg as linalg
import torch.nn as nn
from interfaces.base import IWeightMergeStrategy


class SVDDimensionFitter:
    """
    Resizes a 2-D weight matrix to a target shape using truncated SVD.

    The original matrix W is factorised as W = U Σ Vᴴ. The leading
    singular components are retained up to the target dimensions, and
    zero-padding fills any remaining entries.
    """

    @staticmethod
    def fit(W: torch.Tensor, target_shape: torch.Size) -> torch.Tensor:
        if W.shape == target_shape:
            return W

        U, S, Vh = linalg.svd(W, full_matrices=False)

        r0, r1 = target_shape
        k = min(len(S), r0, r1)

        U_t = U[:r0, :k]
        S_t = torch.diag(S[:k])
        Vh_t = Vh[:k, :r1]

        result = torch.zeros(target_shape, dtype=W.dtype, device=W.device)
        fitted = U_t @ S_t @ Vh_t
        result[: fitted.shape[0], : fitted.shape[1]] = fitted
        return result


class AttentionHeadAverager:
    """
    Compresses multi-head attention weight matrices by averaging head buckets.

    When the source model has more attention heads than the target, groups of
    consecutive source heads are averaged to produce target heads.
    """

    @staticmethod
    def average(
        W_mha: torch.Tensor,
        src_heads: int,
        target_heads: int,
    ) -> torch.Tensor:
        d_model = W_mha.shape[0]
        d_head = d_model // src_heads
        k = src_heads // target_heads

        reshaped = W_mha.view(src_heads, d_head, d_model)
        averaged = torch.stack(
            [reshaped[i * k : (i + 1) * k].mean(dim=0) for i in range(target_heads)]
        )
        return averaged.view(-1, d_model)


class RapidFrankenmergeStrategy(IWeightMergeStrategy):
    """
    Rapid architecture-agnostic weight ingestion via:
      • Attention head compression  (bucket-averaging)
      • Dimensional fitting         (truncated SVD + zero-padding)

    This strategy attempts a best-effort parameter transfer when source and
    target architectures differ in depth, width, or head count.
    """

    def __init__(self) -> None:
        self._fitter = SVDDimensionFitter()
        self._head_averager = AttentionHeadAverager()

    def merge(
        self,
        target_model: nn.Module,
        ext_state_dict: Dict[str, torch.Tensor],
        ext_config: Dict[str, Any],
    ) -> Dict[str, torch.Tensor]:
        target_dict = target_model.state_dict()

        if hasattr(target_model, "config"):
            target_heads = target_model.config.num_heads
        else:
            target_heads = target_model.backbone.num_heads

        new_state: Dict[str, torch.Tensor] = {}

        for name, param in ext_state_dict.items():
            if "attn" in name or "attention" in name:
                new_state[name] = self._head_averager.average(
                    param, ext_config["num_heads"], target_heads
                )
            else:
                target_shape = (
                    target_dict[name].shape
                    if name in target_dict
                    else param.shape
                )
                new_state[name] = self._fitter.fit(param, target_shape)

        return new_state
