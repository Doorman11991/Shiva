"""
cerebellum/swarm_coordinator.py — Multi-agent motor coordination.

The cerebellum coordinates smooth, precise movement across multiple
muscle groups without conscious effort. In Chip, the swarm coordinator
does the same across multiple cognitive agents: it aggregates their
local latent states via cross-attention, broadcasts a consensus vector,
and penalises collapse (all agents thinking the same thing).

Moved from: swarm/SwarmAlgorithmWorkspace.py

Design: Baars' Global Workspace Theory (1988) — consciousness arises
when local specialist processors broadcast to a shared blackboard and
competition + integration produce a single coherent representation.
"""

from __future__ import annotations
import math
from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from interfaces.base import IGlobalWorkspace, ISwarmNode


class SwarmNode(ISwarmNode, nn.Module):
    def __init__(self, latent_dim: int, node_id: str) -> None:
        nn.Module.__init__(self)
        self.node_id = node_id
        self.latent_dim = latent_dim
        self._integration_gate = nn.Parameter(torch.tensor(math.log(0.1 / 0.9)))
        self.register_buffer("_local_latent", torch.zeros(latent_dim))

    def set_conscious_latent(self, z: torch.Tensor) -> None:
        self._local_latent = z.detach()

    def get_conscious_latent(self) -> torch.Tensor:
        return self._local_latent  # type: ignore[return-value]

    def receive_consensus(self, consensus_vector: torch.Tensor) -> None:
        gate = torch.sigmoid(self._integration_gate)
        self._local_latent = self._local_latent + gate * consensus_vector.detach()


class CrossAttentionAggregator(nn.Module):
    def __init__(self, latent_dim: int, num_heads: int = 8) -> None:
        super().__init__()
        assert latent_dim % num_heads == 0
        self.latent_dim = latent_dim
        self.num_heads = num_heads
        self.d_k = latent_dim // num_heads
        self.W_K = nn.Linear(latent_dim, latent_dim, bias=False)
        self.W_V = nn.Linear(latent_dim, latent_dim, bias=False)
        self.query = nn.Parameter(torch.randn(1, latent_dim))
        self.W_out = nn.Linear(latent_dim, latent_dim)
        self.norm = nn.LayerNorm(latent_dim)

    def forward(self, node_latents: torch.Tensor) -> torch.Tensor:
        N, D = node_latents.shape
        H, d_k = self.num_heads, self.d_k
        K = self.W_K(node_latents).view(N, H, d_k).transpose(0, 1)
        V = self.W_V(node_latents).view(N, H, d_k).transpose(0, 1)
        q = self.query.view(1, H, d_k).transpose(0, 1)
        scores = (q @ K.transpose(-2, -1)) / math.sqrt(d_k)
        weights = F.softmax(scores, dim=-1)
        out = (weights @ V).transpose(0, 1).contiguous().view(1, D).squeeze(0)
        residual = node_latents.mean(dim=0)
        return self.norm(self.W_out(out) + residual)


class GlobalWorkspace(IGlobalWorkspace, nn.Module):
    def __init__(self, latent_dim: int, num_heads: int = 8) -> None:
        nn.Module.__init__(self)
        self.latent_dim = latent_dim
        self._nodes: Dict[str, ISwarmNode] = {}
        self.aggregator = CrossAttentionAggregator(latent_dim, num_heads)
        self._last_diversity_loss: Optional[torch.Tensor] = None

    def register_node(self, node_id: str, node: ISwarmNode) -> None:
        self._nodes[node_id] = node

    def broadcast_consensus(self) -> torch.Tensor:
        if not self._nodes:
            raise RuntimeError("No nodes registered in GlobalWorkspace.")
        node_ids = list(self._nodes.keys())
        latents = torch.stack([self._nodes[nid].get_conscious_latent() for nid in node_ids])
        consensus = self.aggregator(latents)
        for nid in node_ids:
            self._nodes[nid].receive_consensus(consensus)
        self._last_diversity_loss = self._compute_diversity_loss(latents)
        return consensus

    @staticmethod
    def _compute_diversity_loss(latents: torch.Tensor) -> torch.Tensor:
        z_norm = F.normalize(latents, p=2, dim=1)
        N = z_norm.shape[0]
        if N < 2:
            return torch.tensor(0.0, device=latents.device)
        diff = z_norm.unsqueeze(0) - z_norm.unsqueeze(1)
        dist = torch.norm(diff, p=2, dim=-1)
        mask = torch.triu(torch.ones(N, N, device=latents.device), diagonal=1)
        n_pairs = mask.sum()
        mean_dist = (dist * mask).sum() / n_pairs
        return -mean_dist

    @property
    def last_diversity_loss(self) -> Optional[torch.Tensor]:
        return self._last_diversity_loss


class SwarmCoordinator(nn.Module):
    """
    Top-level orchestrator for the swarm.

    Manages SwarmNodes and the GlobalWorkspace. Drives per-step consensus
    cycles and exposes the diversity loss for the trainer.
    """

    def __init__(
        self,
        latent_dim: int,
        n_nodes: int,
        num_heads: int = 8,
        node_ids: Optional[list] = None,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        ids = node_ids or [f"node_{i}" for i in range(n_nodes)]
        assert len(ids) == n_nodes
        self.workspace = GlobalWorkspace(latent_dim, num_heads)
        self._node_modules = nn.ModuleDict({
            nid: SwarmNode(latent_dim, nid) for nid in ids
        })
        for nid, node in self._node_modules.items():
            self.workspace.register_node(nid, node)  # type: ignore[arg-type]

    def update_node_latent(self, node_id: str, z: torch.Tensor) -> None:
        self._node_modules[node_id].set_conscious_latent(z)  # type: ignore[attr-defined]

    def step(self) -> Tuple[torch.Tensor, torch.Tensor]:
        consensus = self.workspace.broadcast_consensus()
        div_loss = self.workspace.last_diversity_loss
        return consensus, div_loss if div_loss is not None else torch.tensor(0.0)

    def get_diversity_loss(self) -> torch.Tensor:
        div = self.workspace.last_diversity_loss
        return div if div is not None else torch.tensor(0.0)

    def get_node_latent(self, node_id: str) -> torch.Tensor:
        return self._node_modules[node_id].get_conscious_latent()  # type: ignore[attr-defined]
