"""
thalamus/ — Sensory relay and routing.

The transformer backbone, multi-modal latent alignment, attention
bottleneck, granite-backed text encoder, and external weight integration
all live here. Every signal entering the brain passes through the
thalamus first.
Analogous to the thalamic relay nuclei.
"""

from thalamus.granite_embedder import GraniteEmbedder, get_embedder, reset_embedder

__all__ = ["GraniteEmbedder", "get_embedder", "reset_embedder"]
