"""
amygdala/habituation.py — Dampened response to repeated stimuli.

Biological role
~~~~~~~~~~~~~~~
The first time you hear a loud noise, you startle. The tenth time, you
barely notice. This is habituation — the amygdala's response to a
repeated stimulus decays with exposure. Novel variants of the stimulus
restore the response (dishabituation).

Computational design
~~~~~~~~~~~~~~~~~~~~
Maintain an EMA signature of recently-seen stimuli. For each new
observation, compute its novelty = 1 - similarity to the EMA signature.
Arousal is scaled by this novelty factor:

    effective_arousal = raw_arousal * novelty_factor

Where:
    novelty_factor = 1 - sim(z_new, z_ema)
    z_ema ← decay * z_ema + (1-decay) * z_new

This prevents the agent from being permanently startled by routine
observations while remaining responsive to genuinely new stimuli.
"""

from __future__ import annotations

from collections import deque
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F


class HabituationFilter:
    """
    Dampens arousal response to repeated/familiar stimuli.

    Args:
        latent_dim:      Dimensionality of input latents.
        decay:           EMA decay rate for the stimulus signature.
                         High decay (0.99) = slow habituation (remember longer).
                         Low decay (0.9) = fast habituation.
        floor:           Minimum novelty factor (never fully habituated).
                         Ensures some baseline arousal even for very familiar stimuli.
        dishabit_threshold: Novelty above which the EMA resets (dishabituation).
                           A sudden novel stimulus wipes the accumulated familiarity.
    """

    def __init__(
        self,
        latent_dim: int = 512,
        decay: float = 0.95,
        floor: float = 0.15,
        dishabit_threshold: float = 0.8,
    ) -> None:
        self.latent_dim = latent_dim
        self.decay = decay
        self.floor = floor
        self.dishabit_threshold = dishabit_threshold

        self._ema: Optional[torch.Tensor] = None
        self._exposure_count: int = 0
        self._last_novelty: float = 1.0
        self._novelty_history: deque = deque(maxlen=100)

    def compute_novelty(self, z: torch.Tensor) -> float:
        """
        Compute novelty of a new observation relative to the habituated EMA.

        Args:
            z: (D,) latent vector of the new observation.

        Returns:
            Novelty factor in [floor, 1.0].
            1.0 = completely novel. floor = completely habituated.
        """
        z_flat = z.detach().to('cpu').flatten()
        if z_flat.dim() == 0:
            return 1.0

        z_norm = F.normalize(z_flat, dim=0)

        if self._ema is None:
            # First observation — everything is novel
            self._ema = z_norm.clone()
            self._exposure_count = 1
            self._last_novelty = 1.0
            self._novelty_history.append(1.0)
            return 1.0

        # Compute similarity to habituated signature
        ema_norm = F.normalize(self._ema, dim=0)
        sim = float(torch.dot(z_norm, ema_norm).item())
        novelty = max(self.floor, 1.0 - max(0.0, sim))

        # Dishabituation: if novelty spikes above threshold, reset EMA
        if novelty > self.dishabit_threshold:
            self._ema = z_norm.clone()
            self._exposure_count = 1
        else:
            # Normal EMA update
            self._ema = self.decay * self._ema + (1 - self.decay) * z_norm
            self._exposure_count += 1

        self._last_novelty = novelty
        self._novelty_history.append(novelty)
        return novelty

    def modulate_arousal(self, raw_arousal: float, z: torch.Tensor) -> float:
        """
        Apply habituation to an arousal signal.

        Args:
            raw_arousal: Unfiltered arousal level in [0, 1].
            z:           (D,) current observation latent.

        Returns:
            Modulated arousal = raw_arousal * novelty_factor.
        """
        novelty = self.compute_novelty(z)
        return raw_arousal * novelty

    def reset(self) -> None:
        """Reset habituation (e.g., on context switch or episode boundary)."""
        self._ema = None
        self._exposure_count = 0
        self._last_novelty = 1.0
        self._novelty_history.clear()

    @property
    def is_habituated(self) -> bool:
        """True when recent novelty is near the floor (fully habituated)."""
        if not self._novelty_history:
            return False
        recent_mean = sum(list(self._novelty_history)[-5:]) / min(5, len(self._novelty_history))
        return recent_mean < self.floor * 2.0

    def status(self) -> Dict:
        return {
            "last_novelty": self._last_novelty,
            "exposure_count": self._exposure_count,
            "is_habituated": self.is_habituated,
            "mean_recent_novelty": (
                sum(list(self._novelty_history)[-10:]) / min(10, len(self._novelty_history))
                if self._novelty_history else 1.0
            ),
        }


__all__ = ["HabituationFilter"]
