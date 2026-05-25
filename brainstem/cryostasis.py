"""
brainstem/cryostasis.py — Persistent state / autosave.

Biological role
~~~~~~~~~~~~~~~
The brainstem keeps the body alive across sleep, anaesthesia, even injury.
This module is its computational analogue: it persists the brain's state
to disk on a schedule (and on graceful shutdown) so a process restart
doesn't destroy everything Chip has learned.

Design
~~~~~~
Builds on `locomotion.CognitiveSnapshot` (which already does HMAC-signed
serialisation of the policy + episodic memory + emotional state +
identity token). The new piece is:

    - Disk-backed snapshots with rolling rotation
    - Scheduled autosave (every N ticks)
    - Auto-restore on boot if a checkpoint is present
    - Graceful crash recovery via the latest valid checkpoint
    - Atomic writes (write to .tmp, then rename) so a crash mid-write
      never corrupts the active checkpoint

Storage layout (under `state_dir`):
    latest.snap          → current (atomic-renamed) snapshot
    rolling_001.snap     → previous N snapshots for rollback
    rolling_002.snap
    ...
    metadata.json        → tick count, save count, last save time
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

import torch.nn as nn

from locomotion.ModelMovementAndLocomotion import CognitiveSnapshot


class Cryostasis:
    """
    Disk-backed persistence for the Chip brain.

    Args:
        state_dir:    Directory to store snapshots. Created if missing.
        save_every:   Save every N ticks. 0 = manual only.
        rolling_keep: Number of rolling backups to retain.
        hmac_secret:  Secret for HMAC-SHA256 signing. CHANGE IN PRODUCTION.
        node_id:      Stable id for this Chip instance.
    """

    def __init__(
        self,
        state_dir: str = ".chip_state",
        save_every: int = 500,
        rolling_keep: int = 5,
        hmac_secret: Optional[bytes] = None,
        node_id: str = "Chip",
    ) -> None:
        self.state_dir = Path(state_dir).resolve()
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.save_every = save_every
        self.rolling_keep = rolling_keep
        self.hmac_secret = hmac_secret or b"chip-dev-secret"
        self.node_id = node_id

        self._save_count: int = 0
        self._last_save_time: float = 0.0
        self._last_save_tick: int = 0

        self._meta_path = self.state_dir / "metadata.json"
        self._latest_path = self.state_dir / "latest.snap"
        self._load_metadata()

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    def _load_metadata(self) -> None:
        if self._meta_path.exists():
            try:
                with self._meta_path.open("r", encoding="utf-8") as f:
                    meta = json.load(f)
                self._save_count = int(meta.get("save_count", 0))
                self._last_save_time = float(meta.get("last_save_time", 0.0))
                self._last_save_tick = int(meta.get("last_save_tick", 0))
            except Exception:
                # Corrupt metadata is non-fatal; we just start fresh.
                pass

    def _write_metadata(self, tick: int) -> None:
        meta = {
            "save_count": self._save_count,
            "last_save_time": self._last_save_time,
            "last_save_tick": self._last_save_tick,
            "node_id": self.node_id,
            "tick": tick,
        }
        with self._meta_path.open("w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def maybe_save(
        self,
        tick: int,
        policy: nn.Module,
        episodic_memory: nn.Module,
        emotional_core: nn.Module,
    ) -> bool:
        """
        Save a snapshot if scheduled. Returns True iff a save was performed.
        """
        if self.save_every <= 0:
            return False
        if tick - self._last_save_tick < self.save_every:
            return False
        return self.save(tick, policy, episodic_memory, emotional_core)

    def save(
        self,
        tick: int,
        policy: nn.Module,
        episodic_memory: nn.Module,
        emotional_core: nn.Module,
    ) -> bool:
        """
        Force a snapshot save. Returns True on success.

        Atomic write: serialises to a .tmp file then os.replace's it onto
        the active path so a crash mid-write never leaves a torn checkpoint.
        """
        try:
            snap = CognitiveSnapshot.capture(
                policy=policy,
                episodic_memory=episodic_memory,
                emotional_core=emotional_core,
                node_id=self.node_id,
                hmac_secret=self.hmac_secret,
            )
            blob = snap.serialise()

            # Atomic write
            tmp_path = self._latest_path.with_suffix(".snap.tmp")
            with tmp_path.open("wb") as f:
                f.write(blob)

            # Rotate the previous latest into rolling history
            if self._latest_path.exists():
                self._rotate_rolling()
                rolling_target = self.state_dir / f"rolling_{self._save_count % self.rolling_keep:03d}.snap"
                os.replace(self._latest_path, rolling_target)

            os.replace(tmp_path, self._latest_path)

            self._save_count += 1
            self._last_save_time = time.time()
            self._last_save_tick = tick
            self._write_metadata(tick)
            return True
        except Exception as e:
            print(f"[Cryostasis] save failed: {type(e).__name__}: {e}")
            return False

    def _rotate_rolling(self) -> None:
        """Trim rolling backups to `rolling_keep`."""
        rolling = sorted(self.state_dir.glob("rolling_*.snap"))
        while len(rolling) > self.rolling_keep:
            oldest = rolling.pop(0)
            try:
                oldest.unlink()
            except OSError:
                pass

    # ------------------------------------------------------------------
    # Restore
    # ------------------------------------------------------------------

    def restore_if_available(
        self,
        policy: nn.Module,
        episodic_memory: nn.Module,
        emotional_core: nn.Module,
        device: str = "cpu",
    ) -> Optional[Dict[str, Any]]:
        """
        Restore from `latest.snap` if it exists. On failure, falls back to
        the most recent rolling backup.

        Returns a dict with restore details, or None if nothing was restored.
        """
        if self._latest_path.exists():
            result = self._try_restore(
                self._latest_path, policy, episodic_memory, emotional_core, device
            )
            if result is not None:
                return result
            print("[Cryostasis] latest.snap corrupt or invalid; trying rolling backups")

        # Fall back to rolling backups, newest first
        rolling = sorted(
            self.state_dir.glob("rolling_*.snap"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        for path in rolling:
            result = self._try_restore(path, policy, episodic_memory, emotional_core, device)
            if result is not None:
                print(f"[Cryostasis] recovered from rolling backup: {path.name}")
                return result

        return None

    def _try_restore(
        self,
        path: Path,
        policy: nn.Module,
        episodic_memory: nn.Module,
        emotional_core: nn.Module,
        device: str,
    ) -> Optional[Dict[str, Any]]:
        try:
            with path.open("rb") as f:
                blob = f.read()
            snap = CognitiveSnapshot.deserialise(blob, hmac_secret=self.hmac_secret)
            snap.restore(
                policy=policy,
                episodic_memory=episodic_memory,
                emotional_core=emotional_core,
                device=device,
            )
            return {
                "path": str(path),
                "node_id": snap.metadata.node_id,
                "snapshot_timestamp": snap.metadata.timestamp,
                "bytes": len(blob),
                "schema_version": snap.metadata.schema_version,
            }
        except Exception as e:
            print(f"[Cryostasis] failed to restore {path.name}: {type(e).__name__}: {e}")
            return None

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def status(self) -> Dict[str, Any]:
        return {
            "state_dir": str(self.state_dir),
            "save_count": self._save_count,
            "last_save_tick": self._last_save_tick,
            "last_save_age_s": (
                time.time() - self._last_save_time if self._last_save_time else None
            ),
            "save_every": self.save_every,
            "latest_exists": self._latest_path.exists(),
            "rolling_count": len(list(self.state_dir.glob("rolling_*.snap"))),
        }


__all__ = ["Cryostasis"]
