"""
cerebrum/goal_stack.py — Hierarchical goal stack.

Biological role
~~~~~~~~~~~~~~~
Human goal-directed behaviour is hierarchical. "Get fit" decomposes into
"go to the gym" → "do squats" → "extend legs." Each level monitors
completion and replans on failure. This module does the same for Chip.

Architecture
~~~~~~~~~~~~
A stack of GoalFrame objects, each containing:
    - The goal itself (name, target latent, urgency)
    - A list of sub-goals (ordered plan)
    - Completion criterion (cosine proximity to target, or tick limit)
    - Failure criterion (stuck for N ticks, or homeostasis emergency)
    - A pointer to the current active sub-goal

The stack is LIFO: the topmost frame is the most concrete (closest to
primitive action). When a frame completes, it pops and returns control
to its parent. When a frame fails, it pops AND signals the parent to
replan.

Integration
~~~~~~~~~~~
The GoalStack replaces the flat `_active_goals` list in GoalGenerator.
The brain reads `current_goal()` to get the most concrete active goal,
and the policy conditions on its target latent.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# GoalFrame — one level of the hierarchy
# ---------------------------------------------------------------------------

@dataclass
class GoalFrame:
    """One level in the hierarchical goal stack."""
    name: str
    target_latent: Optional[torch.Tensor] = None    # (D,) desired state
    urgency: float = 0.5
    source: str = ""                                 # what created this frame
    sub_goals: List["GoalFrame"] = field(default_factory=list)
    current_sub_idx: int = 0                         # pointer into sub_goals
    ticks_active: int = 0
    max_ticks: int = 100                             # timeout → failure
    completion_threshold: float = 0.7                # cosine sim to target to declare success
    metadata: Dict = field(default_factory=dict)

    @property
    def current_sub_goal(self) -> Optional["GoalFrame"]:
        if self.sub_goals and self.current_sub_idx < len(self.sub_goals):
            return self.sub_goals[self.current_sub_idx]
        return None

    @property
    def is_leaf(self) -> bool:
        return len(self.sub_goals) == 0

    def advance_sub_goal(self) -> bool:
        """Move to next sub-goal. Returns True if there is one, False if all done."""
        self.current_sub_idx += 1
        return self.current_sub_idx < len(self.sub_goals)

    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "urgency": self.urgency,
            "source": self.source,
            "ticks_active": self.ticks_active,
            "max_ticks": self.max_ticks,
            "n_sub_goals": len(self.sub_goals),
            "current_sub_idx": self.current_sub_idx,
            "is_leaf": self.is_leaf,
        }


# ---------------------------------------------------------------------------
# GoalStack
# ---------------------------------------------------------------------------

class GoalStack:
    """
    Hierarchical goal stack with pop-on-completion and replan-on-failure.

    The stack is a list of GoalFrames where index 0 is the most abstract
    (root) and index -1 is the most concrete (currently executing).
    Sub-goals nest inside frames, creating the tree structure.

    Key operations:
        push(frame)          — add a new goal to the stack
        tick(z_current)      — advance, check completion/failure
        current_goal()       — return the most concrete active goal
        complete_current()   — mark current leaf as done, pop/advance
        fail_current()       — mark current as failed, trigger replan

    Args:
        max_depth:           Maximum nesting depth.
        default_max_ticks:   Default timeout for frames without explicit limit.
    """

    def __init__(self, max_depth: int = 5, default_max_ticks: int = 100) -> None:
        self.max_depth = max_depth
        self.default_max_ticks = default_max_ticks
        self._stack: List[GoalFrame] = []
        self._completed_count: int = 0
        self._failed_count: int = 0
        self._replan_count: int = 0

    # ------------------------------------------------------------------
    # Stack operations
    # ------------------------------------------------------------------

    def push(self, frame: GoalFrame) -> bool:
        """
        Push a goal frame. Returns False if max_depth exceeded.
        """
        if len(self._stack) >= self.max_depth:
            return False
        if frame.max_ticks == 0:
            frame.max_ticks = self.default_max_ticks
        self._stack.append(frame)
        return True

    def push_decomposition(
        self,
        parent_name: str,
        sub_goals: List[GoalFrame],
        urgency: float = 0.5,
        source: str = "decomposition",
    ) -> bool:
        """
        Push a parent frame with pre-defined sub-goals.

        This is how abstract goals become concrete: the goal generator
        proposes "restore_energy" and the decomposer breaks it into
        ["reduce_compute", "wait_for_recovery", "verify_energy_level"].
        """
        parent = GoalFrame(
            name=parent_name,
            urgency=urgency,
            source=source,
            sub_goals=sub_goals,
            max_ticks=sum(s.max_ticks for s in sub_goals) + 20,
        )
        return self.push(parent)

    def pop(self) -> Optional[GoalFrame]:
        """Pop the top frame. Returns it or None if stack is empty."""
        if self._stack:
            return self._stack.pop()
        return None

    # ------------------------------------------------------------------
    # Tick: advance timers, check completion and failure
    # ------------------------------------------------------------------

    def tick(self, z_current: Optional[torch.Tensor] = None) -> Optional[str]:
        """
        Advance the stack by one tick. Check completion and timeout.

        Args:
            z_current: (D,) current latent state for proximity checks.

        Returns:
            Event string if something happened:
                "completed" — current goal achieved
                "failed"    — current goal timed out
                "advanced"  — moved to next sub-goal
                None        — nothing changed
        """
        if not self._stack:
            return None

        top = self._stack[-1]
        top.ticks_active += 1

        # If the top has sub-goals, check the current sub-goal
        if not top.is_leaf:
            sub = top.current_sub_goal
            if sub is not None:
                sub.ticks_active += 1
                # Check sub-goal completion
                if self._is_complete(sub, z_current):
                    if top.advance_sub_goal():
                        return "advanced"
                    else:
                        # All sub-goals done → parent is complete
                        return self._complete_top()
                # Check sub-goal timeout
                if sub.ticks_active >= sub.max_ticks:
                    return self._fail_top("sub_goal_timeout")
            else:
                # No more sub-goals → parent complete
                return self._complete_top()
        else:
            # Leaf goal: check completion directly
            if self._is_complete(top, z_current):
                return self._complete_top()
            # Check timeout
            if top.ticks_active >= top.max_ticks:
                return self._fail_top("timeout")

        return None

    def _is_complete(self, frame: GoalFrame, z_current: Optional[torch.Tensor]) -> bool:
        """Check if a frame's goal is achieved via cosine proximity."""
        if frame.target_latent is None or z_current is None:
            return False
        z = z_current.detach().to('cpu') if z_current.device.type != 'cpu' else z_current.detach()
        t = frame.target_latent.to('cpu') if frame.target_latent.device.type != 'cpu' else frame.target_latent
        if z.dim() > 1:
            z = z.squeeze(0)
        sim = float(F.cosine_similarity(z.unsqueeze(0), t.unsqueeze(0)).item())
        return sim >= frame.completion_threshold

    def _complete_top(self) -> str:
        """Pop the top frame as completed."""
        self._stack.pop()
        self._completed_count += 1
        return "completed"

    def _fail_top(self, reason: str = "") -> str:
        """Pop the top frame as failed."""
        failed = self._stack.pop()
        failed.metadata["failure_reason"] = reason
        self._failed_count += 1
        return "failed"

    # ------------------------------------------------------------------
    # Public: read current goal state
    # ------------------------------------------------------------------

    def current_goal(self) -> Optional[GoalFrame]:
        """
        Return the most concrete currently-active goal.

        Walks to the deepest active sub-goal in the topmost frame.
        """
        if not self._stack:
            return None
        top = self._stack[-1]
        if top.is_leaf:
            return top
        sub = top.current_sub_goal
        return sub if sub is not None else top

    def current_target_latent(self) -> Optional[torch.Tensor]:
        """Return the target latent of the most concrete active goal."""
        goal = self.current_goal()
        return goal.target_latent if goal is not None else None

    @property
    def depth(self) -> int:
        return len(self._stack)

    @property
    def is_empty(self) -> bool:
        return len(self._stack) == 0

    def clear(self) -> None:
        self._stack.clear()

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def status(self) -> Dict:
        current = self.current_goal()
        return {
            "depth": self.depth,
            "is_empty": self.is_empty,
            "current_goal": current.name if current else None,
            "completed_count": self._completed_count,
            "failed_count": self._failed_count,
            "stack": [f.to_dict() for f in self._stack],
        }

    def stack_names(self) -> List[str]:
        """Return goal names from bottom (abstract) to top (concrete)."""
        return [f.name for f in self._stack]


__all__ = ["GoalStack", "GoalFrame"]
