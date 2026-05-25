"""
interfaces/plugins.py — Extension points for external integrations.

Chip exposes blank slots for tools, environments, sensors, and reward
sources so the host application can plug its own systems in without
modifying the brain itself. None of these are implemented inside Chip —
they are pure contracts that downstream code (the user's existing
assistant, in particular) is expected to satisfy.

Usage from the host application:

    from interfaces.plugins import ITool, ToolRegistry

    class WebSearchTool(ITool):
        name = "web_search"
        def call(self, args: dict) -> dict:
            return {"results": [...]}

    registry = ToolRegistry()
    registry.register(WebSearchTool())

    # Hand the registry to the brain at boot time:
    brain = ChipBrain(tool_registry=registry).boot()
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional


# ---------------------------------------------------------------------------
# Tool / action plugin
# ---------------------------------------------------------------------------

class ITool(ABC):
    """
    Contract for a tool the brain can invoke.

    Each tool has a stable name, a JSON schema describing its arguments,
    and a callable that executes it. The brain treats tools as opaque —
    it never inspects their internals, only routes to them by name.
    """

    name: str = ""
    schema: Dict[str, Any] = {}

    @abstractmethod
    def call(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the tool with the given arguments. Return a result dict."""
        ...


class ToolRegistry:
    """A simple name-keyed registry of ITool implementations."""

    def __init__(self) -> None:
        self._tools: Dict[str, ITool] = {}

    def register(self, tool: ITool) -> None:
        if not tool.name:
            raise ValueError("ITool must have a non-empty name")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Optional[ITool]:
        return self._tools.get(name)

    def names(self) -> List[str]:
        return list(self._tools.keys())

    def call(self, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        tool = self.get(name)
        if tool is None:
            return {"error": f"unknown tool: {name}"}
        try:
            return tool.call(args)
        except Exception as e:
            return {"error": f"{type(e).__name__}: {e}"}


# ---------------------------------------------------------------------------
# Environment plugin
# ---------------------------------------------------------------------------

class IEnvironment(ABC):
    """
    Contract for an environment Chip can act inside.

    Implementations might wrap a chat session, a gym environment, a robot
    control loop, or anything else that produces observations and accepts
    actions. The brain never assumes a specific environment type.
    """

    @abstractmethod
    def reset(self) -> Any:
        """Start a new episode. Return the initial observation."""
        ...

    @abstractmethod
    def step(self, action: Any) -> tuple:
        """
        Apply an action and advance one step.
        Returns (next_observation, reward, done, info).
        """
        ...


# ---------------------------------------------------------------------------
# Sensor plugin
# ---------------------------------------------------------------------------

class ISensor(ABC):
    """
    Contract for a custom sensory modality.

    Implementations should produce a fixed-dim float tensor on demand.
    They are registered with the thalamus SensoryEncoder via
    `register_modality(name, input_dim)`.
    """

    name: str = ""
    input_dim: int = 0

    @abstractmethod
    def read(self) -> Any:
        """Return the current sensor reading as a float tensor."""
        ...


# ---------------------------------------------------------------------------
# Reward source plugin
# ---------------------------------------------------------------------------

class IRewardSource(ABC):
    """
    Contract for an external reward signal.

    Multiple sources can be combined (task reward + human feedback +
    safety penalty). The brain composes them additively unless the
    host overrides the combination strategy.
    """

    name: str = ""

    @abstractmethod
    def evaluate(self, observation: Any, action: Any, next_observation: Any) -> float:
        """Return a scalar reward for the transition."""
        ...


# ---------------------------------------------------------------------------
# Generic callback hook (for host-side observation of brain events)
# ---------------------------------------------------------------------------

class HookRegistry:
    """
    Lets the host subscribe to brain lifecycle events without modifying
    the brain. Events fire as named strings — the brain calls
    `hooks.fire(event_name, payload)` and any registered callback runs.
    """

    def __init__(self) -> None:
        self._hooks: Dict[str, List[Callable[[Any], None]]] = {}

    def on(self, event_name: str, callback: Callable[[Any], None]) -> None:
        self._hooks.setdefault(event_name, []).append(callback)

    def fire(self, event_name: str, payload: Any = None) -> None:
        for cb in self._hooks.get(event_name, []):
            try:
                cb(payload)
            except Exception:
                pass  # host-side errors must never crash the brain


__all__ = [
    "ITool",
    "ToolRegistry",
    "IEnvironment",
    "ISensor",
    "IRewardSource",
    "HookRegistry",
]
