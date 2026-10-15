"""Agent drivers: adapters that run one coding-agent CLI headlessly."""

from __future__ import annotations

from collections.abc import Callable

from .base import AgentDriver, AgentRunStatus, RawTranscript, RunPaths
from .fake import FakeDriver
from .opencode import OpenCodeDriver

__all__ = [
    "AgentDriver",
    "AgentRunStatus",
    "RawTranscript",
    "RunPaths",
    "FakeDriver",
    "OpenCodeDriver",
    "DRIVER_REGISTRY",
    "register_driver",
    "get_driver",
]

DRIVER_REGISTRY: dict[str, Callable[[], AgentDriver]] = {}


def register_driver(name: str, factory: Callable[[], AgentDriver]) -> None:
    """Register a driver factory under a CLI-facing name (e.g. "opencode")."""
    DRIVER_REGISTRY[name] = factory


def get_driver(name: str) -> AgentDriver:
    """Instantiate a registered driver, with a helpful error for unknown names."""
    factory = DRIVER_REGISTRY.get(name)
    if factory is None:
        raise ValueError(
            f"Unknown driver {name!r}. Registered drivers: "
            f"{', '.join(sorted(DRIVER_REGISTRY)) or '(none)'}"
        )
    return factory()


register_driver("fake", FakeDriver)
register_driver("opencode", OpenCodeDriver)
