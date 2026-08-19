"""Base types for the programmatic success-check library.

Success checks are the authoritative pass/fail signal for behavioral runs
(design doc §4.1): the LLM judge only ever scores *how* an agent got there,
while these checks assert real DataRobot / filesystem state.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

from ...config import Config
from ..models import CheckResult


@dataclass
class CheckContext:
    """Everything a check may need to assert on real state.

    ``dr_client_factory`` is injectable so tests can supply a fake client;
    when ``None``, DataRobot checks build a real client from ``config``.
    """

    workspace: Path
    run_id: str
    config: Config
    env: Mapping[str, str] = field(default_factory=dict)
    dr_client_factory: Callable[[], Any] | None = None


class OutcomeCheck(ABC):
    """A single programmatic assertion against post-run state.

    Subclasses set ``type_name`` (the YAML ``type`` key) and are registered
    into the check registry via :func:`dr_agents_tester.eval.checks.register_check`.
    """

    type_name: ClassVar[str]

    def __init__(self, params: dict[str, object]) -> None:
        self.params = params

    @abstractmethod
    def run(self, ctx: CheckContext) -> CheckResult: ...


def param_int(params: dict[str, object], key: str, default: int) -> int:
    """Coerce an int-like check param (YAML may deliver int or str)."""
    value = params.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise ValueError(f"Check param {key!r} must be an integer, got {value!r}")
    return int(value)


def render_params(params: dict[str, object], run_id: str) -> dict[str, object]:
    """Substitute the literal token ``{run_id}`` in every string value.

    Uses ``str.replace`` rather than ``str.format`` so any other braces in
    parameter values survive untouched.
    """
    rendered: dict[str, object] = {}
    for key, value in params.items():
        if isinstance(value, str):
            rendered[key] = value.replace("{run_id}", run_id)
        else:
            rendered[key] = value
    return rendered
