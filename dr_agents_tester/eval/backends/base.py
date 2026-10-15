"""ExecutionBackend protocol: the seam between the evaluator loop and how a run executes.

The evaluator (scenarios × conditions × k runs, stats, reports) is backend-
agnostic. ``PlanBackend`` preserves the original single-LLM-call plan
evaluation; ``AgentBackend`` drives a real coding agent in a sandbox.
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from ..models import Difficulty, EvalCondition, EvalResult, ScenarioBase


@dataclass
class RunContext:
    """Per-run execution context handed to a backend."""

    run_number: int
    run_id: str
    artifacts_dir: Path | None = None


class ExecutionBackend(Protocol):
    """Executes one (scenario, condition, run) cell.

    ``load_scenarios`` lives on the backend because the scenario kind and the
    execution semantics are one decision: a backend only ever executes the
    scenario type its loader produces (backends narrow with ``isinstance``).
    """

    name: str

    def load_scenarios(
        self, scenarios_dir: Path, difficulty_filter: Difficulty | None
    ) -> list[ScenarioBase]: ...

    def execute(
        self, scenario: ScenarioBase, condition: EvalCondition, ctx: RunContext
    ) -> EvalResult: ...


def make_run_id(
    scenario: ScenarioBase,
    condition: EvalCondition,
    run_number: int,
    prefix: str | None = None,
) -> str:
    """Build a unique, DataRobot-resource-name-safe run id.

    Format: ``drat-<context>-<condition>-r<n>-<timestamp><4hex>`` — lowercase
    and hyphenated. The literal ``drat-`` family prefix is what the resource
    sweeper matches; the embedded timestamp is its age fallback.
    """
    import datetime as _dt

    context = prefix if prefix else f"drat-{scenario.id[:24]}"
    stamp = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%d%H%M%S")
    raw = f"{context}-{condition.condition_type.value}-r{run_number}-{stamp}{uuid.uuid4().hex[:4]}"
    return re.sub(r"[^a-z0-9-]+", "-", raw.lower())
