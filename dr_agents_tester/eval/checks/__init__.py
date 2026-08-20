"""Programmatic success-check library: registry, dispatch, and execution.

Checks decide behavioral pass/fail (``outcome_pass``); the LLM judge only ever
scores process quality. The registry is importable with no optional
dependencies installed so scenario-YAML validation works offline.
"""

from __future__ import annotations

import time
from collections.abc import Mapping

from ..models import CheckResult, CheckSpec, OutcomeResult
from .base import CheckContext, OutcomeCheck, render_params
from .datarobot import (
    DrDeploymentHealthyCheck,
    DrPredictionsReturnedCheck,
    DrProjectExistsCheck,
    DrTracesReceivedCheck,
    DrUseCaseExistsCheck,
)
from .local import FileExistsCheck, FileMatchesCheck

__all__ = [
    "CheckContext",
    "OutcomeCheck",
    "CHECK_REGISTRY",
    "register_check",
    "known_check_types",
    "build_checks",
    "run_checks",
    "render_params",
]

CHECK_REGISTRY: dict[str, type[OutcomeCheck]] = {}


def register_check(cls: type[OutcomeCheck]) -> type[OutcomeCheck]:
    """Register a check class under its ``type_name``. Usable as a decorator."""
    name = cls.type_name
    existing = CHECK_REGISTRY.get(name)
    if existing is not None and existing is not cls:
        raise ValueError(f"Check type {name!r} already registered by {existing.__name__}")
    CHECK_REGISTRY[name] = cls
    return cls


def known_check_types() -> frozenset[str]:
    """The set of valid ``success_checks[].type`` values."""
    return frozenset(CHECK_REGISTRY)


def build_checks(
    specs: list[CheckSpec],
    run_id: str,
    host_env: Mapping[str, str] | None = None,
) -> list[OutcomeCheck]:
    """Instantiate checks from specs, templating ``{run_id}``/``{env:VAR}`` into params."""
    checks: list[OutcomeCheck] = []
    for spec in specs:
        cls = CHECK_REGISTRY.get(spec.type)
        if cls is None:
            raise ValueError(
                f"Unknown check type {spec.type!r}. Known types: "
                f"{', '.join(sorted(CHECK_REGISTRY))}"
            )
        checks.append(cls(render_params(spec.params, run_id, host_env)))
    return checks


def run_checks(
    specs: list[CheckSpec],
    ctx: CheckContext,
    host_env: Mapping[str, str] | None = None,
) -> OutcomeResult:
    """Run every check (even after failures) and aggregate into an OutcomeResult.

    A check that raises becomes a failed :class:`CheckResult` with ``error``
    set — one broken check never hides the results of the others.
    """
    results: list[CheckResult] = []
    for check in build_checks(specs, ctx.run_id, host_env):
        start = time.monotonic()
        try:
            results.append(check.run(ctx))
        except Exception as exc:  # noqa: BLE001 — checks must never abort the suite
            results.append(
                CheckResult(
                    check_type=type(check).type_name,
                    passed=False,
                    evidence="",
                    error=f"{type(exc).__name__}: {exc}",
                    duration_seconds=round(time.monotonic() - start, 3),
                )
            )
    return OutcomeResult(outcome_pass=all(r.passed for r in results), checks=results)


register_check(FileExistsCheck)
register_check(FileMatchesCheck)
register_check(DrProjectExistsCheck)
register_check(DrDeploymentHealthyCheck)
register_check(DrPredictionsReturnedCheck)
register_check(DrUseCaseExistsCheck)
register_check(DrTracesReceivedCheck)
