"""Success checks that assert on real DataRobot state.

The module itself imports without the ``datarobot`` SDK so the check registry
(and therefore scenario-YAML validation) works offline; the SDK import happens
lazily inside each check's ``run`` — mirroring the optional-dependency pattern
in ``dr_agents_tester.pytest_plugin``.
"""

from __future__ import annotations

import csv
import re
import time
from typing import Any

from ..models import CheckResult
from .base import CheckContext, OutcomeCheck, param_int

_INSTALL_HINT = (
    "The 'datarobot' SDK is required for dr_* checks. "
    "Install with: pip install 'datarobot-agent-tester[behavioral]'"
)


def _dr_module(ctx: CheckContext) -> Any:
    """Return the datarobot module (or an injected fake) with a client configured."""
    if ctx.dr_client_factory is not None:
        return ctx.dr_client_factory()
    try:
        import datarobot as dr
    except ImportError as exc:  # pragma: no cover - exercised via message test
        raise ImportError(_INSTALL_HINT) from exc
    dr.Client(token=ctx.config.api_key, endpoint=ctx.config.endpoint)
    return dr


def _timed(
    check: OutcomeCheck, start: float, passed: bool, evidence: str, error: str | None = None
) -> CheckResult:
    return CheckResult(
        check_type=type(check).type_name,
        passed=passed,
        evidence=evidence,
        error=error,
        duration_seconds=round(time.monotonic() - start, 3),
    )


class DrProjectExistsCheck(OutcomeCheck):
    """Assert a DataRobot project whose name contains a substring exists.

    Params:
        name_contains: substring to match (typically ``{run_id}``).
    """

    type_name = "dr_project_exists"

    def run(self, ctx: CheckContext) -> CheckResult:
        start = time.monotonic()
        needle = str(self.params.get("name_contains", ctx.run_id))
        dr = _dr_module(ctx)
        projects = dr.Project.list(search_params={"project_name": needle})
        if projects:
            names = ", ".join(f"{p.project_name} ({p.id})" for p in projects[:3])
            return _timed(self, start, True, f"{len(projects)} project(s) matched: {names}")
        return _timed(self, start, False, f"no project name contains {needle!r}")


class DrDeploymentHealthyCheck(OutcomeCheck):
    """Assert a deployment matching a label substring exists and is not failing.

    A ``service_health`` status of ``unknown`` passes — deployments that have
    served few or no predictions legitimately report unknown.

    Params:
        name_contains: substring to match against the deployment label.
    """

    type_name = "dr_deployment_healthy"

    def run(self, ctx: CheckContext) -> CheckResult:
        start = time.monotonic()
        needle = str(self.params.get("name_contains", ctx.run_id))
        dr = _dr_module(ctx)
        deployments = [d for d in dr.Deployment.list(search=needle) if needle in (d.label or "")]
        if not deployments:
            return _timed(self, start, False, f"no deployment label contains {needle!r}")

        for dep in deployments:
            model = getattr(dep, "model", None)
            health = (getattr(dep, "service_health", None) or {}).get("status")
            if model and health != "failing":
                model_id = model.get("id") if isinstance(model, dict) else model
                return _timed(
                    self,
                    start,
                    True,
                    f"deployment {dep.label!r} ({dep.id}) model={model_id} health={health}",
                )

        details = ", ".join(
            f"{d.label!r}: model={'yes' if getattr(d, 'model', None) else 'no'} "
            f"health={(getattr(d, 'service_health', None) or {}).get('status')}"
            for d in deployments
        )
        return _timed(self, start, False, f"matched but unhealthy/incomplete: {details}")


class DrPredictionsReturnedCheck(OutcomeCheck):
    """Assert predictions were produced for this run.

    The primary assertion is on the workspace artifact: a CSV with at least
    ``min_rows`` data rows and a prediction-looking column. When
    ``verify_server`` is true, additionally corroborate that a completed batch
    prediction job exists (best-effort — agents may legitimately use the
    real-time prediction API, which leaves no batch job).

    Params:
        path: workspace-relative CSV path (default ``predictions.csv``).
        min_rows: minimum number of data rows (default 1).
        prediction_column: exact column name to require; when omitted, any
            column matching ``prediction`` case-insensitively is accepted.
        verify_server: also look for a completed batch prediction job.
    """

    type_name = "dr_predictions_returned"

    def run(self, ctx: CheckContext) -> CheckResult:
        start = time.monotonic()
        rel_path = str(self.params.get("path", "predictions.csv"))
        min_rows = param_int(self.params, "min_rows", 1)
        wanted_col = self.params.get("prediction_column")
        verify_server = bool(self.params.get("verify_server", False))

        csv_path = ctx.workspace / rel_path
        if not csv_path.is_file():
            return _timed(self, start, False, f"no prediction file at {rel_path!r}")

        with csv_path.open(newline="") as fh:
            reader = csv.reader(fh)
            try:
                header = next(reader)
            except StopIteration:
                return _timed(self, start, False, f"{rel_path!r} is empty")
            n_rows = sum(1 for _ in reader)

        if wanted_col is not None:
            matched = str(wanted_col) if str(wanted_col) in header else None
        else:
            matched = next(
                (col for col in header if re.search("prediction", col, re.IGNORECASE)), None
            )
        if matched is None:
            return _timed(
                self,
                start,
                False,
                f"{rel_path!r} has no prediction column (header: {header[:8]})",
            )
        if n_rows < min_rows:
            return _timed(self, start, False, f"{rel_path!r} has {n_rows} rows, need >= {min_rows}")

        evidence = f"{rel_path!r}: {n_rows} rows, prediction column {matched!r}"
        if verify_server:
            dr = _dr_module(ctx)
            jobs = dr.BatchPredictionJob.list_by_status(["completed"])
            evidence += f"; {len(jobs)} completed batch job(s) on server"
        return _timed(self, start, True, evidence)
