"""Delete DataRobot resources created by behavioral runs, by name prefix.

Two entry points: per-run teardown right after a run's checks (exact run_id
prefix), and the sweeper for leaked resources (the ``drat-`` family prefix
plus an age threshold), wired to ``dr-agent eval sweep``.
"""

from __future__ import annotations

import datetime as dt
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

from ...config import Config

_INSTALL_HINT = (
    "The 'datarobot' SDK is required for resource teardown. "
    "Install with: pip install 'datarobot-agent-tester[behavioral]'"
)

#: Timestamp embedded in run ids by make_run_id (YYYYmmddHHMMSS).
_NAME_STAMP = re.compile(r"(20\d{12})")


@dataclass
class TeardownReport:
    """What a teardown pass deleted (or would delete, in dry-run) and what failed."""

    deleted: dict[str, list[str]] = field(default_factory=dict)
    failed: dict[str, list[str]] = field(default_factory=dict)
    dry_run: bool = False

    def record(self, kind: str, label: str) -> None:
        self.deleted.setdefault(kind, []).append(label)

    def record_failure(self, kind: str, label: str, error: Exception) -> None:
        self.failed.setdefault(kind, []).append(f"{label}: {type(error).__name__}: {error}")


def _dr_module(config: Config) -> Any:
    try:
        import datarobot as dr
    except ImportError as exc:  # pragma: no cover - message checked in tests
        raise ImportError(_INSTALL_HINT) from exc
    dr.Client(token=config.api_key, endpoint=config.endpoint)
    return dr


def _resource_age_hours(resource: Any, now: dt.datetime) -> float | None:
    """Best-effort age: SDK creation timestamp, else the stamp in the name."""
    for attr in ("created", "creation_date", "created_at"):
        value = getattr(resource, attr, None)
        if isinstance(value, dt.datetime):
            created = value if value.tzinfo else value.replace(tzinfo=dt.timezone.utc)
            return (now - created).total_seconds() / 3600

    for attr in ("project_name", "label", "name"):
        name = getattr(resource, attr, None)
        if isinstance(name, str):
            match = _NAME_STAMP.search(name)
            if match:
                try:
                    created = dt.datetime.strptime(match.group(1), "%Y%m%d%H%M%S").replace(
                        tzinfo=dt.timezone.utc
                    )
                except ValueError:
                    continue
                return (now - created).total_seconds() / 3600
    return None


def _delete_matching(
    report: TeardownReport,
    kind: str,
    resources: Iterable[Any],
    name_of: Callable[[Any], str],
    delete: Callable[[Any], None],
    prefix: str,
    older_than_hours: float,
    now: dt.datetime,
    dry_run: bool,
) -> None:
    for resource in resources:
        name = name_of(resource)
        if not name.startswith(prefix):
            continue
        if older_than_hours > 0:
            age = _resource_age_hours(resource, now)
            # Unknown age: skip rather than risk deleting a run still in flight.
            if age is None or age < older_than_hours:
                continue
        try:
            if not dry_run:
                delete(resource)
            report.record(kind, name)
        except Exception as exc:  # noqa: BLE001 — one failure never stops the sweep
            report.record_failure(kind, name, exc)


def delete_run_resources(
    dr: Any,
    prefix: str,
    older_than_hours: float = 0.0,
    dry_run: bool = False,
) -> TeardownReport:
    """Delete every DataRobot resource whose name starts with ``prefix``.

    Children before parents: deployments → projects → datasets → use cases.
    Every deletion is individually guarded; failures are reported, not raised.
    """
    report = TeardownReport(dry_run=dry_run)
    now = dt.datetime.now(dt.timezone.utc)

    _delete_matching(
        report,
        "deployments",
        dr.Deployment.list(search=prefix),
        lambda d: str(getattr(d, "label", "") or ""),
        lambda d: d.delete(),
        prefix,
        older_than_hours,
        now,
        dry_run,
    )
    _delete_matching(
        report,
        "projects",
        dr.Project.list(search_params={"project_name": prefix}),
        lambda p: str(getattr(p, "project_name", "") or ""),
        lambda p: p.delete(),
        prefix,
        older_than_hours,
        now,
        dry_run,
    )
    _delete_matching(
        report,
        "datasets",
        dr.Dataset.list(),
        lambda ds: str(getattr(ds, "name", "") or ""),
        lambda ds: dr.Dataset.delete(ds.id),
        prefix,
        older_than_hours,
        now,
        dry_run,
    )
    _delete_matching(
        report,
        "use_cases",
        dr.UseCase.list(),
        lambda uc: str(getattr(uc, "name", "") or ""),
        lambda uc: dr.UseCase.delete(uc.id),
        prefix,
        older_than_hours,
        now,
        dry_run,
    )
    return report


def delete_run_resources_for_config(config: Config, prefix: str) -> TeardownReport:
    """Per-run teardown entry point: build a real client and delete by run prefix."""
    return delete_run_resources(_dr_module(config), prefix)


def sweep(
    config: Config,
    prefix: str = "drat-",
    older_than_hours: float = 24.0,
    dry_run: bool = True,
) -> TeardownReport:
    """Sweep leaked resources older than a threshold; prints a summary table."""
    report = delete_run_resources(
        _dr_module(config), prefix, older_than_hours=older_than_hours, dry_run=dry_run
    )

    verb = "would delete" if dry_run else "deleted"
    total = sum(len(v) for v in report.deleted.values())
    print(f"Sweep ({prefix!r}, older than {older_than_hours}h): {verb} {total} resource(s)")
    for kind, names in sorted(report.deleted.items()):
        for name in names:
            print(f"  {verb}: {kind[:-1]} {name}")
    for kind, errors in sorted(report.failed.items()):
        for error in errors:
            print(f"  FAILED: {kind[:-1]} {error}")
    if dry_run and total:
        print("Dry run — pass --execute to delete.")
    return report
