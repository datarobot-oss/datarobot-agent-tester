"""Success checks that assert on local workspace state (no external services)."""

from __future__ import annotations

import time

from ..models import CheckResult
from .base import CheckContext, OutcomeCheck, param_int


class FileExistsCheck(OutcomeCheck):
    """Assert a file exists in the workspace (glob allowed) with a minimum size.

    Params:
        path: workspace-relative path or glob pattern (absolute paths and
            ``..`` traversal are rejected).
        min_bytes: minimum file size in bytes (default 1 — empty files fail).
    """

    type_name = "file_exists"

    def run(self, ctx: CheckContext) -> CheckResult:
        start = time.monotonic()
        raw_path = str(self.params.get("path", ""))
        min_bytes = param_int(self.params, "min_bytes", 1)

        def done(passed: bool, evidence: str, error: str | None = None) -> CheckResult:
            return CheckResult(
                check_type=self.type_name,
                passed=passed,
                evidence=evidence,
                error=error,
                duration_seconds=round(time.monotonic() - start, 3),
            )

        if not raw_path:
            return done(False, "", error="file_exists requires a 'path' param")
        if raw_path.startswith("/") or ".." in raw_path.split("/"):
            return done(False, "", error=f"path must be workspace-relative: {raw_path!r}")

        matches = sorted(ctx.workspace.glob(raw_path))
        files = [m for m in matches if m.is_file()]
        if not files:
            return done(False, f"no file matching {raw_path!r} under {ctx.workspace}")

        big_enough = [f for f in files if f.stat().st_size >= min_bytes]
        if not big_enough:
            sizes = ", ".join(f"{f.name}={f.stat().st_size}B" for f in files)
            return done(False, f"matched {len(files)} file(s) but all under {min_bytes}B: {sizes}")

        first = big_enough[0]
        return done(True, f"{first.relative_to(ctx.workspace)} ({first.stat().st_size} bytes)")
