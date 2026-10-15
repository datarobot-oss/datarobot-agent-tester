"""Success checks that assert on local workspace state (no external services)."""

from __future__ import annotations

import re
import time

from ..models import CheckResult
from .base import CheckContext, OutcomeCheck, param_int

#: file_matches skips files larger than this by default — a runaway agent
#: artifact must not stall the check suite.
DEFAULT_MAX_BYTES = 10 * 1024 * 1024


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


class FileMatchesCheck(OutcomeCheck):
    """Assert a workspace file's content matches a regex.

    The pattern runs against raw text (``re.MULTILINE``), so files that are
    not clean structured data — e.g. CSVs with ``#`` comment headers — are
    asserted on directly without parsing.

    Params:
        path: workspace-relative path or glob pattern (absolute paths and
            ``..`` traversal are rejected). With a glob, the check passes when
            any matching file satisfies the pattern.
        pattern: Python regex the file content must match.
        ignorecase: case-insensitive matching (default false).
        min_count: minimum number of (non-overlapping) pattern occurrences in
            a single file (default 1).
        encoding: text encoding (default utf-8; undecodable bytes are
            replaced, never fatal).
        max_bytes: files larger than this are skipped with a note
            (default 10MB); if every candidate is oversized the check fails.
    """

    type_name = "file_matches"

    def run(self, ctx: CheckContext) -> CheckResult:
        start = time.monotonic()

        def done(passed: bool, evidence: str, error: str | None = None) -> CheckResult:
            return CheckResult(
                check_type=self.type_name,
                passed=passed,
                evidence=evidence,
                error=error,
                duration_seconds=round(time.monotonic() - start, 3),
            )

        raw_path = str(self.params.get("path", ""))
        raw_pattern = self.params.get("pattern")
        min_count = param_int(self.params, "min_count", 1)
        max_bytes = param_int(self.params, "max_bytes", DEFAULT_MAX_BYTES)
        encoding = str(self.params.get("encoding", "utf-8"))

        if not raw_path:
            return done(False, "", error="file_matches requires a 'path' param")
        if raw_path.startswith("/") or ".." in raw_path.split("/"):
            return done(False, "", error=f"path must be workspace-relative: {raw_path!r}")
        if not isinstance(raw_pattern, str) or not raw_pattern:
            return done(False, "", error="file_matches requires a non-empty 'pattern' param")

        flags = re.MULTILINE | (re.IGNORECASE if self.params.get("ignorecase") else 0)
        try:
            pattern = re.compile(raw_pattern, flags)
        except re.error as exc:
            return done(False, "", error=f"invalid regex {raw_pattern!r}: {exc}")

        files = [m for m in sorted(ctx.workspace.glob(raw_path)) if m.is_file()]
        if not files:
            return done(False, f"no file matching {raw_path!r} under {ctx.workspace}")

        counts: list[str] = []
        skipped: list[str] = []
        for f in files:
            rel = f.relative_to(ctx.workspace)
            size = f.stat().st_size
            if size > max_bytes:
                skipped.append(f"{rel} ({size}B > max_bytes={max_bytes})")
                continue
            text = f.read_text(encoding=encoding, errors="replace")
            n = len(pattern.findall(text))
            if n >= min_count:
                return done(True, f"{rel}: {n} match(es) of {raw_pattern!r}")
            counts.append(f"{rel}: {n}")

        if skipped and not counts:
            return done(
                False, "", error=f"all candidate files exceed max_bytes: {'; '.join(skipped)}"
            )
        evidence = (
            f"pattern {raw_pattern!r} needs >= {min_count} match(es); got {'; '.join(counts)}"
        )
        if skipped:
            evidence += f" (skipped oversized: {'; '.join(skipped)})"
        return done(False, evidence)
