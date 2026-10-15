"""OpenCodeDriver: run the plain `opencode` CLI headlessly, one subprocess per run.

One process per run (not serve+attach) so each run gets its own skills and
config, timeouts are a clean process-group kill, and a wedged agent can't
poison later runs. Stdout streams straight to disk so timeouts and crashes
preserve the partial transcript. The driver never raises for agent failures —
they come back as ``RawTranscript.status`` so outcome checks still run.
"""

from __future__ import annotations

import os
import signal
import subprocess
import time
from collections.abc import Mapping
from datetime import timedelta
from pathlib import Path

from ..sandbox import LocalSandbox, Sandbox
from ..trajectory import OpenCodeAdapter, TrajectoryEvent
from .base import AgentRunStatus, RawTranscript, RunPaths
from .opencode_config import DEFAULT_MODEL, write_run_config
from .versions import OPENCODE_PINNED_VERSION

_KILL_GRACE_SECONDS = 10.0
_VERSION_CHECK_TIMEOUT = 30.0


class OpenCodeDriver:
    """Drives `opencode run --format json` inside the run sandbox."""

    name = "opencode"

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        opencode_bin: str = "opencode",
        pinned_version: str = OPENCODE_PINNED_VERSION,
        sandbox: Sandbox | None = None,
        allow_version_drift: bool = False,
    ) -> None:
        self.model = model
        self.opencode_bin = opencode_bin
        self.pinned_version = pinned_version
        self.sandbox: Sandbox = sandbox if sandbox is not None else LocalSandbox()
        self.allow_version_drift = allow_version_drift
        self._adapter = OpenCodeAdapter()
        self._checked_version: str | None = None

    @property
    def version(self) -> str:
        return self._checked_version or self.pinned_version

    @property
    def capabilities(self) -> frozenset[str]:
        return OpenCodeAdapter.CAPABILITIES

    def run(
        self,
        workspace: Path,
        prompt: str,
        env: Mapping[str, str],
        timeout: timedelta,
    ) -> RawTranscript:
        paths = RunPaths.from_workspace(workspace)

        actual_version, version_error = self._check_version(env)
        if version_error is not None:
            paths.stderr_log.write_text(version_error + "\n")
            paths.transcript.touch()
            return self._transcript(
                paths, workspace, actual_version, AgentRunStatus.LAUNCH_FAILED, None, 0.0
            )

        endpoint = env.get("DATAROBOT_ENDPOINT", "https://app.datarobot.com/api/v2")
        write_run_config(workspace, endpoint, self.model)

        cmd = [self.opencode_bin, "run", "--format", "json", "--model", self.model, prompt]

        start = time.monotonic()
        status = AgentRunStatus.COMPLETED
        exit_code: int | None = None
        try:
            with paths.transcript.open("wb") as stdout, paths.stderr_log.open("wb") as stderr:
                proc = self.sandbox.spawn(cmd, cwd=workspace, env=env, stdout=stdout, stderr=stderr)
                try:
                    exit_code = proc.wait(timeout=timeout.total_seconds())
                except subprocess.TimeoutExpired:
                    self._kill_group(proc)
                    status = AgentRunStatus.TIMEOUT
                    exit_code = proc.returncode
        except OSError as exc:
            paths.stderr_log.write_text(f"failed to launch {self.opencode_bin!r}: {exc}\n")
            status = AgentRunStatus.LAUNCH_FAILED
        wall_seconds = round(time.monotonic() - start, 3)

        raw = self._transcript(paths, workspace, actual_version, status, exit_code, wall_seconds)
        if status == AgentRunStatus.COMPLETED:
            if exit_code != 0:
                raw.status = AgentRunStatus.NONZERO_EXIT
            elif next(raw.iter_events(), None) is None:
                raw.status = AgentRunStatus.EMPTY_STREAM
        return raw

    def normalize(self, raw: RawTranscript) -> list[TrajectoryEvent]:
        if not raw.transcript_path.is_file():
            return []
        return self._adapter.parse(raw.transcript_path.read_text().splitlines())

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _check_version(self, env: Mapping[str, str]) -> tuple[str, str | None]:
        """Return (actual_version, error). Error is None when the run may proceed."""
        if self._checked_version is None:
            try:
                result = subprocess.run(
                    [self.opencode_bin, "--version"],
                    capture_output=True,
                    text=True,
                    timeout=_VERSION_CHECK_TIMEOUT,
                    env=dict(env),
                )
                self._checked_version = result.stdout.strip() or result.stderr.strip()
            except (OSError, subprocess.TimeoutExpired) as exc:
                return "unknown", (
                    f"cannot execute {self.opencode_bin!r} ({exc}). "
                    f"Install it with: npm install -g opencode-ai@{self.pinned_version}"
                )

        actual = self._checked_version
        if actual != self.pinned_version and not self.allow_version_drift:
            return actual, (
                f"opencode version {actual!r} does not match pinned "
                f"{self.pinned_version!r}. Install the pinned version with: "
                f"npm install -g opencode-ai@{self.pinned_version} "
                f"(or pass allow_version_drift=True for local exploration)"
            )
        return actual, None

    def _kill_group(self, proc: subprocess.Popen[bytes]) -> None:
        """SIGTERM the process group, then SIGKILL after a grace period."""
        try:
            pgid = os.getpgid(proc.pid)
        except ProcessLookupError:
            return
        try:
            os.killpg(pgid, signal.SIGTERM)
            proc.wait(timeout=_KILL_GRACE_SECONDS)
        except subprocess.TimeoutExpired:
            os.killpg(pgid, signal.SIGKILL)
            proc.wait(timeout=_KILL_GRACE_SECONDS)
        except ProcessLookupError:
            pass

    def _transcript(
        self,
        paths: RunPaths,
        workspace: Path,
        version: str,
        status: AgentRunStatus,
        exit_code: int | None,
        wall_seconds: float,
    ) -> RawTranscript:
        return RawTranscript(
            driver_name=self.name,
            driver_version=version,
            model=self.model,
            status=status,
            exit_code=exit_code,
            wall_seconds=wall_seconds,
            transcript_path=paths.transcript,
            stderr_path=paths.stderr_log,
            workspace=workspace,
        )
