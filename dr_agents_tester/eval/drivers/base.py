"""AgentDriver protocol and the raw-transcript / run-layout types drivers share."""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from datetime import timedelta
from enum import Enum
from pathlib import Path
from typing import Protocol

from ..trajectory import TrajectoryEvent


class AgentRunStatus(str, Enum):
    """How an agent run ended, from the harness's perspective."""

    COMPLETED = "completed"  # exit 0 with at least one parseable event
    NONZERO_EXIT = "nonzero_exit"  # agent CLI exited non-zero (partial transcript kept)
    TIMEOUT = "timeout"  # killed by the harness (partial transcript kept)
    EMPTY_STREAM = "empty_stream"  # exited but produced no parseable events
    LAUNCH_FAILED = "launch_failed"  # binary missing / version mismatch / spawn error


@dataclass
class RunPaths:
    """Canonical per-run artifact layout.

    ``run_dir/`` holds everything one behavioral run produces: the agent's
    ``workspace/``, an isolated ``home/`` (HOME/XDG target so ambient agent
    config and skills can never leak in), the raw transcript, and the
    normalized/derived JSON artifacts.
    """

    run_dir: Path

    @property
    def workspace(self) -> Path:
        return self.run_dir / "workspace"

    @property
    def home(self) -> Path:
        return self.run_dir / "home"

    @property
    def tmp(self) -> Path:
        return self.run_dir / "tmp"

    @property
    def transcript(self) -> Path:
        return self.run_dir / "transcript.raw.jsonl"

    @property
    def stderr_log(self) -> Path:
        return self.run_dir / "stderr.log"

    @property
    def meta(self) -> Path:
        return self.run_dir / "meta.json"

    @property
    def trajectory(self) -> Path:
        return self.run_dir / "trajectory.json"

    @property
    def checks(self) -> Path:
        return self.run_dir / "checks.json"

    @classmethod
    def from_workspace(cls, workspace: Path) -> RunPaths:
        return cls(run_dir=workspace.parent)

    def create(self) -> RunPaths:
        for d in (self.workspace, self.home, self.tmp):
            d.mkdir(parents=True, exist_ok=True)
        return self


@dataclass
class RawTranscript:
    """The raw output of one agent run, before normalization."""

    driver_name: str
    driver_version: str
    model: str
    status: AgentRunStatus
    exit_code: int | None
    wall_seconds: float
    transcript_path: Path
    stderr_path: Path | None
    workspace: Path

    def iter_events(self) -> Iterator[dict[str, object]]:
        """Lenient JSONL reader: skips unparseable lines instead of raising."""
        if not self.transcript_path.is_file():
            return
        with self.transcript_path.open() as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(event, dict):
                    yield event


class AgentDriver(Protocol):
    """Drives one coding agent CLI headlessly (design doc §4.2).

    ``run`` must never raise for *agent* failures (crash, timeout, garbage
    output) — those come back as ``RawTranscript.status`` so outcome checks
    still run; an agent can fail after already having produced real state.
    """

    @property
    def name(self) -> str: ...

    @property
    def version(self) -> str: ...

    def run(
        self,
        workspace: Path,
        prompt: str,
        env: Mapping[str, str],
        timeout: timedelta,
    ) -> RawTranscript: ...

    def normalize(self, raw: RawTranscript) -> list[TrajectoryEvent]: ...

    @property
    def capabilities(self) -> frozenset[str]: ...
