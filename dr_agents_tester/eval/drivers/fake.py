"""FakeDriver: fixture-replay driver for offline backend and harness tests.

Ships in the package (not test-only) because it is the contract object other
components' tests build on — the same role FakeDRClient plays for checks.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path

from ..trajectory import ALL_CAPABILITIES, TrajectoryEvent
from .base import AgentRunStatus, RawTranscript, RunPaths


@dataclass
class FakeDriverCall:
    workspace: Path
    prompt: str
    env: dict[str, str]
    timeout: timedelta


@dataclass
class FakeDriver:
    """Replays canned events and writes canned files instead of running an agent.

    Args:
        events: normalized events ``normalize()`` returns.
        files_to_create: workspace-relative path → content written by ``run()``
            (lets file_exists checks exercise real logic).
        raw_lines: raw JSONL lines written to the transcript artifact.
        status / exit_code / wall_seconds: canned RawTranscript fields.
    """

    events: list[TrajectoryEvent] = field(default_factory=list)
    files_to_create: dict[str, str] = field(default_factory=dict)
    raw_lines: list[str] = field(default_factory=list)
    status: AgentRunStatus = AgentRunStatus.COMPLETED
    exit_code: int | None = 0
    wall_seconds: float = 1.0

    name: str = "fake"
    version: str = "0.0.0"

    calls: list[FakeDriverCall] = field(default_factory=list)

    @property
    def capabilities(self) -> frozenset[str]:
        return ALL_CAPABILITIES

    def run(
        self,
        workspace: Path,
        prompt: str,
        env: Mapping[str, str],
        timeout: timedelta,
    ) -> RawTranscript:
        self.calls.append(
            FakeDriverCall(workspace=workspace, prompt=prompt, env=dict(env), timeout=timeout)
        )

        for rel_path, content in self.files_to_create.items():
            target = workspace / rel_path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content)

        paths = RunPaths.from_workspace(workspace)
        lines = self.raw_lines or [json.dumps({"type": "fake", "seq": i}) for i in range(1)]
        paths.transcript.parent.mkdir(parents=True, exist_ok=True)
        paths.transcript.write_text("\n".join(lines) + "\n")

        return RawTranscript(
            driver_name=self.name,
            driver_version=self.version,
            model="fake/model",
            status=self.status,
            exit_code=self.exit_code,
            wall_seconds=self.wall_seconds,
            transcript_path=paths.transcript,
            stderr_path=None,
            workspace=workspace,
        )

    def normalize(self, raw: RawTranscript) -> list[TrajectoryEvent]:
        return list(self.events)
