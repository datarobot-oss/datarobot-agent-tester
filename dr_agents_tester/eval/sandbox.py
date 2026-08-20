"""Run isolation: per-run env allowlist, workspace preparation, process spawning.

v1 isolation is configuration hygiene, not containment: an isolated HOME/XDG
tree guarantees the developer's real agent config, skills, and credentials
never leak into a run (and the ``no_skill`` baseline stays clean), while the
env allowlist guarantees only the intended DataRobot credentials are visible.
A DockerSandbox for network-egress control is a v2 hardening item.
"""

from __future__ import annotations

import os
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import IO, Protocol

from ..config import Config
from .drivers.base import RunPaths
from .models import BehavioralScenario
from .templating import substitute_env_tokens


class Sandbox(Protocol):
    """Spawns the agent process; implementations decide the isolation level."""

    def spawn(
        self,
        cmd: list[str],
        cwd: Path,
        env: Mapping[str, str],
        stdout: IO[bytes],
        stderr: IO[bytes],
    ) -> subprocess.Popen[bytes]: ...


class LocalSandbox:
    """Plain subprocess in its own process group (so timeouts kill the whole tree)."""

    def spawn(
        self,
        cmd: list[str],
        cwd: Path,
        env: Mapping[str, str],
        stdout: IO[bytes],
        stderr: IO[bytes],
    ) -> subprocess.Popen[bytes]:
        return subprocess.Popen(
            cmd,
            cwd=cwd,
            env=dict(env),
            stdout=stdout,
            stderr=stderr,
            start_new_session=True,
        )


def build_agent_env(paths: RunPaths, config: Config, run_id: str) -> dict[str, str]:
    """Build the agent's environment from scratch (allowlist, never a copy).

    Only PATH crosses over from the host environment — the agent CLI and
    python must resolve. Everything stateful (HOME, XDG dirs, TMPDIR) points
    into the run directory; only the intended DataRobot credentials are
    exposed. The env var names (never values) may be recorded in artifacts.
    """
    home = paths.home
    return {
        "PATH": os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin"),
        "HOME": str(home),
        "XDG_CONFIG_HOME": str(home / ".config"),
        "XDG_DATA_HOME": str(home / ".local" / "share"),
        "XDG_CACHE_HOME": str(home / ".cache"),
        "XDG_STATE_HOME": str(home / ".local" / "state"),
        "TMPDIR": str(paths.tmp),
        "DATAROBOT_API_TOKEN": config.api_key,
        "DATAROBOT_ENDPOINT": config.endpoint,
        "DRAT_RUN_ID": run_id,
        "NO_COLOR": "1",
        "CI": "1",
        "TERM": "dumb",
    }


def copy_fixtures(scenario: BehavioralScenario, paths: RunPaths) -> list[str]:
    """Copy scenario fixtures into the workspace; returns the dest paths written.

    Fixture sources resolve against the scenario file's directory. Escaping
    dest paths are rejected — fixtures may only land inside the workspace.
    """
    import shutil

    written: list[str] = []
    for fixture in scenario.fixtures:
        if scenario.source_dir is None:
            raise ValueError(f"Scenario {scenario.id!r} has fixtures but no source_dir")
        src = (scenario.source_dir / fixture.source).resolve()
        if not src.is_file():
            raise FileNotFoundError(f"Scenario {scenario.id!r} fixture not found: {src}")

        if Path(fixture.dest).is_absolute() or ".." in Path(fixture.dest).parts:
            raise ValueError(f"Fixture dest must be workspace-relative: {fixture.dest!r}")
        dest = paths.workspace / fixture.dest
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dest)
        written.append(fixture.dest)
    return written


#: Appended to the scenario prompt when env.resource_prefix is set, so
#: scenario authors never hand-write run-id plumbing into realistic prompts.
RESOURCE_PREFIX_EPILOGUE = (
    "\n\nImportant: name every DataRobot resource you create (use cases, datasets, "
    "projects, deployments) so its name starts with the prefix {prefix!r}."
)


def render_prompt(
    scenario: BehavioralScenario,
    run_id: str,
    host_env: Mapping[str, str] | None = None,
) -> str:
    """Template {run_id}/{env:VAR} into the prompt and append the prefix epilogue.

    ``{env:VAR}`` values (pre-provisioned fixture-resource ids) come from the
    *host* environment — they are substituted into the prompt text rather than
    exposed through the sandbox env allowlist.
    """
    prompt = scenario.prompt.replace("{run_id}", run_id)
    prompt = substitute_env_tokens(prompt, host_env or {})
    prefix = scenario.env.get("resource_prefix", "").replace("{run_id}", run_id)
    if prefix:
        prompt += RESOURCE_PREFIX_EPILOGUE.format(prefix=prefix)
    return prompt
