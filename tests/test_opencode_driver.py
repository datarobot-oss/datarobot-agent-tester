"""Subprocess-level OpenCodeDriver tests using the fake_opencode stub binary."""

import json
import time
from datetime import timedelta
from pathlib import Path

import pytest

from dr_agents_tester.eval.drivers import AgentRunStatus, RunPaths
from dr_agents_tester.eval.drivers.opencode import OpenCodeDriver
from dr_agents_tester.eval.drivers.opencode_config import (
    build_opencode_config,
    gateway_base_url,
    split_model,
)
from dr_agents_tester.eval.trajectory import EventKind

FIXTURES = Path(__file__).parent / "fixtures"
FAKE_BIN = FIXTURES / "bin" / "fake_opencode"


@pytest.fixture(autouse=True)
def _executable_stub() -> None:
    FAKE_BIN.chmod(0o755)


def _driver(**kwargs: object) -> OpenCodeDriver:
    return OpenCodeDriver(opencode_bin=str(FAKE_BIN), **kwargs)  # type: ignore[arg-type]


def _base_env(paths: RunPaths, **extra: str) -> dict[str, str]:
    import os

    env = {
        "PATH": os.environ["PATH"],
        "HOME": str(paths.home),
        "TMPDIR": str(paths.tmp),
        "DATAROBOT_API_TOKEN": "test-token",
        "DATAROBOT_ENDPOINT": "https://app.datarobot.com/api/v2",
    }
    env.update(extra)
    return env


class TestRun:
    def test_completed_run_end_to_end(self, tmp_path: Path) -> None:
        paths = RunPaths(run_dir=tmp_path / "run").create()
        driver = _driver()
        env = _base_env(
            paths, FAKE_OPENCODE_FIXTURE=str(FIXTURES / "opencode" / "skill_trigger.jsonl")
        )

        raw = driver.run(paths.workspace, "trigger the probe", env, timedelta(minutes=1))

        assert raw.status == AgentRunStatus.COMPLETED
        assert raw.exit_code == 0
        assert raw.driver_version == "1.17.11"
        assert raw.wall_seconds > 0

        events = driver.normalize(raw)
        triggered = [e for e in events if e.kind == EventKind.SKILL_TRIGGERED]
        assert triggered and triggered[0].name == "probe-skill"

    def test_command_assembly_and_cwd(self, tmp_path: Path) -> None:
        paths = RunPaths(run_dir=tmp_path / "run").create()
        dump = tmp_path / "args.json"
        driver = _driver(model="datarobot/anthropic/claude-sonnet-4-6")
        env = _base_env(
            paths,
            FAKE_OPENCODE_DUMP_ARGS=str(dump),
            FAKE_OPENCODE_FIXTURE=str(FIXTURES / "opencode" / "simple_text.jsonl"),
        )

        driver.run(paths.workspace, "the prompt", env, timedelta(minutes=1))

        recorded = json.loads(dump.read_text())
        assert recorded["argv"] == [
            "run",
            "--format",
            "json",
            "--model",
            "datarobot/anthropic/claude-sonnet-4-6",
            "the prompt",
        ]
        assert recorded["cwd"] == str(paths.workspace)

    def test_opencode_config_written_without_token_material(self, tmp_path: Path) -> None:
        paths = RunPaths(run_dir=tmp_path / "run").create()
        driver = _driver()
        env = _base_env(
            paths,
            DATAROBOT_API_TOKEN="super-secret-token",
            FAKE_OPENCODE_FIXTURE=str(FIXTURES / "opencode" / "simple_text.jsonl"),
        )

        driver.run(paths.workspace, "p", env, timedelta(minutes=1))

        config_text = (paths.workspace / "opencode.json").read_text()
        assert "super-secret-token" not in config_text
        config = json.loads(config_text)
        provider = config["provider"]["datarobot"]
        assert provider["options"]["apiKey"] == "{env:DATAROBOT_API_TOKEN}"
        assert provider["options"]["baseURL"] == "https://app.datarobot.com/api/v2/genai/llmgw"
        assert "anthropic/claude-sonnet-4-6" in provider["models"]

    def test_nonzero_exit(self, tmp_path: Path) -> None:
        paths = RunPaths(run_dir=tmp_path / "run").create()
        env = _base_env(
            paths,
            FAKE_OPENCODE_EXIT="2",
            FAKE_OPENCODE_FIXTURE=str(FIXTURES / "opencode" / "simple_text.jsonl"),
        )
        raw = _driver().run(paths.workspace, "p", env, timedelta(minutes=1))
        assert raw.status == AgentRunStatus.NONZERO_EXIT
        assert raw.exit_code == 2
        # Partial transcript preserved
        assert raw.transcript_path.stat().st_size > 0

    def test_empty_stream(self, tmp_path: Path) -> None:
        paths = RunPaths(run_dir=tmp_path / "run").create()
        raw = _driver().run(paths.workspace, "p", _base_env(paths), timedelta(minutes=1))
        assert raw.status == AgentRunStatus.EMPTY_STREAM
        assert raw.exit_code == 0

    def test_timeout_kills_process_group(self, tmp_path: Path) -> None:
        paths = RunPaths(run_dir=tmp_path / "run").create()
        env = _base_env(paths, FAKE_OPENCODE_SLEEP="30")

        start = time.monotonic()
        raw = _driver().run(paths.workspace, "p", env, timedelta(seconds=1))
        elapsed = time.monotonic() - start

        assert raw.status == AgentRunStatus.TIMEOUT
        assert elapsed < 15  # killed promptly, not after the 30s sleep

    def test_version_mismatch_is_launch_failed(self, tmp_path: Path) -> None:
        paths = RunPaths(run_dir=tmp_path / "run").create()
        env = _base_env(paths, FAKE_OPENCODE_VERSION="9.9.9")
        raw = _driver().run(paths.workspace, "p", env, timedelta(minutes=1))
        assert raw.status == AgentRunStatus.LAUNCH_FAILED
        assert raw.driver_version == "9.9.9"
        assert "npm install -g opencode-ai@1.17.11" in (raw.stderr_path or Path()).read_text()

    def test_version_drift_allowed_when_overridden(self, tmp_path: Path) -> None:
        paths = RunPaths(run_dir=tmp_path / "run").create()
        env = _base_env(
            paths,
            FAKE_OPENCODE_VERSION="9.9.9",
            FAKE_OPENCODE_FIXTURE=str(FIXTURES / "opencode" / "simple_text.jsonl"),
        )
        raw = _driver(allow_version_drift=True).run(
            paths.workspace, "p", env, timedelta(minutes=1)
        )
        assert raw.status == AgentRunStatus.COMPLETED
        assert raw.driver_version == "9.9.9"

    def test_missing_binary_is_launch_failed(self, tmp_path: Path) -> None:
        paths = RunPaths(run_dir=tmp_path / "run").create()
        driver = OpenCodeDriver(opencode_bin=str(tmp_path / "nope-binary"))
        raw = driver.run(paths.workspace, "p", _base_env(paths), timedelta(minutes=1))
        assert raw.status == AgentRunStatus.LAUNCH_FAILED

    def test_env_passed_verbatim(self, tmp_path: Path) -> None:
        paths = RunPaths(run_dir=tmp_path / "run").create()
        dump = tmp_path / "env.json"
        env = _base_env(
            paths,
            FAKE_OPENCODE_DUMP_ENV=str(dump),
            FAKE_OPENCODE_FIXTURE=str(FIXTURES / "opencode" / "simple_text.jsonl"),
        )
        _driver().run(paths.workspace, "p", env, timedelta(minutes=1))
        child_env = set(json.loads(dump.read_text()))
        assert set(env) <= child_env
        # No ambient env beyond what the harness passed (allowing interpreter extras)
        assert "DATAROBOT_API_TOKEN" in child_env


class TestConfigHelpers:
    def test_gateway_base_url(self) -> None:
        assert (
            gateway_base_url("https://app.datarobot.com/api/v2/")
            == "https://app.datarobot.com/api/v2/genai/llmgw"
        )

    def test_split_model(self) -> None:
        assert split_model("datarobot/anthropic/claude-sonnet-4-6") == (
            "datarobot",
            "anthropic/claude-sonnet-4-6",
        )
        with pytest.raises(ValueError, match="provider"):
            split_model("just-a-name")

    def test_build_config_limits(self) -> None:
        config = build_opencode_config("https://x.example/api/v2", context_limit=100, output_limit=5)
        models = config["provider"]["datarobot"]["models"]  # type: ignore[index]
        assert models["anthropic/claude-sonnet-4-6"]["limit"] == {"context": 100, "output": 5}
