"""Tests for the driver registry, protocol conformance, and FakeDriver."""

from datetime import timedelta
from pathlib import Path

import pytest

from dr_agents_tester.eval.drivers import (
    AgentDriver,
    AgentRunStatus,
    FakeDriver,
    RunPaths,
    get_driver,
    register_driver,
)
from dr_agents_tester.eval.trajectory import EventKind, TrajectoryEvent


class TestRegistry:
    def test_get_unknown_driver(self) -> None:
        with pytest.raises(ValueError, match="Unknown driver 'nope'"):
            get_driver("nope")

    def test_fake_driver_registered(self) -> None:
        driver = get_driver("fake")
        assert driver.name == "fake"

    def test_register_and_get(self) -> None:
        register_driver("fake-2", FakeDriver)
        assert get_driver("fake-2").name == "fake"


class TestProtocolConformance:
    def test_fake_driver_satisfies_protocol(self) -> None:
        driver: AgentDriver = FakeDriver()
        assert driver.capabilities


class TestRunPaths:
    def test_layout_and_from_workspace(self, tmp_path: Path) -> None:
        paths = RunPaths(run_dir=tmp_path / "run1").create()
        assert paths.workspace.is_dir()
        assert paths.home.is_dir()
        assert RunPaths.from_workspace(paths.workspace).run_dir == paths.run_dir
        assert paths.transcript.name == "transcript.raw.jsonl"


class TestFakeDriver:
    def test_run_writes_files_and_transcript(self, tmp_path: Path) -> None:
        paths = RunPaths(run_dir=tmp_path / "run1").create()
        driver = FakeDriver(
            events=[TrajectoryEvent(kind=EventKind.TURN_START, seq=0, turn=1)],
            files_to_create={"predictions.csv": "id,prediction\n1,0.5\n"},
        )

        raw = driver.run(
            paths.workspace, "do the thing", {"DATAROBOT_API_TOKEN": "x"}, timedelta(minutes=5)
        )

        assert raw.status == AgentRunStatus.COMPLETED
        assert (paths.workspace / "predictions.csv").is_file()
        assert paths.transcript.is_file()
        assert len(driver.calls) == 1
        assert driver.calls[0].prompt == "do the thing"
        assert driver.calls[0].env == {"DATAROBOT_API_TOKEN": "x"}
        assert driver.normalize(raw)[0].kind == EventKind.TURN_START

    def test_iter_events_lenient(self, tmp_path: Path) -> None:
        paths = RunPaths(run_dir=tmp_path / "run1").create()
        driver = FakeDriver(raw_lines=['{"type": "a"}', "garbage", '{"type": "b"}'])
        raw = driver.run(paths.workspace, "p", {}, timedelta(minutes=1))
        assert [e["type"] for e in raw.iter_events()] == ["a", "b"]
