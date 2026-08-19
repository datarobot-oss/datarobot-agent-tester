"""End-to-end AgentBackend tests with FakeDriver (no network, no agent binary)."""

import json
from pathlib import Path

import pytest

from dr_agents_tester.config import Config
from dr_agents_tester.eval.backends import AgentBackend, RunContext
from dr_agents_tester.eval.drivers import FakeDriver
from dr_agents_tester.eval.models import (
    BehavioralScenario,
    CheckSpec,
    ConditionType,
    Difficulty,
    EvalCondition,
    FixtureSpec,
    Scenario,
)
from dr_agents_tester.eval.trajectory import EventKind, TrajectoryEvent


@pytest.fixture
def fake_config() -> Config:
    return Config(api_key="test-token", endpoint="https://app.datarobot.com/api/v2")


@pytest.fixture
def scenario(tmp_path: Path) -> BehavioralScenario:
    fixtures_dir = tmp_path / "scenario-src"
    fixtures_dir.mkdir()
    (fixtures_dir / "train.csv").write_text("a,b\n1,2\n")
    return BehavioralScenario(
        id="golden",
        name="Golden",
        difficulty=Difficulty.MEDIUM,
        prompt="Train on ./data/train.csv. Prefix: {run_id}.",
        skills_under_test=["datarobot-model-training"],
        success_checks=[CheckSpec(type="file_exists", params={"path": "predictions.csv"})],
        fixtures=[FixtureSpec(source="train.csv", dest="data/train.csv")],
        env={"automl_mode": "quick", "resource_prefix": "{run_id}"},
        timeout_minutes=7,
        source_dir=fixtures_dir,
    )


@pytest.fixture
def skills_source(tmp_path: Path) -> Path:
    src = tmp_path / "skills"
    (src / "datarobot-model-training").mkdir(parents=True)
    (src / "datarobot-model-training" / "SKILL.md").write_text(
        "---\nname: datarobot-model-training\ndescription: Use when training.\n---\nTrain.\n"
    )
    return src


def _events() -> list[TrajectoryEvent]:
    return [
        TrajectoryEvent(kind=EventKind.TURN_START, seq=0, turn=1),
        TrajectoryEvent(
            kind=EventKind.SKILL_TRIGGERED, seq=1, turn=1, name="datarobot-model-training"
        ),
        TrajectoryEvent(kind=EventKind.TEXT, seq=2, turn=1, detail={"text": "All done!"}),
        TrajectoryEvent(kind=EventKind.TURN_FINISH, seq=3, turn=1),
    ]


class TestAgentBackendExecute:
    def test_full_run_wiring(
        self,
        fake_config: Config,
        scenario: BehavioralScenario,
        skills_source: Path,
        tmp_path: Path,
    ) -> None:
        driver = FakeDriver(
            events=_events(),
            files_to_create={"predictions.csv": "id,prediction\n1,0.4\n"},
        )
        backend = AgentBackend(config=fake_config, driver=driver, work_dir=tmp_path / "runs")
        condition = EvalCondition(
            condition_type=ConditionType.SKILL_PR, skills_source=skills_source
        )
        ctx = RunContext(run_number=1, run_id="drat-golden-skill-pr-r1-x1")

        result = backend.execute(scenario, condition, ctx)

        # Outcome from real check logic against files the driver wrote
        assert result.outcome is not None and result.outcome.outcome_pass
        # Trajectory metrics derived from driver events
        assert result.trajectory is not None
        assert result.trajectory.skill_triggered is True
        assert result.trajectory.skills_used == ["datarobot-model-training"]
        # Final assistant text lands in plan_text
        assert result.agent_response.plan_text == "All done!"
        assert result.run_id == ctx.run_id
        assert result.transcript_path is not None

        # Driver call contract
        call = driver.calls[0]
        assert call.timeout.total_seconds() == 7 * 60
        # {run_id} templated into the prompt + resource-prefix epilogue appended
        assert "Prefix: drat-golden-skill-pr-r1-x1." in call.prompt
        assert "starts with the prefix 'drat-golden-skill-pr-r1-x1'" in call.prompt
        # Env: allowlist + scenario env rendered
        assert call.env["DATAROBOT_API_TOKEN"] == "test-token"
        assert call.env["automl_mode"] == "quick"
        assert call.env["resource_prefix"] == "drat-golden-skill-pr-r1-x1"
        assert call.env["HOME"].endswith("/home")
        # Fixtures copied to dest
        assert (call.workspace / "data" / "train.csv").is_file()

    def test_skills_installed_only_for_skill_conditions(
        self,
        fake_config: Config,
        scenario: BehavioralScenario,
        skills_source: Path,
        tmp_path: Path,
    ) -> None:
        installs: list[Path] = []

        def recording_installer(source: Path, paths: object) -> list[object]:
            installs.append(source)
            return []

        backend = AgentBackend(
            config=fake_config,
            driver=FakeDriver(files_to_create={"predictions.csv": "p\n1\n"}),
            work_dir=tmp_path / "runs",
            skill_installer=recording_installer,  # type: ignore[arg-type]
        )

        backend.execute(
            scenario,
            EvalCondition(condition_type=ConditionType.NO_SKILL),
            RunContext(run_number=1, run_id="r-noskill"),
        )
        assert installs == []

        backend.execute(
            scenario,
            EvalCondition(condition_type=ConditionType.SKILL_PR, skills_source=skills_source),
            RunContext(run_number=1, run_id="r-skill"),
        )
        assert installs == [skills_source]

    def test_real_skill_install_lands_in_isolated_home(
        self,
        fake_config: Config,
        scenario: BehavioralScenario,
        skills_source: Path,
        tmp_path: Path,
    ) -> None:
        backend = AgentBackend(
            config=fake_config,
            driver=FakeDriver(files_to_create={"predictions.csv": "p\n1\n"}),
            work_dir=tmp_path / "runs",
        )
        ctx = RunContext(run_number=1, run_id="r-install")
        backend.execute(
            scenario,
            EvalCondition(condition_type=ConditionType.SKILL_PR, skills_source=skills_source),
            RunContext(run_number=1, run_id=ctx.run_id),
        )
        installed = (
            tmp_path
            / "runs"
            / ctx.run_id
            / "home"
            / ".config"
            / "opencode"
            / "skills"
            / "datarobot-model-training"
            / "SKILL.md"
        )
        assert installed.is_file()

    def test_teardown_called_per_run(
        self, fake_config: Config, scenario: BehavioralScenario, tmp_path: Path
    ) -> None:
        torn_down: list[str] = []
        backend = AgentBackend(
            config=fake_config,
            driver=FakeDriver(),
            work_dir=tmp_path / "runs",
            teardown=lambda cfg, prefix: torn_down.append(prefix),
        )
        backend.execute(
            scenario,
            EvalCondition(condition_type=ConditionType.NO_SKILL),
            RunContext(run_number=1, run_id="r-teardown"),
        )
        assert torn_down == ["r-teardown"]

    def test_keep_resources_skips_teardown(
        self, fake_config: Config, scenario: BehavioralScenario, tmp_path: Path
    ) -> None:
        torn_down: list[str] = []
        backend = AgentBackend(
            config=fake_config,
            driver=FakeDriver(),
            work_dir=tmp_path / "runs",
            teardown=lambda cfg, prefix: torn_down.append(prefix),
            keep_resources=True,
        )
        backend.execute(
            scenario,
            EvalCondition(condition_type=ConditionType.NO_SKILL),
            RunContext(run_number=1, run_id="r-keep"),
        )
        assert torn_down == []

    def test_failed_check_yields_fail_outcome(
        self, fake_config: Config, scenario: BehavioralScenario, tmp_path: Path
    ) -> None:
        backend = AgentBackend(
            config=fake_config,
            driver=FakeDriver(),  # writes no predictions.csv
            work_dir=tmp_path / "runs",
        )
        result = backend.execute(
            scenario,
            EvalCondition(condition_type=ConditionType.NO_SKILL),
            RunContext(run_number=1, run_id="r-fail"),
        )
        assert result.outcome is not None and not result.outcome.outcome_pass

    def test_artifacts_written(
        self, fake_config: Config, scenario: BehavioralScenario, tmp_path: Path
    ) -> None:
        backend = AgentBackend(
            config=fake_config,
            driver=FakeDriver(files_to_create={"predictions.csv": "p\n1\n"}),
            work_dir=tmp_path / "runs",
        )
        ctx = RunContext(run_number=2, run_id="r-artifacts")
        backend.execute(
            scenario, EvalCondition(condition_type=ConditionType.NO_SKILL), ctx
        )
        run_dir = tmp_path / "runs" / ctx.run_id
        meta = json.loads((run_dir / "meta.json").read_text())
        assert meta["run_id"] == ctx.run_id
        assert meta["scenario_id"] == "golden"
        assert meta["run_number"] == 2
        assert (run_dir / "trajectory.json").is_file()
        checks = json.loads((run_dir / "checks.json").read_text())
        assert checks["outcome_pass"] is True

    def test_rejects_plan_scenario(self, fake_config: Config, tmp_path: Path) -> None:
        plan = Scenario(
            id="p",
            name="p",
            difficulty=Difficulty.EASY,
            prompt="x",
            expected_files=[],
            expected_approach="",
            expected_patterns=[],
            common_pitfalls=[],
            acceptance_criteria=[],
        )
        backend = AgentBackend(
            config=fake_config, driver=FakeDriver(), work_dir=tmp_path / "runs"
        )
        with pytest.raises(TypeError, match="behavioral scenarios"):
            backend.execute(
                plan,
                EvalCondition(condition_type=ConditionType.NO_SKILL),
                RunContext(run_number=1, run_id="r"),
            )
