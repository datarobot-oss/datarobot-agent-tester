"""Tests for the ExecutionBackend seam: PlanBackend behavior and Evaluator delegation."""

import json
from pathlib import Path

import pytest

from dr_agents_tester.config import Config
from dr_agents_tester.eval.backends import PlanBackend, RunContext, make_run_id
from dr_agents_tester.eval.models import (
    AgentResponse,
    ConditionType,
    Difficulty,
    EvalCondition,
    EvalResult,
    EvaluationReport,
    OutcomeResult,
    Scenario,
    ScenarioBase,
    ScoreCard,
)
from dr_agents_tester.eval.runner import Evaluator
from dr_agents_tester.llm import LLMUsage

PLAN_SCENARIO = Scenario(
    id="s-1",
    name="Test",
    difficulty=Difficulty.EASY,
    prompt="Add a health endpoint",
    expected_files=["a.py"],
    expected_approach="Do it",
    expected_patterns=["p"],
    common_pitfalls=["c"],
    acceptance_criteria=["ok"],
)

SCORE_JSON = json.dumps(
    {
        "file_identification": 0.8,
        "approach_correctness": 0.7,
        "pattern_awareness": 0.6,
        "pitfall_avoidance": 0.5,
        "completeness": 0.9,
        "rationale": "r",
    }
)


@pytest.fixture
def fake_config() -> Config:
    return Config(
        api_key="test-token",
        endpoint="https://app.datarobot.com/api/v2",
        model="datarobot/agent-model",
        test_model="datarobot/judge-model",
    )


class TestPlanBackend:
    def test_plan_then_score_with_injected_llm(self, fake_config: Config) -> None:
        calls: list[tuple[str, str]] = []

        def fake_llm(prompt: str, model: str, config: Config) -> tuple[str, LLMUsage]:
            calls.append((prompt[:40], model))
            if len(calls) == 1:
                return "My plan", LLMUsage(prompt_tokens=10)
            return SCORE_JSON, LLMUsage(prompt_tokens=20)

        backend = PlanBackend(config=fake_config, repo_tree="a/\nb/", llm=fake_llm)
        result = backend.execute(
            PLAN_SCENARIO,
            EvalCondition(condition_type=ConditionType.NO_CONTEXT),
            RunContext(run_number=1, run_id="r1"),
        )

        # Plan call first (agent model), scoring call second (judge model)
        assert [model for _, model in calls] == ["datarobot/agent-model", "datarobot/judge-model"]
        assert result.agent_response.plan_text == "My plan"
        assert result.score_card.file_identification == 0.8
        # Behavioral-only fields stay unset for plan runs
        assert result.outcome is None
        assert result.trajectory is None
        assert result.transcript_path is None

    def test_rejects_behavioral_scenario(self, fake_config: Config) -> None:
        base = ScenarioBase(id="x", name="x", difficulty=Difficulty.EASY, prompt="p")
        backend = PlanBackend(config=fake_config, repo_tree="t")
        with pytest.raises(TypeError, match="PlanBackend expects plan scenarios"):
            backend.execute(
                base,
                EvalCondition(condition_type=ConditionType.NO_CONTEXT),
                RunContext(run_number=1, run_id="r1"),
            )


class _FakeBackend:
    """Canned-result backend that records execute() calls."""

    name = "fake"

    def __init__(self, scenarios: list[ScenarioBase], outcome_pass: bool | None = None) -> None:
        self._scenarios = scenarios
        self._outcome_pass = outcome_pass
        self.calls: list[tuple[str, ConditionType, int, str]] = []

    def load_scenarios(
        self, scenarios_dir: Path, difficulty_filter: Difficulty | None
    ) -> list[ScenarioBase]:
        return list(self._scenarios)

    def execute(
        self, scenario: ScenarioBase, condition: EvalCondition, ctx: RunContext
    ) -> EvalResult:
        self.calls.append((scenario.id, condition.condition_type, ctx.run_number, ctx.run_id))
        outcome = None
        if self._outcome_pass is not None:
            outcome = OutcomeResult(outcome_pass=self._outcome_pass)
        return EvalResult(
            scenario_id=scenario.id,
            condition=condition.condition_type,
            run_number=ctx.run_number,
            agent_response=AgentResponse(plan_text="", usage=LLMUsage()),
            score_card=ScoreCard(),
            scoring_usage=LLMUsage(),
            outcome=outcome,
            run_id=ctx.run_id,
        )


class TestEvaluatorDelegation:
    def test_cardinality_and_backend_name(self, fake_config: Config, tmp_path: Path) -> None:
        scenarios: list[ScenarioBase] = [
            ScenarioBase(id=f"s-{i}", name="n", difficulty=Difficulty.EASY, prompt="p")
            for i in range(2)
        ]
        backend = _FakeBackend(scenarios)
        conditions = [
            EvalCondition(condition_type=ConditionType.NO_SKILL),
            EvalCondition(condition_type=ConditionType.SKILL_PR),
        ]
        evaluator = Evaluator(
            config=fake_config,
            scenarios_dir=tmp_path,
            conditions=conditions,
            n_runs=3,
            backend=backend,
        )
        report = evaluator.run()

        assert len(report.results) == 2 * 2 * 3
        assert len(backend.calls) == 12
        assert report.backend_name == "fake"
        assert len(report.condition_stats) == 2
        assert len(report.pairwise_comparisons) == 1

    def test_outcome_progress_output(
        self, fake_config: Config, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        scenarios: list[ScenarioBase] = [
            ScenarioBase(id="s", name="n", difficulty=Difficulty.EASY, prompt="p")
        ]
        evaluator = Evaluator(
            config=fake_config,
            scenarios_dir=tmp_path,
            conditions=[EvalCondition(condition_type=ConditionType.NO_SKILL)],
            n_runs=1,
            backend=_FakeBackend(scenarios, outcome_pass=True),
        )
        evaluator.run()
        assert "outcome=PASS" in capsys.readouterr().out

    def test_unique_run_ids_per_run(self, fake_config: Config, tmp_path: Path) -> None:
        scenarios: list[ScenarioBase] = [
            ScenarioBase(id="s", name="n", difficulty=Difficulty.EASY, prompt="p")
        ]
        backend = _FakeBackend(scenarios)
        evaluator = Evaluator(
            config=fake_config,
            scenarios_dir=tmp_path,
            conditions=[EvalCondition(condition_type=ConditionType.NO_SKILL)],
            n_runs=3,
            backend=backend,
        )
        evaluator.run()
        run_ids = [c[3] for c in backend.calls]
        assert len(set(run_ids)) == 3

    def test_requires_repo_tree_without_backend(self, fake_config: Config, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="repo_tree_path is required"):
            Evaluator(config=fake_config, scenarios_dir=tmp_path)

    def test_empty_scenarios_report_keeps_backend_name(
        self, fake_config: Config, tmp_path: Path
    ) -> None:
        evaluator = Evaluator(
            config=fake_config,
            scenarios_dir=tmp_path,
            conditions=[EvalCondition(condition_type=ConditionType.NO_SKILL)],
            backend=_FakeBackend([]),
        )
        report = evaluator.run()
        assert report == EvaluationReport(backend_name="fake")


class TestMakeRunId:
    def _scenario(self) -> ScenarioBase:
        return ScenarioBase(
            id="Train-Deploy-Predict-Golden-Journey",
            name="n",
            difficulty=Difficulty.MEDIUM,
            prompt="p",
        )

    def test_format_and_charset(self) -> None:
        run_id = make_run_id(
            self._scenario(), EvalCondition(condition_type=ConditionType.SKILL_PR), 2
        )
        assert run_id.startswith("drat-")
        assert "-skill_pr-".replace("_", "-") in run_id or "skill-pr" in run_id
        assert run_id == run_id.lower()
        import re

        assert re.fullmatch(r"[a-z0-9-]+", run_id)

    def test_prefix_override(self) -> None:
        run_id = make_run_id(
            self._scenario(),
            EvalCondition(condition_type=ConditionType.NO_SKILL),
            1,
            prefix="drat-gh12345-1",
        )
        assert run_id.startswith("drat-gh12345-1-no-skill-r1-")
