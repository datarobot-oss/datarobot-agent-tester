"""Tests for eval/runner.py Evaluator with mocked LLM."""

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from dr_agents_tester.config import Config
from dr_agents_tester.eval.models import ConditionType, EvalCondition
from dr_agents_tester.eval.runner import Evaluator
from dr_agents_tester.llm import LLMUsage


@pytest.fixture
def fake_config() -> Config:
    return Config(
        api_key="test-token",
        endpoint="https://app.datarobot.com/api/v2",
        model="datarobot/test-model",
        test_model="datarobot/test-model",
    )


@pytest.fixture
def eval_scenarios_dir(tmp_path: Path) -> Path:
    d = tmp_path / "scenarios"
    d.mkdir()
    (d / "easy.yaml").write_text(
        """scenarios:
  - id: test-easy-1
    name: Test scenario
    difficulty: easy
    prompt: Add a health endpoint
    expected_files:
      - backend/routes/health.py
    expected_approach: Create a new route
    expected_patterns:
      - FastAPI router
    common_pitfalls:
      - Adding auth
    acceptance_criteria:
      - Returns 200
"""
    )
    return d


@pytest.fixture
def repo_tree_path(tmp_path: Path) -> Path:
    tree_file = tmp_path / "repo_tree.txt"
    tree_file.write_text("backend/\n  routes/\n  main.py\n")
    return tree_file


def _mock_score_response() -> str:
    return json.dumps(
        {
            "file_identification": 0.8,
            "approach_correctness": 0.7,
            "pattern_awareness": 0.6,
            "pitfall_avoidance": 0.5,
            "completeness": 0.9,
            "rationale": "Test rationale",
        }
    )


class TestEvaluator:
    @patch("dr_agents_tester.eval.backends.plan.call_llm_with_usage")
    def test_run_single_condition(
        self,
        mock_llm: object,
        fake_config: Config,
        eval_scenarios_dir: Path,
        repo_tree_path: Path,
    ) -> None:
        mock_llm.side_effect = [  # type: ignore[union-attr]
            ("Here is my plan for the health endpoint...", LLMUsage(prompt_tokens=100)),
            (_mock_score_response(), LLMUsage(prompt_tokens=200)),
        ]

        evaluator = Evaluator(
            config=fake_config,
            scenarios_dir=eval_scenarios_dir,
            repo_tree_path=repo_tree_path,
            conditions=[EvalCondition(condition_type=ConditionType.NO_CONTEXT)],
            n_runs=1,
        )
        report = evaluator.run()

        assert len(report.results) == 1
        assert report.results[0].scenario_id == "test-easy-1"
        assert report.results[0].condition == ConditionType.NO_CONTEXT
        assert report.results[0].score_card.file_identification == 0.8
        assert report.n_scenarios == 1
        assert report.n_runs == 1

    @patch("dr_agents_tester.eval.backends.plan.call_llm_with_usage")
    def test_run_multiple_conditions(
        self,
        mock_llm: object,
        fake_config: Config,
        eval_scenarios_dir: Path,
        repo_tree_path: Path,
    ) -> None:
        mock_llm.return_value = (  # type: ignore[union-attr]
            _mock_score_response(),
            LLMUsage(prompt_tokens=100),
        )

        conditions = [
            EvalCondition(condition_type=ConditionType.NO_CONTEXT),
            EvalCondition(
                condition_type=ConditionType.GENERATED,
                agents_md_content="# Generated AGENTS.md",
            ),
        ]

        evaluator = Evaluator(
            config=fake_config,
            scenarios_dir=eval_scenarios_dir,
            repo_tree_path=repo_tree_path,
            conditions=conditions,
            n_runs=2,
        )
        report = evaluator.run()

        # 1 scenario x 2 conditions x 2 runs = 4 results
        assert len(report.results) == 4
        assert len(report.condition_stats) == 2
        assert len(report.pairwise_comparisons) == 1

    @patch("dr_agents_tester.eval.backends.plan.call_llm_with_usage")
    def test_save_report(
        self,
        mock_llm: object,
        fake_config: Config,
        eval_scenarios_dir: Path,
        repo_tree_path: Path,
        tmp_path: Path,
    ) -> None:
        mock_llm.return_value = (  # type: ignore[union-attr]
            _mock_score_response(),
            LLMUsage(),
        )

        evaluator = Evaluator(
            config=fake_config,
            scenarios_dir=eval_scenarios_dir,
            repo_tree_path=repo_tree_path,
            conditions=[EvalCondition(condition_type=ConditionType.NO_CONTEXT)],
            n_runs=1,
        )
        report = evaluator.run()

        output_dir = tmp_path / "results"
        md_path, json_path = evaluator.save_report(report, output_dir)

        assert md_path.exists()
        assert json_path.exists()
        assert "Evaluation Report" in md_path.read_text()

        # Verify JSON is valid
        data = json.loads(json_path.read_text())
        assert data["n_scenarios"] == 1

    @patch("dr_agents_tester.eval.backends.plan.call_llm_with_usage")
    def test_empty_scenarios(
        self,
        mock_llm: object,
        fake_config: Config,
        tmp_path: Path,
    ) -> None:
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        tree_path = tmp_path / "tree.txt"
        tree_path.write_text("empty/\n")

        evaluator = Evaluator(
            config=fake_config,
            scenarios_dir=empty_dir,
            repo_tree_path=tree_path,
            conditions=[EvalCondition(condition_type=ConditionType.NO_CONTEXT)],
        )
        report = evaluator.run()

        assert len(report.results) == 0
        mock_llm.assert_not_called()  # type: ignore[union-attr]


_BEHAVIORAL_TEMPLATE = """\
scenarios:
  - id: {sid}
    kind: behavioral
    name: Multi-dir scenario
    difficulty: easy
    prompt: Do it
    skills_under_test: [datarobot-predictions]
    success_checks: [{{type: file_exists, path: out.csv}}]
"""


class TestEvaluatorMultiDir:
    def _behavioral_evaluator(
        self, fake_config: Config, dirs: list[Path], work_dir: Path
    ) -> Evaluator:
        from dr_agents_tester.eval.backends import AgentBackend
        from dr_agents_tester.eval.drivers import FakeDriver

        backend = AgentBackend(
            config=fake_config,
            driver=FakeDriver(files_to_create={"out.csv": "data\n"}),
            work_dir=work_dir,
        )
        return Evaluator(
            config=fake_config,
            scenarios_dir=dirs,
            conditions=[EvalCondition(condition_type=ConditionType.NO_SKILL)],
            n_runs=1,
            backend=backend,
        )

    def test_scenarios_from_multiple_dirs_aggregate(
        self, fake_config: Config, tmp_path: Path
    ) -> None:
        dirs = []
        for i, name in enumerate(("journeys", "per-skill")):
            d = tmp_path / name
            d.mkdir()
            (d / "s.yaml").write_text(_BEHAVIORAL_TEMPLATE.format(sid=f"scenario-{i}"))
            dirs.append(d)

        report = self._behavioral_evaluator(fake_config, dirs, tmp_path / "runs").run()

        assert sorted(r.scenario_id for r in report.results) == ["scenario-0", "scenario-1"]
        assert report.n_scenarios == 2

    def test_duplicate_ids_across_dirs_rejected(self, fake_config: Config, tmp_path: Path) -> None:
        dirs = []
        for name in ("a", "b"):
            d = tmp_path / name
            d.mkdir()
            (d / "s.yaml").write_text(_BEHAVIORAL_TEMPLATE.format(sid="same-id"))
            dirs.append(d)

        evaluator = self._behavioral_evaluator(fake_config, dirs, tmp_path / "runs")
        with pytest.raises(ValueError, match="Duplicate scenario id"):
            evaluator.run()
