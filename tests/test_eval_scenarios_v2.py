"""Tests for behavioral (schema v2) scenario loading and kind discrimination."""

from pathlib import Path

import pytest

from dr_agents_tester.eval.models import BehavioralScenario, Difficulty
from dr_agents_tester.eval.scenarios import load_behavioral_scenarios, load_scenarios

REPO_ROOT = Path(__file__).resolve().parents[1]

PLAN_SCENARIO_YAML = """\
scenarios:
  - id: plan-1
    name: A plan scenario
    difficulty: easy
    prompt: Do a thing
    expected_files: [a.py]
    expected_approach: Do it
    expected_patterns: [pattern]
    common_pitfalls: [pitfall]
    acceptance_criteria: [works]
"""

BEHAVIORAL_SCENARIO_YAML = """\
scenarios:
  - id: golden-1
    kind: behavioral
    name: Golden journey
    difficulty: medium
    prompt: |
      Train a model on ./data/train.csv with target churn, then deploy it.
      Use the prefix {run_id} for resources.
    skills_under_test:
      - datarobot-model-training
      - datarobot-model-deployment
    fixtures:
      - fixtures/train.csv
      - source: fixtures/holdout.csv
        dest: data/holdout.csv
    env:
      automl_mode: quick
      resource_prefix: "{run_id}"
    success_checks:
      - type: dr_project_exists
        name_contains: "{run_id}"
      - type: file_exists
        path: predictions.csv
    rubric: Penalize hallucinated SDK methods.
    common_pitfalls:
      - Skipping the Use Case linkage
    timeout_minutes: 25
"""


class TestLoadBehavioralScenarios:
    def test_happy_path(self, tmp_path: Path) -> None:
        (tmp_path / "golden.yaml").write_text(BEHAVIORAL_SCENARIO_YAML)

        scenarios = load_behavioral_scenarios(tmp_path)

        assert len(scenarios) == 1
        s = scenarios[0]
        assert isinstance(s, BehavioralScenario)
        assert s.id == "golden-1"
        assert s.difficulty == Difficulty.MEDIUM
        assert s.skills_under_test == [
            "datarobot-model-training",
            "datarobot-model-deployment",
        ]
        assert [c.type for c in s.success_checks] == ["dr_project_exists", "file_exists"]
        assert s.success_checks[0].params == {"name_contains": "{run_id}"}
        assert s.fixtures[0].source == "fixtures/train.csv"
        assert s.fixtures[0].dest == "fixtures/train.csv"
        assert s.fixtures[1].source == "fixtures/holdout.csv"
        assert s.fixtures[1].dest == "data/holdout.csv"
        assert s.env == {"automl_mode": "quick", "resource_prefix": "{run_id}"}
        assert s.timeout_minutes == 25
        assert s.source_dir == tmp_path

    def test_defaults(self, tmp_path: Path) -> None:
        (tmp_path / "min.yaml").write_text(
            """\
scenarios:
  - id: min-1
    kind: behavioral
    name: Minimal
    difficulty: easy
    prompt: Do it
    skills_under_test: [datarobot-predictions]
    success_checks:
      - type: file_exists
        path: out.csv
"""
        )
        s = load_behavioral_scenarios(tmp_path)[0]
        assert s.fixtures == []
        assert s.env == {}
        assert s.rubric == ""
        assert s.common_pitfalls == []
        assert s.timeout_minutes == 30

    def test_difficulty_filter(self, tmp_path: Path) -> None:
        (tmp_path / "golden.yaml").write_text(BEHAVIORAL_SCENARIO_YAML)
        assert load_behavioral_scenarios(tmp_path, Difficulty.EASY) == []
        assert len(load_behavioral_scenarios(tmp_path, Difficulty.MEDIUM)) == 1


class TestKindDiscrimination:
    def test_mixed_file(self, tmp_path: Path) -> None:
        (tmp_path / "mixed.yaml").write_text(
            PLAN_SCENARIO_YAML + BEHAVIORAL_SCENARIO_YAML.replace("scenarios:\n", "")
        )

        plan = load_scenarios(tmp_path)
        behavioral = load_behavioral_scenarios(tmp_path)

        assert [s.id for s in plan] == ["plan-1"]
        assert [s.id for s in behavioral] == ["golden-1"]

    def test_explicit_plan_kind(self, tmp_path: Path) -> None:
        (tmp_path / "p.yaml").write_text(
            PLAN_SCENARIO_YAML.replace("    name:", "    kind: plan\n    name:")
        )
        assert [s.id for s in load_scenarios(tmp_path)] == ["plan-1"]

    def test_real_plan_scenarios_still_load(self) -> None:
        """The pre-v2 scenario corpus (no kind keys) loads identically."""
        scenarios_dir = REPO_ROOT / "eval_scenarios" / "datarobot-agent-application"
        scenarios = load_scenarios(scenarios_dir)
        assert len(scenarios) >= 10
        assert load_behavioral_scenarios(scenarios_dir) == []


class TestBehavioralValidation:
    def _write(self, tmp_path: Path, body: str) -> Path:
        (tmp_path / "bad.yaml").write_text(body)
        return tmp_path

    def test_missing_required_fields(self, tmp_path: Path) -> None:
        self._write(
            tmp_path,
            """\
scenarios:
  - id: bad-1
    kind: behavioral
    name: Bad
    difficulty: easy
    prompt: Do it
""",
        )
        with pytest.raises(ValueError, match="missing fields"):
            load_behavioral_scenarios(tmp_path)

    def test_empty_skills_under_test(self, tmp_path: Path) -> None:
        self._write(
            tmp_path,
            """\
scenarios:
  - id: bad-2
    kind: behavioral
    name: Bad
    difficulty: easy
    prompt: Do it
    skills_under_test: []
    success_checks: [{type: file_exists, path: x}]
""",
        )
        with pytest.raises(ValueError, match="non-empty list"):
            load_behavioral_scenarios(tmp_path)

    def test_unknown_check_type(self, tmp_path: Path) -> None:
        self._write(
            tmp_path,
            """\
scenarios:
  - id: bad-3
    kind: behavioral
    name: Bad
    difficulty: easy
    prompt: Do it
    skills_under_test: [some-skill]
    success_checks: [{type: dr_flying_cars}]
""",
        )
        with pytest.raises(ValueError, match="unknown check type"):
            load_behavioral_scenarios(tmp_path)

    def test_check_without_type(self, tmp_path: Path) -> None:
        self._write(
            tmp_path,
            """\
scenarios:
  - id: bad-4
    kind: behavioral
    name: Bad
    difficulty: easy
    prompt: Do it
    skills_under_test: [some-skill]
    success_checks: [{path: x}]
""",
        )
        with pytest.raises(ValueError, match="needs a 'type' key"):
            load_behavioral_scenarios(tmp_path)

    def test_plan_only_fields_rejected(self, tmp_path: Path) -> None:
        self._write(
            tmp_path,
            """\
scenarios:
  - id: bad-5
    kind: behavioral
    name: Bad
    difficulty: easy
    prompt: Do it
    skills_under_test: [some-skill]
    success_checks: [{type: file_exists, path: x}]
    expected_files: [a.py]
""",
        )
        with pytest.raises(ValueError, match="plan-eval-only"):
            load_behavioral_scenarios(tmp_path)

    def test_invalid_timeout(self, tmp_path: Path) -> None:
        self._write(
            tmp_path,
            """\
scenarios:
  - id: bad-6
    kind: behavioral
    name: Bad
    difficulty: easy
    prompt: Do it
    skills_under_test: [some-skill]
    success_checks: [{type: file_exists, path: x}]
    timeout_minutes: 0
""",
        )
        with pytest.raises(ValueError, match="timeout_minutes"):
            load_behavioral_scenarios(tmp_path)

    def test_invalid_fixture_entry(self, tmp_path: Path) -> None:
        self._write(
            tmp_path,
            """\
scenarios:
  - id: bad-7
    kind: behavioral
    name: Bad
    difficulty: easy
    prompt: Do it
    skills_under_test: [some-skill]
    success_checks: [{type: file_exists, path: x}]
    fixtures: [{dest: only-dest.csv}]
""",
        )
        with pytest.raises(ValueError, match="each fixture"):
            load_behavioral_scenarios(tmp_path)

    def test_invalid_difficulty(self, tmp_path: Path) -> None:
        self._write(
            tmp_path,
            """\
scenarios:
  - id: bad-8
    kind: behavioral
    name: Bad
    difficulty: impossible
    prompt: Do it
    skills_under_test: [some-skill]
    success_checks: [{type: file_exists, path: x}]
""",
        )
        with pytest.raises(ValueError, match="Invalid difficulty"):
            load_behavioral_scenarios(tmp_path)
