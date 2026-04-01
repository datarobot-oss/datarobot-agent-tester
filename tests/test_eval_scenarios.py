"""Tests for eval/scenarios.py YAML loading and validation."""

from pathlib import Path

import pytest

from dr_agents_tester.eval.models import Difficulty
from dr_agents_tester.eval.scenarios import load_scenarios


@pytest.fixture
def scenarios_dir(tmp_path: Path) -> Path:
    """Create a temporary scenarios directory with test YAML files."""
    d = tmp_path / "scenarios"
    d.mkdir()

    easy = d / "easy.yaml"
    easy.write_text(
        """scenarios:
  - id: test-easy-1
    name: Test easy scenario
    difficulty: easy
    prompt: Do a simple task
    expected_files:
      - file1.py
      - file2.py
    expected_approach: Simple approach
    expected_patterns:
      - pattern1
    common_pitfalls:
      - pitfall1
    acceptance_criteria:
      - criterion1
  - id: test-easy-2
    name: Another easy scenario
    difficulty: easy
    prompt: Another simple task
    expected_files:
      - file3.py
    expected_approach: Another approach
    expected_patterns:
      - pattern2
    common_pitfalls:
      - pitfall2
    acceptance_criteria:
      - criterion2
"""
    )

    medium = d / "medium.yaml"
    medium.write_text(
        """scenarios:
  - id: test-medium-1
    name: Test medium scenario
    difficulty: medium
    prompt: Do a medium task
    expected_files:
      - file4.py
    expected_approach: Medium approach
    expected_patterns:
      - pattern3
    common_pitfalls:
      - pitfall3
    acceptance_criteria:
      - criterion3
"""
    )

    return d


class TestLoadScenarios:
    def test_loads_all_scenarios(self, scenarios_dir: Path) -> None:
        scenarios = load_scenarios(scenarios_dir)
        assert len(scenarios) == 3

    def test_filter_by_difficulty(self, scenarios_dir: Path) -> None:
        easy = load_scenarios(scenarios_dir, difficulty_filter=Difficulty.EASY)
        assert len(easy) == 2
        assert all(s.difficulty == Difficulty.EASY for s in easy)

    def test_filter_empty_result(self, scenarios_dir: Path) -> None:
        hard = load_scenarios(scenarios_dir, difficulty_filter=Difficulty.HARD)
        assert len(hard) == 0

    def test_scenario_fields(self, scenarios_dir: Path) -> None:
        scenarios = load_scenarios(scenarios_dir, difficulty_filter=Difficulty.EASY)
        s = scenarios[0]
        assert s.id == "test-easy-1"
        assert s.name == "Test easy scenario"
        assert s.difficulty == Difficulty.EASY
        assert s.prompt == "Do a simple task"
        assert s.expected_files == ["file1.py", "file2.py"]
        assert s.expected_approach == "Simple approach"

    def test_missing_directory(self) -> None:
        with pytest.raises(FileNotFoundError):
            load_scenarios(Path("/nonexistent/path"))

    def test_missing_required_field(self, tmp_path: Path) -> None:
        d = tmp_path / "bad_scenarios"
        d.mkdir()
        (d / "bad.yaml").write_text(
            """scenarios:
  - id: bad-1
    name: Bad scenario
    difficulty: easy
    prompt: Missing fields
"""
        )
        with pytest.raises(ValueError, match="missing fields"):
            load_scenarios(d)

    def test_invalid_difficulty(self, tmp_path: Path) -> None:
        d = tmp_path / "bad_diff"
        d.mkdir()
        (d / "bad.yaml").write_text(
            """scenarios:
  - id: bad-1
    name: Bad scenario
    difficulty: impossible
    prompt: Invalid difficulty
    expected_files: [a.py]
    expected_approach: approach
    expected_patterns: [p1]
    common_pitfalls: [pp1]
    acceptance_criteria: [c1]
"""
        )
        with pytest.raises(ValueError, match="Invalid difficulty"):
            load_scenarios(d)

    def test_skips_empty_yaml(self, tmp_path: Path) -> None:
        d = tmp_path / "empty"
        d.mkdir()
        (d / "empty.yaml").write_text("")
        scenarios = load_scenarios(d)
        assert len(scenarios) == 0

    def test_skips_repo_tree_yaml(self, tmp_path: Path) -> None:
        d = tmp_path / "with_tree"
        d.mkdir()
        (d / "repo_tree.yaml").write_text("not: a scenario file")
        (d / "easy.yaml").write_text(
            """scenarios:
  - id: test-1
    name: Test
    difficulty: easy
    prompt: Task
    expected_files: [a.py]
    expected_approach: approach
    expected_patterns: [p]
    common_pitfalls: [pp]
    acceptance_criteria: [c]
"""
        )
        scenarios = load_scenarios(d)
        assert len(scenarios) == 1
