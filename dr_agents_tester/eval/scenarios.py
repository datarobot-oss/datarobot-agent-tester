"""Load and validate evaluation scenarios from YAML files."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .models import Difficulty, Scenario


def load_scenarios(
    scenarios_dir: Path,
    difficulty_filter: Difficulty | None = None,
) -> list[Scenario]:
    """Load scenarios from YAML files in a directory.

    Each YAML file should contain a list of scenario dicts under a top-level
    ``scenarios`` key.  Files are expected to be named ``<difficulty>.yaml``.

    Args:
        scenarios_dir: Directory containing scenario YAML files.
        difficulty_filter: If set, only load scenarios with this difficulty.

    Returns:
        List of validated Scenario objects.

    Raises:
        FileNotFoundError: If the directory doesn't exist.
        ValueError: If a scenario is missing required fields.
    """
    try:
        import yaml  # type: ignore[import-untyped]
    except ImportError:
        raise ImportError(
            "pyyaml is not installed. Run:\n  uv add pyyaml\nor:\n  pip install pyyaml"
        )

    if not scenarios_dir.is_dir():
        raise FileNotFoundError(f"Scenarios directory not found: {scenarios_dir}")

    scenarios: list[Scenario] = []

    for yaml_path in sorted(scenarios_dir.glob("*.yaml")):
        if yaml_path.stem == "repo_tree":
            continue

        raw = yaml.safe_load(yaml_path.read_text())
        if not raw or "scenarios" not in raw:
            continue

        for item in raw["scenarios"]:
            scenario = _parse_scenario(item, yaml_path)
            if difficulty_filter is None or scenario.difficulty == difficulty_filter:
                scenarios.append(scenario)

    return scenarios


_REQUIRED_FIELDS = {
    "id",
    "name",
    "difficulty",
    "prompt",
    "expected_files",
    "expected_approach",
    "expected_patterns",
    "common_pitfalls",
    "acceptance_criteria",
}


def _parse_scenario(data: dict[str, Any], source: Path) -> Scenario:
    """Parse and validate a single scenario dict."""
    missing = _REQUIRED_FIELDS - set(data.keys())
    if missing:
        raise ValueError(f"Scenario in {source.name} missing fields: {', '.join(sorted(missing))}")

    try:
        difficulty = Difficulty(str(data["difficulty"]).lower())
    except ValueError:
        raise ValueError(
            f"Invalid difficulty {data['difficulty']!r} in {source.name}. "
            f"Must be one of: {', '.join(d.value for d in Difficulty)}"
        )

    return Scenario(
        id=str(data["id"]),
        name=str(data["name"]),
        difficulty=difficulty,
        prompt=str(data["prompt"]),
        expected_files=[str(f) for f in data["expected_files"]],
        expected_approach=str(data["expected_approach"]),
        expected_patterns=[str(p) for p in data["expected_patterns"]],
        common_pitfalls=[str(p) for p in data["common_pitfalls"]],
        acceptance_criteria=[str(c) for c in data["acceptance_criteria"]],
    )
