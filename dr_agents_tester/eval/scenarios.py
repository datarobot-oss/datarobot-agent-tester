"""Load and validate evaluation scenarios from YAML files.

Two scenario kinds share the file format (a ``scenarios`` list per YAML file),
discriminated by an explicit ``kind`` field on each entry:

- ``kind: plan`` (or absent — the v1 default): AGENTS.md plan-eval scenarios.
- ``kind: behavioral``: schema-v2 scenarios executed by a real coding agent,
  with programmatic ``success_checks`` (design doc §4.1).

Discrimination is explicit rather than inferred from field presence so a
half-written behavioral scenario fails loudly instead of silently validating
as a (broken) plan scenario.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .models import BehavioralScenario, CheckSpec, Difficulty, FixtureSpec, Scenario
from .templating import ALLOWED_ENV_NAME_RE, find_env_tokens


def _load_raw(
    scenarios_dir: Path, include_subdirs: bool = False
) -> list[tuple[dict[str, Any], Path]]:
    """Shared file walk: yield every scenario dict with its source file.

    With ``include_subdirs``, one subdirectory level is also scanned — the
    ``scenarios/<skill-name>/*.yaml`` layout the skills repo uses for
    single-skill behavioral scenarios.
    """
    try:
        import yaml  # type: ignore[import-untyped]
    except ImportError:
        raise ImportError(
            "pyyaml is not installed. Run:\n  uv add pyyaml\nor:\n  pip install pyyaml"
        )

    if not scenarios_dir.is_dir():
        raise FileNotFoundError(f"Scenarios directory not found: {scenarios_dir}")

    yaml_paths = list(scenarios_dir.glob("*.yaml"))
    if include_subdirs:
        yaml_paths += scenarios_dir.glob("*/*.yaml")

    entries: list[tuple[dict[str, Any], Path]] = []
    for yaml_path in sorted(yaml_paths):
        if yaml_path.stem == "repo_tree":
            continue

        raw = yaml.safe_load(yaml_path.read_text())
        if not raw or "scenarios" not in raw:
            continue

        for item in raw["scenarios"]:
            entries.append((item, yaml_path))
    return entries


def _kind(data: dict[str, Any]) -> str:
    return str(data.get("kind", "plan")).lower()


def load_scenarios(
    scenarios_dir: Path,
    difficulty_filter: Difficulty | None = None,
) -> list[Scenario]:
    """Load plan-eval scenarios from YAML files in a directory.

    Entries with ``kind: behavioral`` are skipped (load them with
    :func:`load_behavioral_scenarios`); directories may co-locate both kinds.

    Args:
        scenarios_dir: Directory containing scenario YAML files.
        difficulty_filter: If set, only load scenarios with this difficulty.

    Returns:
        List of validated Scenario objects.

    Raises:
        FileNotFoundError: If the directory doesn't exist.
        ValueError: If a scenario is missing required fields.
    """
    scenarios: list[Scenario] = []
    for item, yaml_path in _load_raw(scenarios_dir):
        if _kind(item) != "plan":
            continue
        scenario = _parse_scenario(item, yaml_path)
        if difficulty_filter is None or scenario.difficulty == difficulty_filter:
            scenarios.append(scenario)
    return scenarios


def load_behavioral_scenarios(
    scenarios_dir: Path,
    difficulty_filter: Difficulty | None = None,
) -> list[BehavioralScenario]:
    """Load behavioral (schema v2) scenarios from YAML files in a directory.

    Entries without ``kind: behavioral`` are skipped. One subdirectory level
    is scanned too (the ``scenarios/<skill-name>/`` layout).
    """
    scenarios: list[BehavioralScenario] = []
    for item, yaml_path in _load_raw(scenarios_dir, include_subdirs=True):
        if _kind(item) != "behavioral":
            continue
        scenario = _parse_behavioral(item, yaml_path)
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


_BEHAVIORAL_REQUIRED_FIELDS = {
    "id",
    "name",
    "difficulty",
    "prompt",
    "skills_under_test",
    "success_checks",
}

# Plan-eval-only fields; their presence on a behavioral scenario is almost
# certainly copy-paste schema confusion, so fail loudly.
_PLAN_ONLY_FIELDS = {
    "expected_files",
    "expected_patterns",
    "expected_approach",
    "acceptance_criteria",
}


def _parse_behavioral(data: dict[str, Any], source: Path) -> BehavioralScenario:
    """Parse and validate a single behavioral (schema v2) scenario dict."""
    from .checks import known_check_types

    missing = _BEHAVIORAL_REQUIRED_FIELDS - set(data.keys())
    if missing:
        raise ValueError(
            f"Behavioral scenario in {source.name} missing fields: {', '.join(sorted(missing))}"
        )

    stray = _PLAN_ONLY_FIELDS & set(data.keys())
    if stray:
        raise ValueError(
            f"Behavioral scenario {data.get('id')!r} in {source.name} has plan-eval-only "
            f"fields: {', '.join(sorted(stray))}. Did you mean kind: plan?"
        )

    try:
        difficulty = Difficulty(str(data["difficulty"]).lower())
    except ValueError:
        raise ValueError(
            f"Invalid difficulty {data['difficulty']!r} in {source.name}. "
            f"Must be one of: {', '.join(d.value for d in Difficulty)}"
        )

    skills = [str(s) for s in data["skills_under_test"]]
    if not skills:
        raise ValueError(
            f"Behavioral scenario {data['id']!r} in {source.name}: "
            "skills_under_test must be a non-empty list"
        )

    checks = _parse_checks(data["success_checks"], data["id"], source, known_check_types())

    timeout_minutes = int(data.get("timeout_minutes", 30))
    if timeout_minutes <= 0:
        raise ValueError(
            f"Behavioral scenario {data['id']!r} in {source.name}: "
            f"timeout_minutes must be positive, got {timeout_minutes}"
        )

    scenario = BehavioralScenario(
        id=str(data["id"]),
        name=str(data["name"]),
        difficulty=difficulty,
        prompt=str(data["prompt"]),
        skills_under_test=skills,
        success_checks=checks,
        fixtures=_parse_fixtures(data.get("fixtures", []), data["id"], source),
        env={str(k): str(v) for k, v in (data.get("env") or {}).items()},
        requires_env=[str(v) for v in (data.get("requires_env") or [])],
        rubric=str(data.get("rubric", "")),
        common_pitfalls=[str(p) for p in data.get("common_pitfalls", [])],
        timeout_minutes=timeout_minutes,
        source_dir=source.parent,
    )
    _validate_env_tokens(scenario, source)
    return scenario


def _validate_env_tokens(scenario: BehavioralScenario, source: Path) -> None:
    """Cross-check ``requires_env`` declarations against ``{env:VAR}`` references.

    Purely syntactic so YAML validation stays offline; whether the variables
    are actually set is enforced at run start. Declared-but-unreferenced and
    referenced-but-undeclared are both errors — each is how a typo hides.
    """
    where = f"Behavioral scenario {scenario.id!r} in {source.name}"

    bad_names = [v for v in scenario.requires_env if not ALLOWED_ENV_NAME_RE.match(v)]
    if bad_names:
        raise ValueError(
            f"{where}: requires_env entries must match {ALLOWED_ENV_NAME_RE.pattern!r} "
            f"(scenarios may only template fixture/run variables, never credentials): "
            f"{', '.join(bad_names)}"
        )

    referenced = find_env_tokens(scenario.prompt)
    for value in scenario.env.values():
        referenced |= find_env_tokens(value)
    for check in scenario.success_checks:
        for value in check.params.values():
            if isinstance(value, str):
                referenced |= find_env_tokens(value)

    declared = set(scenario.requires_env)
    undeclared = referenced - declared
    if undeclared:
        raise ValueError(
            f"{where}: {{env:...}} references not declared in requires_env: "
            f"{', '.join(sorted(undeclared))}"
        )
    unused = declared - referenced
    if unused:
        raise ValueError(
            f"{where}: requires_env declares variables never referenced as "
            f"{{env:...}} tokens: {', '.join(sorted(unused))}"
        )


def _parse_checks(
    raw: Any, scenario_id: str, source: Path, known: frozenset[str]
) -> list[CheckSpec]:
    if not isinstance(raw, list) or not raw:
        raise ValueError(
            f"Behavioral scenario {scenario_id!r} in {source.name}: "
            "success_checks must be a non-empty list"
        )
    specs: list[CheckSpec] = []
    for entry in raw:
        if not isinstance(entry, dict) or "type" not in entry:
            raise ValueError(
                f"Behavioral scenario {scenario_id!r} in {source.name}: each success_check "
                f"needs a 'type' key, got: {entry!r}"
            )
        check_type = str(entry["type"])
        if check_type not in known:
            raise ValueError(
                f"Behavioral scenario {scenario_id!r} in {source.name}: unknown check type "
                f"{check_type!r}. Known types: {', '.join(sorted(known))}"
            )
        params = {str(k): v for k, v in entry.items() if k != "type"}
        specs.append(CheckSpec(type=check_type, params=params))
    return specs


def _parse_fixtures(raw: Any, scenario_id: str, source: Path) -> list[FixtureSpec]:
    if not isinstance(raw, list):
        raise ValueError(
            f"Behavioral scenario {scenario_id!r} in {source.name}: fixtures must be a list"
        )
    fixtures: list[FixtureSpec] = []
    for entry in raw:
        if isinstance(entry, str):
            fixtures.append(FixtureSpec(source=entry, dest=entry))
        elif isinstance(entry, dict) and "source" in entry:
            fixtures.append(
                FixtureSpec(
                    source=str(entry["source"]),
                    dest=str(entry.get("dest", entry["source"])),
                )
            )
        else:
            raise ValueError(
                f"Behavioral scenario {scenario_id!r} in {source.name}: each fixture must "
                f"be a string or a mapping with 'source' (and optional 'dest'), got: {entry!r}"
            )
    return fixtures
