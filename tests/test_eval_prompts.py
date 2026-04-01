"""Tests for eval/prompts.py prompt construction and response parsing."""

from dr_agents_tester.eval.models import Difficulty, Scenario, ScoreCard
from dr_agents_tester.eval.prompts import (
    build_agent_prompt,
    build_scoring_prompt,
    parse_agent_response,
    parse_score_card_json,
)


def _make_scenario() -> Scenario:
    return Scenario(
        id="test-1",
        name="Test scenario",
        difficulty=Difficulty.EASY,
        prompt="Add a health endpoint to the backend.",
        expected_files=["backend/routes/health.py", "backend/main.py"],
        expected_approach="Create a new route module for health checks.",
        expected_patterns=["FastAPI router pattern", "Pydantic models"],
        common_pitfalls=["Adding auth to health endpoint"],
        acceptance_criteria=["Returns 200 with status JSON"],
    )


class TestBuildAgentPrompt:
    def test_contains_scenario_prompt(self) -> None:
        prompt = build_agent_prompt(_make_scenario(), "file_tree_here")
        assert "Add a health endpoint" in prompt

    def test_contains_repo_tree(self) -> None:
        prompt = build_agent_prompt(_make_scenario(), "src/\n  main.py")
        assert "src/\n  main.py" in prompt

    def test_no_agents_md(self) -> None:
        prompt = build_agent_prompt(_make_scenario(), "tree", agents_md_content=None)
        assert "Repository Guide" not in prompt

    def test_with_agents_md(self) -> None:
        prompt = build_agent_prompt(
            _make_scenario(), "tree", agents_md_content="# Guide\nUse FastAPI."
        )
        assert "Repository Guide" in prompt
        assert "Use FastAPI." in prompt

    def test_instructions_present(self) -> None:
        prompt = build_agent_prompt(_make_scenario(), "tree")
        assert "implementation plan" in prompt.lower()


class TestBuildScoringPrompt:
    def test_contains_expected_files(self) -> None:
        prompt = build_scoring_prompt(_make_scenario(), "My plan text")
        assert "backend/routes/health.py" in prompt

    def test_contains_plan_text(self) -> None:
        prompt = build_scoring_prompt(_make_scenario(), "My detailed plan here")
        assert "My detailed plan here" in prompt

    def test_contains_scoring_dimensions(self) -> None:
        prompt = build_scoring_prompt(_make_scenario(), "plan")
        assert "file_identification" in prompt
        assert "approach_correctness" in prompt
        assert "pattern_awareness" in prompt
        assert "pitfall_avoidance" in prompt
        assert "completeness" in prompt

    def test_requests_json_response(self) -> None:
        prompt = build_scoring_prompt(_make_scenario(), "plan")
        assert "json" in prompt.lower()


class TestParseAgentResponse:
    def test_strips_whitespace(self) -> None:
        assert parse_agent_response("  plan text  \n") == "plan text"

    def test_passthrough(self) -> None:
        text = "## Plan\n\n1. Step one\n2. Step two"
        assert parse_agent_response(text) == text


class TestParseScoreCardJson:
    def test_valid_json(self) -> None:
        raw = """{
            "file_identification": 0.8,
            "approach_correctness": 0.9,
            "pattern_awareness": 0.7,
            "pitfall_avoidance": 0.6,
            "completeness": 0.85,
            "rationale": "Good plan overall"
        }"""
        card = parse_score_card_json(raw)
        assert card.file_identification == 0.8
        assert card.approach_correctness == 0.9
        assert card.rationale == "Good plan overall"
        assert card.overall > 0

    def test_json_in_code_fence(self) -> None:
        raw = """Here is my evaluation:

```json
{
    "file_identification": 0.7,
    "approach_correctness": 0.8,
    "pattern_awareness": 0.6,
    "pitfall_avoidance": 0.5,
    "completeness": 0.7,
    "rationale": "Decent"
}
```"""
        card = parse_score_card_json(raw)
        assert card.file_identification == 0.7
        assert card.rationale == "Decent"

    def test_clamps_values(self) -> None:
        raw = """{
            "file_identification": 1.5,
            "approach_correctness": -0.2,
            "pattern_awareness": 0.5,
            "pitfall_avoidance": 0.5,
            "completeness": 0.5,
            "rationale": "Out of range"
        }"""
        card = parse_score_card_json(raw)
        assert card.file_identification == 1.0
        assert card.approach_correctness == 0.0

    def test_invalid_json(self) -> None:
        card = parse_score_card_json("not json at all")
        assert card.overall == 0.0
        assert "Failed to parse" in card.rationale

    def test_missing_fields_default_zero(self) -> None:
        raw = '{"file_identification": 0.5, "rationale": "partial"}'
        card = parse_score_card_json(raw)
        assert card.file_identification == 0.5
        assert card.approach_correctness == 0.0

    def test_overall_computed(self) -> None:
        raw = """{
            "file_identification": 1.0,
            "approach_correctness": 1.0,
            "pattern_awareness": 1.0,
            "pitfall_avoidance": 1.0,
            "completeness": 1.0,
            "rationale": "Perfect"
        }"""
        card = parse_score_card_json(raw)
        assert card.overall == 1.0
