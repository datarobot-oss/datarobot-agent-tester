"""Tests for behavioral extensions to report serialization and markdown output."""

import json

from dr_agents_tester.eval.models import (
    AgentResponse,
    CheckResult,
    ConditionType,
    EvalResult,
    EvaluationReport,
    LLMUsage,
    OutcomeResult,
    OutcomeStats,
    ScoreCard,
    TrajectoryMetrics,
)
from dr_agents_tester.eval.report import (
    dict_to_report,
    generate_markdown_report,
    report_to_dict,
)


def _plan_result(run_number: int = 1) -> EvalResult:
    return EvalResult(
        scenario_id="plan-s",
        condition=ConditionType.NO_CONTEXT,
        run_number=run_number,
        agent_response=AgentResponse(plan_text="the plan", usage=LLMUsage(prompt_tokens=10)),
        score_card=ScoreCard(overall=0.5),
        scoring_usage=LLMUsage(),
    )


def _behavioral_result(passed: bool = True, run_number: int = 1) -> EvalResult:
    return EvalResult(
        scenario_id="golden",
        condition=ConditionType.SKILL_PR,
        run_number=run_number,
        agent_response=AgentResponse(plan_text="final message", usage=LLMUsage()),
        score_card=ScoreCard(),
        scoring_usage=LLMUsage(),
        outcome=OutcomeResult(
            outcome_pass=passed,
            checks=[
                CheckResult(check_type="dr_project_exists", passed=True, evidence="proj-1"),
                CheckResult(
                    check_type="file_exists",
                    passed=passed,
                    evidence="" if not passed else "predictions.csv (120 bytes)",
                    error=None if passed else "no file matching",
                ),
            ],
        ),
        trajectory=TrajectoryMetrics(
            wall_seconds=120.5,
            total_tokens=5000,
            num_turns=4,
            num_tool_calls=9,
            num_errors=1,
            skill_triggered=True,
            skill_triggered_expected=True,
            skills_used=["datarobot-model-training"],
            skill_first_turn=1,
        ),
        transcript_path="runs/golden-r1/transcript.raw.jsonl",
        run_id="drat-golden-skill-pr-r1-x",
    )


class TestPlanJsonUnchanged:
    def test_plan_result_dict_has_no_behavioral_keys(self) -> None:
        report = EvaluationReport(results=[_plan_result()])
        data = report_to_dict(report)
        results = data["results"]
        assert isinstance(results, list)
        assert set(results[0].keys()) == {
            "scenario_id",
            "condition",
            "run_number",
            "agent_response",
            "score_card",
            "scoring_usage",
        }
        assert "backend_name" not in data
        assert "outcome_stats" not in data

    def test_legacy_json_without_new_keys_loads(self) -> None:
        report = EvaluationReport(results=[_plan_result()])
        legacy = json.loads(json.dumps(report_to_dict(report)))
        loaded = dict_to_report(legacy)
        assert loaded.backend_name == "plan"
        assert loaded.outcome_stats == []
        r = loaded.results[0]
        assert r.outcome is None
        assert r.trajectory is None
        assert r.transcript_path is None
        assert r.run_id is None


class TestBehavioralRoundTrip:
    def test_round_trip_preserves_behavioral_fields(self) -> None:
        report = EvaluationReport(
            results=[_behavioral_result(passed=False)],
            outcome_stats=[
                OutcomeStats(
                    condition=ConditionType.SKILL_PR,
                    n_results=3,
                    k=3,
                    pass_rate=0.6667,
                    pass_at_k=1.0,
                    mean_turns=4.0,
                    skill_trigger_rate=1.0,
                    expected_skill_trigger_rate=0.5,
                )
            ],
            backend_name="agent",
        )
        loaded = dict_to_report(json.loads(json.dumps(report_to_dict(report))))

        assert loaded.backend_name == "agent"
        stats = loaded.outcome_stats[0]
        assert stats.condition == ConditionType.SKILL_PR
        assert stats.pass_rate == 0.6667
        assert stats.pass_at_k == 1.0
        assert stats.expected_skill_trigger_rate == 0.5

        r = loaded.results[0]
        assert r.outcome is not None and r.outcome.outcome_pass is False
        assert [c.check_type for c in r.outcome.checks] == [
            "dr_project_exists",
            "file_exists",
        ]
        assert r.outcome.checks[1].error == "no file matching"
        assert r.trajectory is not None
        assert r.trajectory.wall_seconds == 120.5
        assert r.trajectory.num_tool_calls == 9
        assert r.trajectory.skill_triggered is True
        assert r.trajectory.skill_triggered_expected is True
        assert r.trajectory.skills_used == ["datarobot-model-training"]
        # Unreported driver metrics stay None through the round trip
        assert r.trajectory.cost is None
        assert r.trajectory.num_retries is None
        assert r.transcript_path == "runs/golden-r1/transcript.raw.jsonl"
        assert r.run_id == "drat-golden-skill-pr-r1-x"


class TestBehavioralMarkdown:
    def test_plan_report_title_and_no_behavioral_sections(self) -> None:
        md = generate_markdown_report(EvaluationReport(results=[_plan_result()]))
        assert "# AGENTS.md Evaluation Report" in md
        assert "Behavioral" not in md

    def test_behavioral_report_sections(self) -> None:
        report = EvaluationReport(
            results=[
                _behavioral_result(passed=True),
                _behavioral_result(passed=False, run_number=2),
            ],
            outcome_stats=[
                OutcomeStats(
                    condition=ConditionType.SKILL_PR,
                    n_results=2,
                    k=2,
                    pass_rate=0.5,
                    pass_at_k=1.0,
                    mean_turns=4.0,
                    mean_tool_calls=9.0,
                    mean_errors=1.0,
                    mean_total_tokens=5000.0,
                    mean_wall_seconds=120.0,
                    skill_trigger_rate=1.0,
                    expected_skill_trigger_rate=1.0,
                )
            ],
            backend_name="agent",
        )
        md = generate_markdown_report(report)
        assert "# Skill Behavioral Evaluation Report" in md
        assert "## Behavioral Outcomes" in md
        assert "Expected skill" in md
        assert "## Behavioral Per-Scenario Results" in md
        assert "PASS" in md and "FAIL" in md
        assert "file_exists: no file matching" in md
        assert "runs/golden-r1/transcript.raw.jsonl" in md
        # Efficiency columns render
        assert "50%" in md  # pass rate
