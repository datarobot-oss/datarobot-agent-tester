"""Tests for eval/report.py Markdown and JSON report generation."""

from dr_agents_tester.eval.models import (
    AgentResponse,
    ConditionStats,
    ConditionType,
    EvalResult,
    EvaluationReport,
    LLMUsage,
    PairwiseComparison,
    ScoreCard,
)
from dr_agents_tester.eval.report import (
    dict_to_report,
    generate_markdown_report,
    report_to_dict,
)


def _make_report() -> EvaluationReport:
    results = [
        EvalResult(
            scenario_id="test-1",
            condition=ConditionType.NO_CONTEXT,
            run_number=1,
            agent_response=AgentResponse(
                plan_text="Plan for test-1",
                usage=LLMUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150),
            ),
            score_card=ScoreCard(
                file_identification=0.8,
                approach_correctness=0.7,
                pattern_awareness=0.6,
                pitfall_avoidance=0.5,
                completeness=0.9,
                overall=0.72,
                rationale="Good overall",
            ),
            scoring_usage=LLMUsage(prompt_tokens=200, total_tokens=250),
        ),
        EvalResult(
            scenario_id="test-1",
            condition=ConditionType.GENERATED,
            run_number=1,
            agent_response=AgentResponse(plan_text="Better plan", usage=LLMUsage()),
            score_card=ScoreCard(
                file_identification=0.9,
                approach_correctness=0.85,
                pattern_awareness=0.8,
                pitfall_avoidance=0.7,
                completeness=0.95,
                overall=0.85,
                rationale="Excellent",
            ),
            scoring_usage=LLMUsage(),
        ),
    ]
    return EvaluationReport(
        results=results,
        condition_stats=[
            ConditionStats(
                condition=ConditionType.NO_CONTEXT,
                mean_overall=0.72,
                std_overall=0.0,
                n_results=1,
            ),
            ConditionStats(
                condition=ConditionType.GENERATED,
                mean_overall=0.85,
                std_overall=0.0,
                n_results=1,
            ),
        ],
        pairwise_comparisons=[
            PairwiseComparison(
                condition_a=ConditionType.NO_CONTEXT,
                condition_b=ConditionType.GENERATED,
                mean_diff=-0.13,
                t_statistic=-2.5,
                p_value=0.03,
                n_pairs=1,
                significant=True,
            ),
        ],
        n_scenarios=1,
        n_runs=1,
        conditions_tested=[ConditionType.NO_CONTEXT, ConditionType.GENERATED],
    )


class TestGenerateMarkdownReport:
    def test_contains_header(self) -> None:
        md = generate_markdown_report(_make_report())
        assert "# AGENTS.md Evaluation Report" in md

    def test_contains_summary_stats(self) -> None:
        md = generate_markdown_report(_make_report())
        assert "Scenarios:" in md
        assert "Runs per scenario:" in md
        assert "no_context" in md
        assert "generated" in md

    def test_contains_condition_table(self) -> None:
        md = generate_markdown_report(_make_report())
        assert "Summary by Condition" in md
        assert "0.720" in md
        assert "0.850" in md

    def test_contains_pairwise_table(self) -> None:
        md = generate_markdown_report(_make_report())
        assert "Pairwise Comparisons" in md
        assert "no_context vs generated" in md
        assert "Yes" in md  # significant

    def test_contains_per_scenario(self) -> None:
        md = generate_markdown_report(_make_report())
        assert "Per-Scenario Results" in md
        assert "test-1" in md

    def test_empty_report(self) -> None:
        md = generate_markdown_report(EvaluationReport())
        assert "# AGENTS.md Evaluation Report" in md
        assert "Total evaluations:** 0" in md


class TestReportSerialization:
    def test_roundtrip(self) -> None:
        original = _make_report()
        data = report_to_dict(original)
        restored = dict_to_report(data)

        assert restored.n_scenarios == original.n_scenarios
        assert restored.n_runs == original.n_runs
        assert len(restored.results) == len(original.results)
        assert len(restored.condition_stats) == len(original.condition_stats)
        assert len(restored.pairwise_comparisons) == len(original.pairwise_comparisons)

    def test_result_fields_preserved(self) -> None:
        original = _make_report()
        data = report_to_dict(original)
        restored = dict_to_report(data)

        r = restored.results[0]
        assert r.scenario_id == "test-1"
        assert r.condition == ConditionType.NO_CONTEXT
        assert r.score_card.file_identification == 0.8
        assert r.agent_response.plan_text == "Plan for test-1"
        assert r.agent_response.usage.prompt_tokens == 100

    def test_condition_stats_preserved(self) -> None:
        original = _make_report()
        data = report_to_dict(original)
        restored = dict_to_report(data)

        s = restored.condition_stats[0]
        assert s.condition == ConditionType.NO_CONTEXT
        assert s.mean_overall == 0.72

    def test_comparison_preserved(self) -> None:
        original = _make_report()
        data = report_to_dict(original)
        restored = dict_to_report(data)

        pc = restored.pairwise_comparisons[0]
        assert pc.condition_a == ConditionType.NO_CONTEXT
        assert pc.significant is True
        assert pc.mean_diff == -0.13

    def test_dict_structure(self) -> None:
        data = report_to_dict(_make_report())
        assert "n_scenarios" in data
        assert "results" in data
        assert "condition_stats" in data
        assert "pairwise_comparisons" in data
        assert "conditions_tested" in data
