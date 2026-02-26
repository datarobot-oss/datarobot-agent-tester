"""Markdown and JSON report generation for evaluation results."""

from __future__ import annotations

from .models import (
    AgentResponse,
    ConditionStats,
    ConditionType,
    EvalResult,
    EvaluationReport,
    LLMUsage,
    PairwiseComparison,
    ScoreCard,
)


def generate_markdown_report(report: EvaluationReport) -> str:
    """Generate a human-readable Markdown report from evaluation results."""
    lines: list[str] = []

    lines.append("# AGENTS.md Evaluation Report\n")
    lines.append(f"- **Scenarios:** {report.n_scenarios}")
    lines.append(f"- **Runs per scenario:** {report.n_runs}")
    lines.append(f"- **Conditions tested:** {', '.join(c.value for c in report.conditions_tested)}")
    lines.append(f"- **Total evaluations:** {len(report.results)}")
    lines.append("")

    # Condition summary table
    lines.append("## Summary by Condition\n")
    lines.append(
        "| Condition | N | Overall (mean) | Std | Files | Approach | Patterns "
        "| Pitfalls | Completeness | Tokens (mean) | Duration (s) |"
    )
    lines.append("|" + "---|" * 12)

    for s in report.condition_stats:
        lines.append(
            f"| {s.condition.value} | {s.n_results} "
            f"| {s.mean_overall:.3f} | {s.std_overall:.3f} "
            f"| {s.mean_file_identification:.3f} "
            f"| {s.mean_approach_correctness:.3f} "
            f"| {s.mean_pattern_awareness:.3f} "
            f"| {s.mean_pitfall_avoidance:.3f} "
            f"| {s.mean_completeness:.3f} "
            f"| {s.mean_total_tokens:.0f} "
            f"| {s.mean_duration_seconds:.1f} |"
        )
    lines.append("")

    # Pairwise comparisons
    if report.pairwise_comparisons:
        lines.append("## Pairwise Comparisons (Paired t-test)\n")
        lines.append("| Comparison | Mean Diff | t-statistic | p-value | N pairs | Significant |")
        lines.append("|" + "---|" * 6)

        for pc in report.pairwise_comparisons:
            sig = "Yes" if pc.significant else "No"
            lines.append(
                f"| {pc.condition_a.value} vs {pc.condition_b.value} "
                f"| {pc.mean_diff:+.4f} | {pc.t_statistic:.3f} "
                f"| {pc.p_value:.4f} | {pc.n_pairs} | {sig} |"
            )
        lines.append("")

    # Per-scenario breakdown
    lines.append("## Per-Scenario Results\n")
    # Group by scenario
    by_scenario: dict[str, list[EvalResult]] = {}
    for r in report.results:
        by_scenario.setdefault(r.scenario_id, []).append(r)

    for scenario_id in sorted(by_scenario.keys()):
        scenario_results = by_scenario[scenario_id]
        lines.append(f"### {scenario_id}\n")
        lines.append(
            "| Condition | Run | Overall | Files | Approach | Patterns | Pitfalls | Compl |"
        )
        lines.append("|" + "---|" * 8)

        for r in sorted(scenario_results, key=lambda x: (x.condition.value, x.run_number)):
            sc = r.score_card
            lines.append(
                f"| {r.condition.value} | {r.run_number} "
                f"| {sc.overall:.3f} | {sc.file_identification:.2f} "
                f"| {sc.approach_correctness:.2f} | {sc.pattern_awareness:.2f} "
                f"| {sc.pitfall_avoidance:.2f} | {sc.completeness:.2f} |"
            )
        lines.append("")

    return "\n".join(lines)


def report_to_dict(report: EvaluationReport) -> dict[str, object]:
    """Serialize an EvaluationReport to a JSON-compatible dict."""
    return {
        "n_scenarios": report.n_scenarios,
        "n_runs": report.n_runs,
        "conditions_tested": [c.value for c in report.conditions_tested],
        "condition_stats": [_condition_stats_to_dict(s) for s in report.condition_stats],
        "pairwise_comparisons": [_comparison_to_dict(pc) for pc in report.pairwise_comparisons],
        "results": [_result_to_dict(r) for r in report.results],
    }


def dict_to_report(data: dict[str, object]) -> EvaluationReport:
    """Deserialize a dict back into an EvaluationReport."""
    results_raw = data.get("results", [])
    assert isinstance(results_raw, list)

    stats_raw = data.get("condition_stats", [])
    assert isinstance(stats_raw, list)

    comparisons_raw = data.get("pairwise_comparisons", [])
    assert isinstance(comparisons_raw, list)

    conditions_raw = data.get("conditions_tested", [])
    assert isinstance(conditions_raw, list)

    return EvaluationReport(
        results=[_dict_to_result(r) for r in results_raw],
        condition_stats=[_dict_to_condition_stats(s) for s in stats_raw],
        pairwise_comparisons=[_dict_to_comparison(pc) for pc in comparisons_raw],
        n_scenarios=int(data.get("n_scenarios", 0)),  # type: ignore[call-overload]
        n_runs=int(data.get("n_runs", 0)),  # type: ignore[call-overload]
        conditions_tested=[ConditionType(c) for c in conditions_raw],
    )


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------


def _condition_stats_to_dict(s: ConditionStats) -> dict[str, object]:
    return {
        "condition": s.condition.value,
        "mean_overall": s.mean_overall,
        "std_overall": s.std_overall,
        "mean_file_identification": s.mean_file_identification,
        "mean_approach_correctness": s.mean_approach_correctness,
        "mean_pattern_awareness": s.mean_pattern_awareness,
        "mean_pitfall_avoidance": s.mean_pitfall_avoidance,
        "mean_completeness": s.mean_completeness,
        "mean_prompt_tokens": s.mean_prompt_tokens,
        "mean_completion_tokens": s.mean_completion_tokens,
        "mean_total_tokens": s.mean_total_tokens,
        "mean_duration_seconds": s.mean_duration_seconds,
        "n_results": s.n_results,
    }


def _comparison_to_dict(pc: PairwiseComparison) -> dict[str, object]:
    return {
        "condition_a": pc.condition_a.value,
        "condition_b": pc.condition_b.value,
        "mean_diff": pc.mean_diff,
        "t_statistic": pc.t_statistic,
        "p_value": pc.p_value,
        "n_pairs": pc.n_pairs,
        "significant": pc.significant,
    }


def _result_to_dict(r: EvalResult) -> dict[str, object]:
    return {
        "scenario_id": r.scenario_id,
        "condition": r.condition.value,
        "run_number": r.run_number,
        "agent_response": {
            "plan_text": r.agent_response.plan_text,
            "usage": _usage_to_dict(r.agent_response.usage),
        },
        "score_card": {
            "file_identification": r.score_card.file_identification,
            "approach_correctness": r.score_card.approach_correctness,
            "pattern_awareness": r.score_card.pattern_awareness,
            "pitfall_avoidance": r.score_card.pitfall_avoidance,
            "completeness": r.score_card.completeness,
            "overall": r.score_card.overall,
            "rationale": r.score_card.rationale,
        },
        "scoring_usage": _usage_to_dict(r.scoring_usage),
    }


def _usage_to_dict(u: LLMUsage) -> dict[str, object]:
    return {
        "prompt_tokens": u.prompt_tokens,
        "completion_tokens": u.completion_tokens,
        "total_tokens": u.total_tokens,
        "duration_seconds": u.duration_seconds,
    }


# ---------------------------------------------------------------------------
# Deserialization helpers
# ---------------------------------------------------------------------------


def _dict_to_result(d: object) -> EvalResult:
    assert isinstance(d, dict)
    agent_resp = d["agent_response"]
    assert isinstance(agent_resp, dict)
    score = d["score_card"]
    assert isinstance(score, dict)
    scoring_usage = d.get("scoring_usage", {})
    assert isinstance(scoring_usage, dict)

    return EvalResult(
        scenario_id=str(d["scenario_id"]),
        condition=ConditionType(d["condition"]),
        run_number=int(d["run_number"]),
        agent_response=AgentResponse(
            plan_text=str(agent_resp["plan_text"]),
            usage=_dict_to_usage(agent_resp.get("usage", {})),
        ),
        score_card=ScoreCard(
            file_identification=float(score.get("file_identification", 0)),
            approach_correctness=float(score.get("approach_correctness", 0)),
            pattern_awareness=float(score.get("pattern_awareness", 0)),
            pitfall_avoidance=float(score.get("pitfall_avoidance", 0)),
            completeness=float(score.get("completeness", 0)),
            overall=float(score.get("overall", 0)),
            rationale=str(score.get("rationale", "")),
        ),
        scoring_usage=_dict_to_usage(scoring_usage),
    )


def _dict_to_condition_stats(d: object) -> ConditionStats:
    assert isinstance(d, dict)
    return ConditionStats(
        condition=ConditionType(d["condition"]),
        mean_overall=float(d.get("mean_overall", 0)),
        std_overall=float(d.get("std_overall", 0)),
        mean_file_identification=float(d.get("mean_file_identification", 0)),
        mean_approach_correctness=float(d.get("mean_approach_correctness", 0)),
        mean_pattern_awareness=float(d.get("mean_pattern_awareness", 0)),
        mean_pitfall_avoidance=float(d.get("mean_pitfall_avoidance", 0)),
        mean_completeness=float(d.get("mean_completeness", 0)),
        mean_prompt_tokens=float(d.get("mean_prompt_tokens", 0)),
        mean_completion_tokens=float(d.get("mean_completion_tokens", 0)),
        mean_total_tokens=float(d.get("mean_total_tokens", 0)),
        mean_duration_seconds=float(d.get("mean_duration_seconds", 0)),
        n_results=int(d.get("n_results", 0)),
    )


def _dict_to_comparison(d: object) -> PairwiseComparison:
    assert isinstance(d, dict)
    return PairwiseComparison(
        condition_a=ConditionType(d["condition_a"]),
        condition_b=ConditionType(d["condition_b"]),
        mean_diff=float(d.get("mean_diff", 0)),
        t_statistic=float(d.get("t_statistic", 0)),
        p_value=float(d.get("p_value", 1)),
        n_pairs=int(d.get("n_pairs", 0)),
        significant=bool(d.get("significant", False)),
    )


def _dict_to_usage(d: object) -> LLMUsage:
    if not isinstance(d, dict):
        return LLMUsage()
    return LLMUsage(
        prompt_tokens=int(d.get("prompt_tokens", 0)),
        completion_tokens=int(d.get("completion_tokens", 0)),
        total_tokens=int(d.get("total_tokens", 0)),
        duration_seconds=float(d.get("duration_seconds", 0)),
    )
