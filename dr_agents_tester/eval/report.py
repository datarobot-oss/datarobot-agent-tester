"""Markdown and JSON report generation for evaluation results."""

from __future__ import annotations

from .models import (
    AgentResponse,
    CheckResult,
    ConditionStats,
    ConditionType,
    EvalResult,
    EvaluationReport,
    LLMUsage,
    OutcomeResult,
    OutcomeStats,
    PairwiseComparison,
    ScoreCard,
    TrajectoryMetrics,
)


def _has_behavioral(report: EvaluationReport) -> bool:
    return any(r.outcome is not None for r in report.results)


def generate_markdown_report(report: EvaluationReport) -> str:
    """Generate a human-readable Markdown report from evaluation results."""
    lines: list[str] = []

    if _has_behavioral(report):
        lines.append("# Skill Behavioral Evaluation Report\n")
    else:
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

    if _has_behavioral(report):
        lines.extend(_behavioral_sections(report))

    return "\n".join(lines)


def _fmt_opt(value: float | int | None, spec: str = ".0f") -> str:
    return "n/a" if value is None else format(value, spec)


def _behavioral_sections(report: EvaluationReport) -> list[str]:
    """Behavioral-outcome tables, appended after the score-based sections."""
    lines: list[str] = []

    if report.outcome_stats:
        lines.append("## Behavioral Outcomes\n")
        lines.append(
            "| Condition | N | Pass rate | pass@k | Turns | Tool calls | Errors "
            "| Tokens | Wall (s) | Skill trigger |"
        )
        lines.append("|" + "---|" * 10)
        for os_ in report.outcome_stats:
            lines.append(
                f"| {os_.condition.value} | {os_.n_results} "
                f"| {os_.pass_rate:.0%} | {os_.pass_at_k:.0%} (k={os_.k}) "
                f"| {os_.mean_turns:.1f} | {os_.mean_tool_calls:.1f} "
                f"| {os_.mean_errors:.1f} | {os_.mean_total_tokens:.0f} "
                f"| {os_.mean_wall_seconds:.0f} | {os_.skill_trigger_rate:.0%} |"
            )
        lines.append("")

    lines.append("## Behavioral Per-Scenario Results\n")
    by_scenario: dict[str, list[EvalResult]] = {}
    for r in report.results:
        if r.outcome is not None:
            by_scenario.setdefault(r.scenario_id, []).append(r)

    for scenario_id in sorted(by_scenario.keys()):
        lines.append(f"### {scenario_id}\n")
        lines.append(
            "| Condition | Run | Outcome | Failed checks | Turns | Tool calls "
            "| Errors | Tokens | Wall (s) | Transcript |"
        )
        lines.append("|" + "---|" * 10)
        for r in sorted(by_scenario[scenario_id], key=lambda x: (x.condition.value, x.run_number)):
            assert r.outcome is not None
            verdict = "PASS" if r.outcome.outcome_pass else "FAIL"
            failed = (
                "; ".join(
                    f"{c.check_type}: {c.error or c.evidence}"
                    for c in r.outcome.checks
                    if not c.passed
                )
                or "—"
            )
            t = r.trajectory or TrajectoryMetrics()
            lines.append(
                f"| {r.condition.value} | {r.run_number} | {verdict} "
                f"| {failed} | {_fmt_opt(t.num_turns)} | {_fmt_opt(t.num_tool_calls)} "
                f"| {_fmt_opt(t.num_errors)} | {_fmt_opt(t.total_tokens)} "
                f"| {_fmt_opt(t.wall_seconds, '.0f')} | {r.transcript_path or '—'} |"
            )
        lines.append("")

    return lines


def report_to_dict(report: EvaluationReport) -> dict[str, object]:
    """Serialize an EvaluationReport to a JSON-compatible dict.

    Behavioral-only keys are emitted only when present so plan-eval JSON stays
    byte-identical to the pre-behavioral format.
    """
    data: dict[str, object] = {
        "n_scenarios": report.n_scenarios,
        "n_runs": report.n_runs,
        "conditions_tested": [c.value for c in report.conditions_tested],
        "condition_stats": [_condition_stats_to_dict(s) for s in report.condition_stats],
        "pairwise_comparisons": [_comparison_to_dict(pc) for pc in report.pairwise_comparisons],
        "results": [_result_to_dict(r) for r in report.results],
    }
    if report.outcome_stats or report.backend_name != "plan":
        data["backend_name"] = report.backend_name
        data["outcome_stats"] = [_outcome_stats_to_dict(s) for s in report.outcome_stats]
    return data


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

    outcome_stats_raw = data.get("outcome_stats", [])
    assert isinstance(outcome_stats_raw, list)

    return EvaluationReport(
        results=[_dict_to_result(r) for r in results_raw],
        condition_stats=[_dict_to_condition_stats(s) for s in stats_raw],
        pairwise_comparisons=[_dict_to_comparison(pc) for pc in comparisons_raw],
        n_scenarios=int(data.get("n_scenarios", 0)),  # type: ignore[call-overload]
        n_runs=int(data.get("n_runs", 0)),  # type: ignore[call-overload]
        conditions_tested=[ConditionType(c) for c in conditions_raw],
        outcome_stats=[_dict_to_outcome_stats(s) for s in outcome_stats_raw],
        backend_name=str(data.get("backend_name", "plan")),
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
    data: dict[str, object] = {
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
    # Behavioral-only keys: emitted only when set, so plan-eval JSON is unchanged.
    if r.outcome is not None:
        data["outcome"] = _outcome_to_dict(r.outcome)
    if r.trajectory is not None:
        data["trajectory"] = _trajectory_to_dict(r.trajectory)
    if r.transcript_path is not None:
        data["transcript_path"] = r.transcript_path
    if r.run_id is not None:
        data["run_id"] = r.run_id
    return data


def _outcome_to_dict(o: OutcomeResult) -> dict[str, object]:
    return {
        "outcome_pass": o.outcome_pass,
        "checks": [
            {
                "check_type": c.check_type,
                "passed": c.passed,
                "evidence": c.evidence,
                "error": c.error,
                "duration_seconds": c.duration_seconds,
            }
            for c in o.checks
        ],
    }


def _trajectory_to_dict(t: TrajectoryMetrics) -> dict[str, object]:
    return {
        "wall_seconds": t.wall_seconds,
        "total_tokens": t.total_tokens,
        "input_tokens": t.input_tokens,
        "output_tokens": t.output_tokens,
        "cache_read_tokens": t.cache_read_tokens,
        "cache_write_tokens": t.cache_write_tokens,
        "cost": t.cost,
        "num_turns": t.num_turns,
        "num_tool_calls": t.num_tool_calls,
        "num_errors": t.num_errors,
        "num_retries": t.num_retries,
        "skill_triggered": t.skill_triggered,
        "skills_used": list(t.skills_used),
        "skill_first_turn": t.skill_first_turn,
        "num_unknown_events": t.num_unknown_events,
    }


def _outcome_stats_to_dict(s: OutcomeStats) -> dict[str, object]:
    return {
        "condition": s.condition.value,
        "n_results": s.n_results,
        "k": s.k,
        "pass_rate": s.pass_rate,
        "pass_at_k": s.pass_at_k,
        "mean_turns": s.mean_turns,
        "mean_tool_calls": s.mean_tool_calls,
        "mean_errors": s.mean_errors,
        "mean_total_tokens": s.mean_total_tokens,
        "mean_wall_seconds": s.mean_wall_seconds,
        "skill_trigger_rate": s.skill_trigger_rate,
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
    agent_resp = d.get("agent_response", {})
    assert isinstance(agent_resp, dict)
    score = d.get("score_card", {})
    assert isinstance(score, dict)
    scoring_usage = d.get("scoring_usage", {})
    assert isinstance(scoring_usage, dict)

    transcript_path = d.get("transcript_path")
    run_id = d.get("run_id")

    return EvalResult(
        scenario_id=str(d["scenario_id"]),
        condition=ConditionType(d["condition"]),
        run_number=int(d["run_number"]),
        agent_response=AgentResponse(
            plan_text=str(agent_resp.get("plan_text", "")),
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
        outcome=_dict_to_outcome(d.get("outcome")),
        trajectory=_dict_to_trajectory(d.get("trajectory")),
        transcript_path=str(transcript_path) if transcript_path is not None else None,
        run_id=str(run_id) if run_id is not None else None,
    )


def _dict_to_outcome(d: object) -> OutcomeResult | None:
    if not isinstance(d, dict):
        return None
    checks_raw = d.get("checks", [])
    checks: list[CheckResult] = []
    if isinstance(checks_raw, list):
        for c in checks_raw:
            if not isinstance(c, dict):
                continue
            error = c.get("error")
            checks.append(
                CheckResult(
                    check_type=str(c.get("check_type", "")),
                    passed=bool(c.get("passed", False)),
                    evidence=str(c.get("evidence", "")),
                    error=str(error) if error is not None else None,
                    duration_seconds=float(c.get("duration_seconds", 0)),
                )
            )
    return OutcomeResult(outcome_pass=bool(d.get("outcome_pass", False)), checks=checks)


def _opt_float(value: object) -> float | None:
    return float(value) if isinstance(value, (int, float)) else None


def _opt_int(value: object) -> int | None:
    return int(value) if isinstance(value, (int, float)) else None


def _dict_to_trajectory(d: object) -> TrajectoryMetrics | None:
    if not isinstance(d, dict):
        return None
    skill_triggered = d.get("skill_triggered")
    skills_used_raw = d.get("skills_used", [])
    return TrajectoryMetrics(
        wall_seconds=_opt_float(d.get("wall_seconds")),
        total_tokens=_opt_int(d.get("total_tokens")),
        input_tokens=_opt_int(d.get("input_tokens")),
        output_tokens=_opt_int(d.get("output_tokens")),
        cache_read_tokens=_opt_int(d.get("cache_read_tokens")),
        cache_write_tokens=_opt_int(d.get("cache_write_tokens")),
        cost=_opt_float(d.get("cost")),
        num_turns=_opt_int(d.get("num_turns")),
        num_tool_calls=_opt_int(d.get("num_tool_calls")),
        num_errors=_opt_int(d.get("num_errors")),
        num_retries=_opt_int(d.get("num_retries")),
        skill_triggered=bool(skill_triggered) if skill_triggered is not None else None,
        skills_used=[str(s) for s in skills_used_raw] if isinstance(skills_used_raw, list) else [],
        skill_first_turn=_opt_int(d.get("skill_first_turn")),
        num_unknown_events=_opt_int(d.get("num_unknown_events")) or 0,
    )


def _dict_to_outcome_stats(d: object) -> OutcomeStats:
    assert isinstance(d, dict)
    return OutcomeStats(
        condition=ConditionType(d["condition"]),
        n_results=int(d.get("n_results", 0)),
        k=int(d.get("k", 0)),
        pass_rate=float(d.get("pass_rate", 0)),
        pass_at_k=float(d.get("pass_at_k", 0)),
        mean_turns=float(d.get("mean_turns", 0)),
        mean_tool_calls=float(d.get("mean_tool_calls", 0)),
        mean_errors=float(d.get("mean_errors", 0)),
        mean_total_tokens=float(d.get("mean_total_tokens", 0)),
        mean_wall_seconds=float(d.get("mean_wall_seconds", 0)),
        skill_trigger_rate=float(d.get("skill_trigger_rate", 0)),
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
