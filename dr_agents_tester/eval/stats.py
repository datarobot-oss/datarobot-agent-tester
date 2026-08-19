"""Statistical aggregation and hypothesis testing for evaluation results."""

from __future__ import annotations

import math
from collections.abc import Callable

from .models import (
    ConditionStats,
    ConditionType,
    EvalResult,
    EvaluationReport,
    OutcomeStats,
    PairwiseComparison,
)

ResultMetric = Callable[[EvalResult], float]


def _overall_metric(result: EvalResult) -> float:
    return result.score_card.overall


def outcome_metric(result: EvalResult) -> float:
    """0/1 pass indicator — the pairwise metric for behavioral comparisons."""
    return 1.0 if result.outcome is not None and result.outcome.outcome_pass else 0.0


def compute_condition_stats(
    results: list[EvalResult],
    condition: ConditionType,
) -> ConditionStats:
    """Aggregate results for a single condition into summary statistics."""
    filtered = [r for r in results if r.condition == condition]
    n = len(filtered)
    if n == 0:
        return ConditionStats(condition=condition)

    overalls = [r.score_card.overall for r in filtered]
    file_ids = [r.score_card.file_identification for r in filtered]
    approaches = [r.score_card.approach_correctness for r in filtered]
    patterns = [r.score_card.pattern_awareness for r in filtered]
    pitfalls = [r.score_card.pitfall_avoidance for r in filtered]
    completeness_vals = [r.score_card.completeness for r in filtered]
    prompt_toks = [float(r.agent_response.usage.prompt_tokens) for r in filtered]
    completion_toks = [float(r.agent_response.usage.completion_tokens) for r in filtered]
    total_toks = [float(r.agent_response.usage.total_tokens) for r in filtered]
    durations = [r.agent_response.usage.duration_seconds for r in filtered]

    return ConditionStats(
        condition=condition,
        mean_overall=_mean(overalls),
        std_overall=_std(overalls),
        mean_file_identification=_mean(file_ids),
        mean_approach_correctness=_mean(approaches),
        mean_pattern_awareness=_mean(patterns),
        mean_pitfall_avoidance=_mean(pitfalls),
        mean_completeness=_mean(completeness_vals),
        mean_prompt_tokens=_mean(prompt_toks),
        mean_completion_tokens=_mean(completion_toks),
        mean_total_tokens=_mean(total_toks),
        mean_duration_seconds=_mean(durations),
        n_results=n,
    )


def compute_pairwise_comparisons(
    results: list[EvalResult],
    conditions: list[ConditionType],
    metric: ResultMetric | None = None,
) -> list[PairwiseComparison]:
    """Compute paired t-tests between all condition pairs.

    Pairs are matched on (scenario_id, run_number).  Uses scipy if
    available, otherwise falls back to a manual implementation.

    Args:
        metric: Value extracted from each result for comparison. Defaults to
            the judge's overall score; behavioral comparisons pass
            :func:`outcome_metric` (the 0/1 pass indicator).
    """
    comparisons: list[PairwiseComparison] = []

    for i, cond_a in enumerate(conditions):
        for cond_b in conditions[i + 1 :]:
            comparison = _paired_comparison(results, cond_a, cond_b, metric or _overall_metric)
            comparisons.append(comparison)

    return comparisons


def _paired_comparison(
    results: list[EvalResult],
    cond_a: ConditionType,
    cond_b: ConditionType,
    metric: ResultMetric = _overall_metric,
) -> PairwiseComparison:
    """Run a paired t-test between two conditions."""
    # Build lookup: (scenario_id, run_number) -> metric value
    scores_a: dict[tuple[str, int], float] = {}
    scores_b: dict[tuple[str, int], float] = {}

    for r in results:
        key = (r.scenario_id, r.run_number)
        if r.condition == cond_a:
            scores_a[key] = metric(r)
        elif r.condition == cond_b:
            scores_b[key] = metric(r)

    # Only use pairs where both conditions have a result
    common_keys = sorted(set(scores_a.keys()) & set(scores_b.keys()))
    n = len(common_keys)

    if n < 2:
        return PairwiseComparison(
            condition_a=cond_a,
            condition_b=cond_b,
            n_pairs=n,
        )

    diffs = [scores_a[k] - scores_b[k] for k in common_keys]

    try:
        return _scipy_paired_ttest(cond_a, cond_b, diffs, n)
    except ImportError:
        return _manual_paired_ttest(cond_a, cond_b, diffs, n)


def _scipy_paired_ttest(
    cond_a: ConditionType,
    cond_b: ConditionType,
    diffs: list[float],
    n: int,
) -> PairwiseComparison:
    """Paired t-test using scipy."""
    from scipy import stats

    t_stat, p_val = stats.ttest_1samp(diffs, 0.0)

    return PairwiseComparison(
        condition_a=cond_a,
        condition_b=cond_b,
        mean_diff=_mean(diffs),
        t_statistic=float(t_stat),
        p_value=float(p_val),
        n_pairs=n,
        significant=float(p_val) < 0.05,
    )


def _manual_paired_ttest(
    cond_a: ConditionType,
    cond_b: ConditionType,
    diffs: list[float],
    n: int,
) -> PairwiseComparison:
    """Manual paired t-test fallback when scipy is not installed."""
    mean_d = _mean(diffs)
    std_d = _std(diffs)

    if std_d == 0.0:
        # All differences are identical
        t_stat = float("inf") if mean_d != 0 else 0.0
        p_val = 0.0 if mean_d != 0 else 1.0
    else:
        se = std_d / math.sqrt(n)
        t_stat = mean_d / se
        # Two-tailed p-value approximation using the normal distribution
        # (adequate for n >= 20; conservative for smaller n)
        p_val = 2.0 * _normal_cdf(-abs(t_stat))

    return PairwiseComparison(
        condition_a=cond_a,
        condition_b=cond_b,
        mean_diff=mean_d,
        t_statistic=t_stat,
        p_value=p_val,
        n_pairs=n,
        significant=p_val < 0.05,
    )


def compute_outcome_stats(
    results: list[EvalResult],
    condition: ConditionType,
) -> OutcomeStats:
    """Aggregate behavioral outcomes and efficiency metrics for one condition.

    Only results carrying an ``outcome`` participate; ``k`` is derived from the
    highest run number observed. Efficiency means skip runs whose driver did
    not report the metric.
    """
    filtered = [r for r in results if r.condition == condition and r.outcome is not None]
    n = len(filtered)
    if n == 0:
        return OutcomeStats(condition=condition)

    passes = [r for r in filtered if r.outcome is not None and r.outcome.outcome_pass]

    by_scenario: dict[str, list[EvalResult]] = {}
    for r in filtered:
        by_scenario.setdefault(r.scenario_id, []).append(r)
    scenarios_with_pass = sum(
        1
        for runs in by_scenario.values()
        if any(r.outcome is not None and r.outcome.outcome_pass for r in runs)
    )

    def _traj_mean(attr: str) -> float:
        values = [
            float(v)
            for r in filtered
            if r.trajectory is not None and (v := getattr(r.trajectory, attr)) is not None
        ]
        return _mean(values)

    triggered = [
        r for r in filtered if r.trajectory is not None and r.trajectory.skill_triggered is not None
    ]
    trigger_rate = (
        _mean(
            [
                1.0 if r.trajectory is not None and r.trajectory.skill_triggered else 0.0
                for r in triggered
            ]
        )
        if triggered
        else 0.0
    )

    return OutcomeStats(
        condition=condition,
        n_results=n,
        k=max(r.run_number for r in filtered),
        pass_rate=round(len(passes) / n, 4),
        pass_at_k=round(scenarios_with_pass / len(by_scenario), 4),
        mean_turns=_traj_mean("num_turns"),
        mean_tool_calls=_traj_mean("num_tool_calls"),
        mean_errors=_traj_mean("num_errors"),
        mean_total_tokens=_traj_mean("total_tokens"),
        mean_wall_seconds=_traj_mean("wall_seconds"),
        skill_trigger_rate=trigger_rate,
    )


def outcome_pass_rate(report: EvaluationReport, condition: ConditionType) -> float | None:
    """Pass rate for one condition, or None when it has no behavioral results.

    The CI gate consumes this: with ``--fail-under-pass-rate`` unset the run
    stays advisory regardless of the value.
    """
    filtered = [r for r in report.results if r.condition == condition and r.outcome is not None]
    if not filtered:
        return None
    passed = sum(1 for r in filtered if r.outcome is not None and r.outcome.outcome_pass)
    return round(passed / len(filtered), 4)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return round(sum(values) / len(values), 4)


def _std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    m = sum(values) / len(values)
    variance = sum((x - m) ** 2 for x in values) / (len(values) - 1)
    return round(math.sqrt(variance), 4)


def _normal_cdf(x: float) -> float:
    """Approximate the standard normal CDF using the error function."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))
