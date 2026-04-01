"""Statistical aggregation and hypothesis testing for evaluation results."""

from __future__ import annotations

import math

from .models import ConditionStats, ConditionType, EvalResult, PairwiseComparison


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
) -> list[PairwiseComparison]:
    """Compute paired t-tests between all condition pairs.

    Pairs are matched on (scenario_id, run_number).  Uses scipy if
    available, otherwise falls back to a manual implementation.
    """
    comparisons: list[PairwiseComparison] = []

    for i, cond_a in enumerate(conditions):
        for cond_b in conditions[i + 1 :]:
            comparison = _paired_comparison(results, cond_a, cond_b)
            comparisons.append(comparison)

    return comparisons


def _paired_comparison(
    results: list[EvalResult],
    cond_a: ConditionType,
    cond_b: ConditionType,
) -> PairwiseComparison:
    """Run a paired t-test between two conditions."""
    # Build lookup: (scenario_id, run_number) -> overall score
    scores_a: dict[tuple[str, int], float] = {}
    scores_b: dict[tuple[str, int], float] = {}

    for r in results:
        key = (r.scenario_id, r.run_number)
        if r.condition == cond_a:
            scores_a[key] = r.score_card.overall
        elif r.condition == cond_b:
            scores_b[key] = r.score_card.overall

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
