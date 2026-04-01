"""Tests for eval/stats.py statistical aggregation and t-tests."""

from dr_agents_tester.eval.models import (
    AgentResponse,
    ConditionType,
    EvalResult,
    LLMUsage,
    ScoreCard,
)
from dr_agents_tester.eval.stats import (
    compute_condition_stats,
    compute_pairwise_comparisons,
)


def _make_result(
    scenario_id: str,
    condition: ConditionType,
    run: int,
    overall: float,
    prompt_tokens: int = 100,
    duration: float = 1.0,
) -> EvalResult:
    return EvalResult(
        scenario_id=scenario_id,
        condition=condition,
        run_number=run,
        agent_response=AgentResponse(
            plan_text="plan",
            usage=LLMUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=50,
                total_tokens=prompt_tokens + 50,
                duration_seconds=duration,
            ),
        ),
        score_card=ScoreCard(
            file_identification=overall,
            approach_correctness=overall,
            pattern_awareness=overall,
            pitfall_avoidance=overall,
            completeness=overall,
            overall=overall,
        ),
        scoring_usage=LLMUsage(),
    )


class TestComputeConditionStats:
    def test_empty_results(self) -> None:
        stats = compute_condition_stats([], ConditionType.NO_CONTEXT)
        assert stats.n_results == 0
        assert stats.mean_overall == 0.0

    def test_single_result(self) -> None:
        results = [_make_result("s1", ConditionType.NO_CONTEXT, 1, 0.8)]
        stats = compute_condition_stats(results, ConditionType.NO_CONTEXT)
        assert stats.n_results == 1
        assert stats.mean_overall == 0.8
        assert stats.std_overall == 0.0  # single value

    def test_multiple_results(self) -> None:
        results = [
            _make_result("s1", ConditionType.NO_CONTEXT, 1, 0.6),
            _make_result("s1", ConditionType.NO_CONTEXT, 2, 0.8),
            _make_result("s2", ConditionType.NO_CONTEXT, 1, 1.0),
        ]
        stats = compute_condition_stats(results, ConditionType.NO_CONTEXT)
        assert stats.n_results == 3
        assert abs(stats.mean_overall - 0.8) < 0.001

    def test_filters_by_condition(self) -> None:
        results = [
            _make_result("s1", ConditionType.NO_CONTEXT, 1, 0.5),
            _make_result("s1", ConditionType.GENERATED, 1, 0.9),
        ]
        stats_nc = compute_condition_stats(results, ConditionType.NO_CONTEXT)
        stats_gen = compute_condition_stats(results, ConditionType.GENERATED)
        assert stats_nc.n_results == 1
        assert stats_gen.n_results == 1
        assert stats_nc.mean_overall == 0.5
        assert stats_gen.mean_overall == 0.9

    def test_token_stats(self) -> None:
        results = [
            _make_result("s1", ConditionType.NO_CONTEXT, 1, 0.8, prompt_tokens=200),
            _make_result("s1", ConditionType.NO_CONTEXT, 2, 0.8, prompt_tokens=300),
        ]
        stats = compute_condition_stats(results, ConditionType.NO_CONTEXT)
        assert stats.mean_prompt_tokens == 250.0


class TestComputePairwiseComparisons:
    def test_no_conditions(self) -> None:
        comps = compute_pairwise_comparisons([], [])
        assert comps == []

    def test_single_condition(self) -> None:
        comps = compute_pairwise_comparisons([], [ConditionType.NO_CONTEXT])
        assert comps == []

    def test_two_conditions(self) -> None:
        results = [
            _make_result("s1", ConditionType.NO_CONTEXT, 1, 0.5),
            _make_result("s1", ConditionType.GENERATED, 1, 0.8),
            _make_result("s2", ConditionType.NO_CONTEXT, 1, 0.6),
            _make_result("s2", ConditionType.GENERATED, 1, 0.9),
        ]
        comps = compute_pairwise_comparisons(
            results, [ConditionType.NO_CONTEXT, ConditionType.GENERATED]
        )
        assert len(comps) == 1
        assert comps[0].condition_a == ConditionType.NO_CONTEXT
        assert comps[0].condition_b == ConditionType.GENERATED
        assert comps[0].n_pairs == 2
        assert comps[0].mean_diff < 0  # NO_CONTEXT scores lower

    def test_three_conditions(self) -> None:
        conditions = [ConditionType.NO_CONTEXT, ConditionType.GENERATED, ConditionType.REFINED]
        results = []
        for cond, score in zip(conditions, [0.5, 0.7, 0.9]):
            results.append(_make_result("s1", cond, 1, score))
            results.append(_make_result("s2", cond, 1, score + 0.05))

        comps = compute_pairwise_comparisons(results, conditions)
        assert len(comps) == 3  # 3 pairs from 3 conditions

    def test_insufficient_pairs(self) -> None:
        results = [
            _make_result("s1", ConditionType.NO_CONTEXT, 1, 0.5),
            _make_result("s1", ConditionType.GENERATED, 1, 0.8),
        ]
        comps = compute_pairwise_comparisons(
            results, [ConditionType.NO_CONTEXT, ConditionType.GENERATED]
        )
        assert len(comps) == 1
        assert comps[0].n_pairs == 1  # only 1 pair, not enough for t-test

    def test_significant_difference(self) -> None:
        # Create many paired results with consistent difference
        results = []
        for i in range(20):
            results.append(_make_result(f"s{i}", ConditionType.NO_CONTEXT, 1, 0.4))
            results.append(_make_result(f"s{i}", ConditionType.GENERATED, 1, 0.8))

        comps = compute_pairwise_comparisons(
            results, [ConditionType.NO_CONTEXT, ConditionType.GENERATED]
        )
        assert len(comps) == 1
        assert comps[0].n_pairs == 20
        assert comps[0].significant  # large consistent difference should be significant
