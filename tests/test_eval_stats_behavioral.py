"""Tests for behavioral outcome statistics and the pairwise metric parameter."""

from dr_agents_tester.eval.models import (
    AgentResponse,
    ConditionType,
    EvalResult,
    EvaluationReport,
    LLMUsage,
    OutcomeResult,
    ScoreCard,
    TrajectoryMetrics,
)
from dr_agents_tester.eval.stats import (
    compute_outcome_stats,
    compute_pairwise_comparisons,
    outcome_metric,
    outcome_pass_rate,
)


def _result(
    condition: ConditionType,
    run_number: int,
    passed: bool | None,
    scenario_id: str = "s1",
    trajectory: TrajectoryMetrics | None = None,
) -> EvalResult:
    return EvalResult(
        scenario_id=scenario_id,
        condition=condition,
        run_number=run_number,
        agent_response=AgentResponse(plan_text="", usage=LLMUsage()),
        score_card=ScoreCard(),
        scoring_usage=LLMUsage(),
        outcome=OutcomeResult(outcome_pass=passed) if passed is not None else None,
        trajectory=trajectory,
    )


class TestComputeOutcomeStats:
    def test_pass_rate_and_pass_at_k(self) -> None:
        results = [
            # scenario s1: 2/3 pass
            _result(ConditionType.SKILL_PR, 1, True),
            _result(ConditionType.SKILL_PR, 2, False),
            _result(ConditionType.SKILL_PR, 3, True),
            # scenario s2: 0/3 pass
            _result(ConditionType.SKILL_PR, 1, False, scenario_id="s2"),
            _result(ConditionType.SKILL_PR, 2, False, scenario_id="s2"),
            _result(ConditionType.SKILL_PR, 3, False, scenario_id="s2"),
        ]
        stats = compute_outcome_stats(results, ConditionType.SKILL_PR)
        assert stats.n_results == 6
        assert stats.k == 3
        assert stats.pass_rate == round(2 / 6, 4)
        assert stats.pass_at_k == 0.5  # one of two scenarios has >=1 pass

    def test_efficiency_means_skip_unreported(self) -> None:
        results = [
            _result(
                ConditionType.SKILL_PR,
                1,
                True,
                trajectory=TrajectoryMetrics(
                    num_turns=4, total_tokens=1000, wall_seconds=60.0, skill_triggered=True
                ),
            ),
            _result(
                ConditionType.SKILL_PR,
                2,
                True,
                trajectory=TrajectoryMetrics(
                    num_turns=6, total_tokens=None, wall_seconds=120.0, skill_triggered=False
                ),
            ),
        ]
        stats = compute_outcome_stats(results, ConditionType.SKILL_PR)
        assert stats.mean_turns == 5.0
        assert stats.mean_total_tokens == 1000.0  # None skipped, not zero-filled
        assert stats.mean_wall_seconds == 90.0
        assert stats.skill_trigger_rate == 0.5

    def test_expected_skill_trigger_rate(self) -> None:
        results = [
            _result(
                ConditionType.SKILL_PR,
                1,
                True,
                trajectory=TrajectoryMetrics(skill_triggered=True, skill_triggered_expected=True),
            ),
            # A collision run: some skill fired, but not one under test.
            _result(
                ConditionType.SKILL_PR,
                2,
                True,
                trajectory=TrajectoryMetrics(skill_triggered=True, skill_triggered_expected=False),
            ),
            # Driver without trigger detection: excluded from both rates.
            _result(
                ConditionType.SKILL_PR,
                3,
                True,
                trajectory=TrajectoryMetrics(skill_triggered=None, skill_triggered_expected=None),
            ),
        ]
        stats = compute_outcome_stats(results, ConditionType.SKILL_PR)
        assert stats.skill_trigger_rate == 1.0
        assert stats.expected_skill_trigger_rate == 0.5

    def test_empty_condition(self) -> None:
        stats = compute_outcome_stats([], ConditionType.NO_SKILL)
        assert stats.n_results == 0
        assert stats.pass_rate == 0.0
        assert stats.expected_skill_trigger_rate == 0.0

    def test_plan_results_excluded(self) -> None:
        stats = compute_outcome_stats(
            [_result(ConditionType.SKILL_PR, 1, None)], ConditionType.SKILL_PR
        )
        assert stats.n_results == 0


class TestOutcomeMetricPairwise:
    def test_pairwise_on_outcome_indicator(self) -> None:
        results = []
        # skill_pr passes all 5 paired runs; no_skill fails all 5
        for run in range(1, 6):
            results.append(_result(ConditionType.SKILL_PR, run, True))
            results.append(_result(ConditionType.NO_SKILL, run, False))

        comparisons = compute_pairwise_comparisons(
            results,
            [ConditionType.SKILL_PR, ConditionType.NO_SKILL],
            metric=outcome_metric,
        )
        assert len(comparisons) == 1
        pc = comparisons[0]
        assert pc.mean_diff == 1.0
        assert pc.n_pairs == 5

    def test_default_metric_unchanged(self) -> None:
        results = []
        for run in range(1, 4):
            r = _result(ConditionType.NO_CONTEXT, run, None)
            r.score_card.overall = 0.5
            results.append(r)
            r2 = _result(ConditionType.GENERATED, run, None)
            r2.score_card.overall = 0.7
            results.append(r2)
        comparisons = compute_pairwise_comparisons(
            results, [ConditionType.NO_CONTEXT, ConditionType.GENERATED]
        )
        assert round(comparisons[0].mean_diff, 4) == -0.2


class TestOutcomePassRate:
    def test_rate(self) -> None:
        report = EvaluationReport(
            results=[
                _result(ConditionType.SKILL_PR, 1, True),
                _result(ConditionType.SKILL_PR, 2, False),
            ]
        )
        assert outcome_pass_rate(report, ConditionType.SKILL_PR) == 0.5

    def test_none_without_behavioral_results(self) -> None:
        report = EvaluationReport(results=[_result(ConditionType.NO_CONTEXT, 1, None)])
        assert outcome_pass_rate(report, ConditionType.NO_CONTEXT) is None
