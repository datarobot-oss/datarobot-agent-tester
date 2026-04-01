"""Tests for eval/scoring.py score weights and computation."""

from dr_agents_tester.eval.models import ScoreCard
from dr_agents_tester.eval.scoring import SCORE_WEIGHTS, compute_overall_score


class TestScoreWeights:
    def test_weights_sum_to_one(self) -> None:
        total = sum(SCORE_WEIGHTS.values())
        assert abs(total - 1.0) < 1e-9

    def test_all_dimensions_present(self) -> None:
        expected = {
            "file_identification",
            "approach_correctness",
            "pattern_awareness",
            "pitfall_avoidance",
            "completeness",
        }
        assert set(SCORE_WEIGHTS.keys()) == expected


class TestComputeOverallScore:
    def test_all_ones(self) -> None:
        card = ScoreCard(
            file_identification=1.0,
            approach_correctness=1.0,
            pattern_awareness=1.0,
            pitfall_avoidance=1.0,
            completeness=1.0,
        )
        assert compute_overall_score(card) == 1.0

    def test_all_zeros(self) -> None:
        card = ScoreCard()
        assert compute_overall_score(card) == 0.0

    def test_weighted_calculation(self) -> None:
        card = ScoreCard(
            file_identification=0.8,
            approach_correctness=0.6,
            pattern_awareness=0.4,
            pitfall_avoidance=0.2,
            completeness=0.0,
        )
        expected = (
            0.8 * 0.25 + 0.6 * 0.30 + 0.4 * 0.15 + 0.2 * 0.15 + 0.0 * 0.15
        )
        assert abs(compute_overall_score(card) - round(expected, 4)) < 1e-9

    def test_approach_has_highest_weight(self) -> None:
        # A perfect approach score alone should contribute more than any other
        card_approach = ScoreCard(approach_correctness=1.0)
        card_files = ScoreCard(file_identification=1.0)
        assert compute_overall_score(card_approach) > compute_overall_score(card_files)
