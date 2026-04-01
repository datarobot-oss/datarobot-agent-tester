"""Score weighting and computation for the evaluation framework."""

from __future__ import annotations

from .models import ScoreCard

SCORE_WEIGHTS: dict[str, float] = {
    "file_identification": 0.25,
    "approach_correctness": 0.30,
    "pattern_awareness": 0.15,
    "pitfall_avoidance": 0.15,
    "completeness": 0.15,
}


def compute_overall_score(card: ScoreCard) -> float:
    """Compute the weighted overall score from individual dimension scores."""
    return round(
        card.file_identification * SCORE_WEIGHTS["file_identification"]
        + card.approach_correctness * SCORE_WEIGHTS["approach_correctness"]
        + card.pattern_awareness * SCORE_WEIGHTS["pattern_awareness"]
        + card.pitfall_avoidance * SCORE_WEIGHTS["pitfall_avoidance"]
        + card.completeness * SCORE_WEIGHTS["completeness"],
        4,
    )
