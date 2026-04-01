"""Data models for the AGENTS.md evaluation framework."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from ..llm import LLMUsage

__all__ = ["LLMUsage"]  # re-export for convenience


class Difficulty(str, Enum):
    """Scenario difficulty level."""

    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"
    EXPERT = "expert"


class ConditionType(str, Enum):
    """Which AGENTS.md condition is being tested."""

    NO_CONTEXT = "no_context"
    GENERATED = "generated"
    REFINED = "refined"


@dataclass
class Scenario:
    """A single evaluation scenario loaded from YAML."""

    id: str
    name: str
    difficulty: Difficulty
    prompt: str
    expected_files: list[str]
    expected_approach: str
    expected_patterns: list[str]
    common_pitfalls: list[str]
    acceptance_criteria: list[str]


@dataclass
class EvalCondition:
    """Configuration for one evaluation condition."""

    condition_type: ConditionType
    agents_md_content: str | None = None


@dataclass
class AgentResponse:
    """Raw response from the agent LLM."""

    plan_text: str
    usage: LLMUsage


@dataclass
class ScoreCard:
    """Scores for a single evaluation across all dimensions."""

    file_identification: float = 0.0
    approach_correctness: float = 0.0
    pattern_awareness: float = 0.0
    pitfall_avoidance: float = 0.0
    completeness: float = 0.0
    overall: float = 0.0
    rationale: str = ""


@dataclass
class EvalResult:
    """Result of a single scenario + condition + run."""

    scenario_id: str
    condition: ConditionType
    run_number: int
    agent_response: AgentResponse
    score_card: ScoreCard
    scoring_usage: LLMUsage


@dataclass
class ConditionStats:
    """Aggregated statistics for one condition across all scenarios/runs."""

    condition: ConditionType
    mean_overall: float = 0.0
    std_overall: float = 0.0
    mean_file_identification: float = 0.0
    mean_approach_correctness: float = 0.0
    mean_pattern_awareness: float = 0.0
    mean_pitfall_avoidance: float = 0.0
    mean_completeness: float = 0.0
    mean_prompt_tokens: float = 0.0
    mean_completion_tokens: float = 0.0
    mean_total_tokens: float = 0.0
    mean_duration_seconds: float = 0.0
    n_results: int = 0


@dataclass
class PairwiseComparison:
    """Statistical comparison between two conditions."""

    condition_a: ConditionType
    condition_b: ConditionType
    mean_diff: float = 0.0
    t_statistic: float = 0.0
    p_value: float = 1.0
    n_pairs: int = 0
    significant: bool = False


@dataclass
class EvaluationReport:
    """Complete evaluation report with all results and statistics."""

    results: list[EvalResult] = field(default_factory=list)
    condition_stats: list[ConditionStats] = field(default_factory=list)
    pairwise_comparisons: list[PairwiseComparison] = field(default_factory=list)
    n_scenarios: int = 0
    n_runs: int = 0
    conditions_tested: list[ConditionType] = field(default_factory=list)
