"""Data models for the evaluation framework (AGENTS.md plan evals + skill behavioral evals)."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from ..llm import LLMUsage

__all__ = ["LLMUsage"]  # re-export for convenience


class Difficulty(str, Enum):
    """Scenario difficulty level."""

    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"
    EXPERT = "expert"


class ConditionType(str, Enum):
    """Which evaluation condition is being tested.

    The first three are AGENTS.md plan-eval conditions; the skill variants
    drive behavioral evaluations (design doc §4.4).
    """

    NO_CONTEXT = "no_context"
    GENERATED = "generated"
    REFINED = "refined"
    NO_SKILL = "no_skill"
    SKILL_MAIN = "skill_main"
    SKILL_PR = "skill_pr"


@dataclass
class ScenarioBase:
    """Fields shared by every scenario kind."""

    id: str
    name: str
    difficulty: Difficulty
    prompt: str


@dataclass
class Scenario(ScenarioBase):
    """A plan-eval scenario loaded from YAML (schema v1, ``kind: plan``)."""

    expected_files: list[str]
    expected_approach: str
    expected_patterns: list[str]
    common_pitfalls: list[str]
    acceptance_criteria: list[str]


@dataclass
class CheckSpec:
    """One entry of a behavioral scenario's ``success_checks`` list."""

    type: str
    params: dict[str, object] = field(default_factory=dict)


@dataclass
class FixtureSpec:
    """A file copied into the sandbox workspace before the agent runs.

    ``source`` is relative to the scenario file's directory; ``dest`` is the
    path inside the workspace. A bare string in YAML means source == dest.
    """

    source: str
    dest: str


@dataclass
class BehavioralScenario(ScenarioBase):
    """A behavioral scenario loaded from YAML (schema v2, ``kind: behavioral``)."""

    skills_under_test: list[str]
    success_checks: list[CheckSpec]
    fixtures: list[FixtureSpec] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    rubric: str = ""
    common_pitfalls: list[str] = field(default_factory=list)
    timeout_minutes: int = 30
    source_dir: Path | None = None


@dataclass
class EvalCondition:
    """Configuration for one evaluation condition."""

    condition_type: ConditionType
    agents_md_content: str | None = None
    skills_source: Path | None = None


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
class CheckResult:
    """Outcome of a single success check."""

    check_type: str
    passed: bool
    evidence: str = ""
    error: str | None = None
    duration_seconds: float = 0.0


@dataclass
class OutcomeResult:
    """Hard pass/fail verdict from a behavioral run's success checks."""

    outcome_pass: bool
    checks: list[CheckResult] = field(default_factory=list)


@dataclass
class TrajectoryMetrics:
    """Efficiency metrics derived from a normalized agent trajectory.

    ``None`` means "not available from this driver" — coarser drivers report
    honestly rather than faking zeros.
    """

    wall_seconds: float | None = None
    total_tokens: int | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    cache_read_tokens: int | None = None
    cache_write_tokens: int | None = None
    cost: float | None = None
    num_turns: int | None = None
    num_tool_calls: int | None = None
    num_errors: int | None = None
    num_retries: int | None = None
    skill_triggered: bool | None = None
    skills_used: list[str] = field(default_factory=list)
    skill_first_turn: int | None = None
    num_unknown_events: int = 0

    def available(self) -> list[str]:
        """Names of metrics this driver actually populated."""
        return [
            name
            for name in (
                "wall_seconds",
                "total_tokens",
                "input_tokens",
                "output_tokens",
                "cache_read_tokens",
                "cache_write_tokens",
                "cost",
                "num_turns",
                "num_tool_calls",
                "num_errors",
                "num_retries",
                "skill_triggered",
                "skill_first_turn",
            )
            if getattr(self, name) is not None
        ]


@dataclass
class EvalResult:
    """Result of a single scenario + condition + run.

    The optional fields are populated only by behavioral (agent-backend) runs;
    they stay ``None`` for plan-eval results and are omitted from serialized
    JSON when absent.
    """

    scenario_id: str
    condition: ConditionType
    run_number: int
    agent_response: AgentResponse
    score_card: ScoreCard
    scoring_usage: LLMUsage
    outcome: OutcomeResult | None = None
    trajectory: TrajectoryMetrics | None = None
    transcript_path: str | None = None
    run_id: str | None = None


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
class OutcomeStats:
    """Aggregated behavioral-outcome statistics for one condition."""

    condition: ConditionType
    n_results: int = 0
    k: int = 0
    pass_rate: float = 0.0
    pass_at_k: float = 0.0
    mean_turns: float = 0.0
    mean_tool_calls: float = 0.0
    mean_errors: float = 0.0
    mean_total_tokens: float = 0.0
    mean_wall_seconds: float = 0.0
    skill_trigger_rate: float = 0.0


@dataclass
class EvaluationReport:
    """Complete evaluation report with all results and statistics."""

    results: list[EvalResult] = field(default_factory=list)
    condition_stats: list[ConditionStats] = field(default_factory=list)
    pairwise_comparisons: list[PairwiseComparison] = field(default_factory=list)
    n_scenarios: int = 0
    n_runs: int = 0
    conditions_tested: list[ConditionType] = field(default_factory=list)
    outcome_stats: list[OutcomeStats] = field(default_factory=list)
    backend_name: str = "plan"
