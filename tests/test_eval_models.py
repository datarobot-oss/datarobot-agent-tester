"""Tests for eval/models.py dataclass construction."""

from dr_agents_tester.eval.models import (
    AgentResponse,
    ConditionStats,
    ConditionType,
    Difficulty,
    EvalCondition,
    EvalResult,
    EvaluationReport,
    LLMUsage,
    PairwiseComparison,
    Scenario,
    ScoreCard,
)


class TestDifficulty:
    def test_values(self) -> None:
        assert Difficulty.EASY == "easy"
        assert Difficulty.MEDIUM == "medium"
        assert Difficulty.HARD == "hard"
        assert Difficulty.EXPERT == "expert"

    def test_from_string(self) -> None:
        assert Difficulty("easy") == Difficulty.EASY


class TestConditionType:
    def test_values(self) -> None:
        assert ConditionType.NO_CONTEXT == "no_context"
        assert ConditionType.GENERATED == "generated"
        assert ConditionType.REFINED == "refined"


class TestScenario:
    def test_construction(self) -> None:
        s = Scenario(
            id="test-1",
            name="Test scenario",
            difficulty=Difficulty.EASY,
            prompt="Do something",
            expected_files=["a.py", "b.py"],
            expected_approach="The approach",
            expected_patterns=["pattern1"],
            common_pitfalls=["pitfall1"],
            acceptance_criteria=["criterion1"],
        )
        assert s.id == "test-1"
        assert s.difficulty == Difficulty.EASY
        assert len(s.expected_files) == 2


class TestEvalCondition:
    def test_no_context(self) -> None:
        c = EvalCondition(condition_type=ConditionType.NO_CONTEXT)
        assert c.agents_md_content is None

    def test_with_content(self) -> None:
        c = EvalCondition(
            condition_type=ConditionType.GENERATED,
            agents_md_content="# AGENTS.md content",
        )
        assert c.agents_md_content == "# AGENTS.md content"


class TestLLMUsage:
    def test_defaults(self) -> None:
        u = LLMUsage()
        assert u.prompt_tokens == 0
        assert u.completion_tokens == 0
        assert u.total_tokens == 0
        assert u.duration_seconds == 0.0


class TestScoreCard:
    def test_defaults(self) -> None:
        sc = ScoreCard()
        assert sc.overall == 0.0
        assert sc.rationale == ""

    def test_with_scores(self) -> None:
        sc = ScoreCard(
            file_identification=0.8,
            approach_correctness=0.9,
            pattern_awareness=0.7,
            pitfall_avoidance=0.6,
            completeness=0.85,
            overall=0.8,
            rationale="Good plan",
        )
        assert sc.file_identification == 0.8


class TestEvalResult:
    def test_construction(self) -> None:
        r = EvalResult(
            scenario_id="test-1",
            condition=ConditionType.NO_CONTEXT,
            run_number=1,
            agent_response=AgentResponse(plan_text="plan", usage=LLMUsage()),
            score_card=ScoreCard(),
            scoring_usage=LLMUsage(),
        )
        assert r.scenario_id == "test-1"
        assert r.run_number == 1


class TestConditionStats:
    def test_defaults(self) -> None:
        cs = ConditionStats(condition=ConditionType.NO_CONTEXT)
        assert cs.n_results == 0
        assert cs.mean_overall == 0.0


class TestPairwiseComparison:
    def test_defaults(self) -> None:
        pc = PairwiseComparison(
            condition_a=ConditionType.NO_CONTEXT,
            condition_b=ConditionType.GENERATED,
        )
        assert pc.p_value == 1.0
        assert not pc.significant


class TestEvaluationReport:
    def test_empty_report(self) -> None:
        report = EvaluationReport()
        assert report.results == []
        assert report.n_scenarios == 0
        assert report.conditions_tested == []
