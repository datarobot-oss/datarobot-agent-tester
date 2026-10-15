"""PlanBackend: the original single-LLM-call plan evaluation, unchanged in behavior.

The agent LLM writes an implementation plan against a static repo tree; a
second (scoring) LLM grades the plan into a :class:`ScoreCard`.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from ...config import Config
from ...llm import LLMUsage, call_llm_with_usage
from ..models import (
    AgentResponse,
    Difficulty,
    EvalCondition,
    EvalResult,
    Scenario,
    ScenarioBase,
)
from ..prompts import (
    build_agent_prompt,
    build_scoring_prompt,
    parse_agent_response,
    parse_score_card_json,
)
from ..scenarios import load_scenarios
from .base import RunContext

LLMCallable = Callable[[str, str, Config], tuple[str, LLMUsage]]


class PlanBackend:
    """Evaluate scenarios by asking one LLM for a plan and another to score it."""

    name = "plan"

    def __init__(
        self,
        config: Config,
        repo_tree: str,
        llm: LLMCallable | None = None,
    ) -> None:
        self.config = config
        self.repo_tree = repo_tree
        # Resolved at call time (not bound here) so tests can patch the
        # module-level call_llm_with_usage.
        self._llm_override = llm

    def load_scenarios(
        self, scenarios_dir: Path, difficulty_filter: Difficulty | None
    ) -> list[ScenarioBase]:
        return list(load_scenarios(scenarios_dir, difficulty_filter))

    def execute(
        self, scenario: ScenarioBase, condition: EvalCondition, ctx: RunContext
    ) -> EvalResult:
        if not isinstance(scenario, Scenario):
            raise TypeError(f"PlanBackend expects plan scenarios, got {type(scenario).__name__}")

        llm = self._llm_override if self._llm_override is not None else call_llm_with_usage

        # Step 1: Ask the agent for an implementation plan
        agent_prompt = build_agent_prompt(
            scenario=scenario,
            repo_tree=self.repo_tree,
            agents_md_content=condition.agents_md_content,
        )
        agent_text, agent_usage = llm(agent_prompt, self.config.model, self.config)
        plan_text = parse_agent_response(agent_text)

        # Step 2: Score the plan
        scoring_prompt = build_scoring_prompt(
            scenario=scenario,
            plan_text=plan_text,
        )
        scoring_text, scoring_usage = llm(scoring_prompt, self.config.test_model, self.config)
        score_card = parse_score_card_json(scoring_text)

        return EvalResult(
            scenario_id=scenario.id,
            condition=condition.condition_type,
            run_number=ctx.run_number,
            agent_response=AgentResponse(plan_text=plan_text, usage=agent_usage),
            score_card=score_card,
            scoring_usage=scoring_usage,
        )
