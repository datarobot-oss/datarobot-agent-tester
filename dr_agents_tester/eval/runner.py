"""Evaluation runner that orchestrates scenario execution across conditions."""

from __future__ import annotations

import sys
from pathlib import Path

from ..config import Config
from ..llm import call_llm_with_usage
from .models import (
    AgentResponse,
    Difficulty,
    EvalCondition,
    EvalResult,
    EvaluationReport,
    Scenario,
)
from .prompts import (
    build_agent_prompt,
    build_scoring_prompt,
    parse_agent_response,
    parse_score_card_json,
)
from .report import generate_markdown_report
from .scenarios import load_scenarios
from .stats import compute_condition_stats, compute_pairwise_comparisons


class Evaluator:
    """Orchestrates the AGENTS.md evaluation across scenarios and conditions.

    Args:
        config: LLM gateway configuration.
        scenarios_dir: Directory containing scenario YAML files.
        repo_tree_path: Path to the repo file tree text file.
        conditions: List of conditions to evaluate.
        n_runs: Number of runs per scenario/condition pair.
        difficulty_filter: If set, only run scenarios of this difficulty.
    """

    def __init__(
        self,
        config: Config,
        scenarios_dir: Path,
        repo_tree_path: Path,
        conditions: list[EvalCondition],
        n_runs: int = 5,
        difficulty_filter: Difficulty | None = None,
    ) -> None:
        self.config = config
        self.scenarios_dir = scenarios_dir
        self.repo_tree = repo_tree_path.read_text().strip()
        self.conditions = conditions
        self.n_runs = n_runs
        self.difficulty_filter = difficulty_filter
        self.results: list[EvalResult] = []

    def run(self) -> EvaluationReport:
        """Execute the full evaluation and return a report.

        Iterates over all scenarios × conditions × runs, calling the agent
        LLM for each combination and then scoring the response.
        """
        scenarios = load_scenarios(self.scenarios_dir, self.difficulty_filter)

        if not scenarios:
            print("No scenarios found.", file=sys.stderr)
            return EvaluationReport()

        total = len(scenarios) * len(self.conditions) * self.n_runs
        completed = 0

        for scenario in scenarios:
            for condition in self.conditions:
                for run_num in range(1, self.n_runs + 1):
                    completed += 1
                    label = (
                        f"[{completed}/{total}] {scenario.id} "
                        f"| {condition.condition_type.value} | run {run_num}"
                    )
                    print(f"  {label} ...", flush=True)

                    result = self._run_single(scenario, condition, run_num)
                    self.results.append(result)

                    print(
                        f"  {label} -> overall={result.score_card.overall:.3f}",
                        flush=True,
                    )

        return self._build_report(scenarios)

    def _run_single(
        self,
        scenario: Scenario,
        condition: EvalCondition,
        run_number: int,
    ) -> EvalResult:
        """Execute a single scenario + condition + run."""
        # Step 1: Ask the agent for an implementation plan
        agent_prompt = build_agent_prompt(
            scenario=scenario,
            repo_tree=self.repo_tree,
            agents_md_content=condition.agents_md_content,
        )
        agent_text, agent_usage = call_llm_with_usage(
            prompt=agent_prompt,
            model=self.config.model,
            config=self.config,
        )
        plan_text = parse_agent_response(agent_text)

        # Step 2: Score the plan
        scoring_prompt = build_scoring_prompt(
            scenario=scenario,
            plan_text=plan_text,
        )
        scoring_text, scoring_usage = call_llm_with_usage(
            prompt=scoring_prompt,
            model=self.config.test_model,
            config=self.config,
        )
        score_card = parse_score_card_json(scoring_text)

        return EvalResult(
            scenario_id=scenario.id,
            condition=condition.condition_type,
            run_number=run_number,
            agent_response=AgentResponse(plan_text=plan_text, usage=agent_usage),
            score_card=score_card,
            scoring_usage=scoring_usage,
        )

    def _build_report(self, scenarios: list[Scenario]) -> EvaluationReport:
        """Aggregate results into a full evaluation report."""
        condition_types = [c.condition_type for c in self.conditions]

        stats = [compute_condition_stats(self.results, ct) for ct in condition_types]
        comparisons = compute_pairwise_comparisons(self.results, condition_types)

        return EvaluationReport(
            results=self.results,
            condition_stats=stats,
            pairwise_comparisons=comparisons,
            n_scenarios=len(scenarios),
            n_runs=self.n_runs,
            conditions_tested=condition_types,
        )

    def save_report(self, report: EvaluationReport, output_dir: Path) -> tuple[Path, Path]:
        """Save the report as both Markdown and JSON.

        Returns:
            Tuple of (markdown_path, json_path).
        """
        import json

        from .report import report_to_dict

        output_dir.mkdir(parents=True, exist_ok=True)

        md_path = output_dir / "eval_report.md"
        md_path.write_text(generate_markdown_report(report))

        json_path = output_dir / "eval_results.json"
        json_path.write_text(json.dumps(report_to_dict(report), indent=2))

        return md_path, json_path
