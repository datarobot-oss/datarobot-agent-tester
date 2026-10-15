"""Evaluation runner that orchestrates scenario execution across conditions."""

from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path

from ..config import Config
from .backends import ExecutionBackend, PlanBackend, RunContext, make_run_id
from .models import (
    Difficulty,
    EvalCondition,
    EvalResult,
    EvaluationReport,
    ScenarioBase,
)
from .report import generate_markdown_report
from .stats import (
    compute_condition_stats,
    compute_outcome_stats,
    compute_pairwise_comparisons,
    outcome_metric,
)


class Evaluator:
    """Orchestrates an evaluation across scenarios and conditions.

    Args:
        config: LLM gateway configuration.
        scenarios_dir: Directory (or sequence of directories) containing
            scenario YAML files; scenario ids must be unique across all of
            them.
        repo_tree_path: Path to the repo file tree text file (required only
            for the default plan backend).
        conditions: List of conditions to evaluate.
        n_runs: Number of runs per scenario/condition pair.
        difficulty_filter: If set, only run scenarios of this difficulty.
        backend: Execution backend; defaults to :class:`PlanBackend` built
            from ``repo_tree_path`` (the original AGENTS.md plan evaluation).
        run_prefix: Optional prefix for generated run ids (used by CI to make
            created resources traceable to a workflow run).
    """

    def __init__(
        self,
        config: Config,
        scenarios_dir: Path | Sequence[Path],
        repo_tree_path: Path | None = None,
        conditions: list[EvalCondition] | None = None,
        n_runs: int = 5,
        difficulty_filter: Difficulty | None = None,
        backend: ExecutionBackend | None = None,
        run_prefix: str | None = None,
    ) -> None:
        if backend is None:
            if repo_tree_path is None:
                raise ValueError("repo_tree_path is required for the default plan backend")
            backend = PlanBackend(config=config, repo_tree=repo_tree_path.read_text().strip())
        self.backend = backend
        self.config = config
        self.scenarios_dirs: list[Path] = (
            [scenarios_dir] if isinstance(scenarios_dir, Path) else list(scenarios_dir)
        )
        self.conditions = conditions or []
        self.n_runs = n_runs
        self.difficulty_filter = difficulty_filter
        self.run_prefix = run_prefix
        self.results: list[EvalResult] = []

    def run(self) -> EvaluationReport:
        """Execute the full evaluation and return a report.

        Iterates over all scenarios × conditions × runs, delegating each cell
        to the execution backend.
        """
        scenarios: list[ScenarioBase] = []
        for directory in self.scenarios_dirs:
            scenarios.extend(self.backend.load_scenarios(directory, self.difficulty_filter))

        seen: dict[str, int] = {}
        for s in scenarios:
            seen[s.id] = seen.get(s.id, 0) + 1
        duplicates = sorted(sid for sid, n in seen.items() if n > 1)
        if duplicates:
            raise ValueError(
                f"Duplicate scenario id(s) across --scenarios directories: {', '.join(duplicates)}"
            )

        if not scenarios:
            print("No scenarios found.", file=sys.stderr)
            return EvaluationReport(backend_name=self.backend.name)

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

                    ctx = RunContext(
                        run_number=run_num,
                        run_id=make_run_id(scenario, condition, run_num, self.run_prefix),
                    )
                    result = self.backend.execute(scenario, condition, ctx)
                    self.results.append(result)

                    if result.outcome is not None:
                        verdict = "PASS" if result.outcome.outcome_pass else "FAIL"
                        print(f"  {label} -> outcome={verdict}", flush=True)
                    else:
                        print(
                            f"  {label} -> overall={result.score_card.overall:.3f}",
                            flush=True,
                        )

        return self._build_report(scenarios)

    def _build_report(self, scenarios: list[ScenarioBase]) -> EvaluationReport:
        """Aggregate results into a full evaluation report."""
        condition_types = [c.condition_type for c in self.conditions]

        stats = [compute_condition_stats(self.results, ct) for ct in condition_types]

        has_outcomes = any(r.outcome is not None for r in self.results)
        comparisons = compute_pairwise_comparisons(
            self.results,
            condition_types,
            metric=outcome_metric if has_outcomes else None,
        )
        outcome_stats = (
            [compute_outcome_stats(self.results, ct) for ct in condition_types]
            if has_outcomes
            else []
        )

        return EvaluationReport(
            results=self.results,
            condition_stats=stats,
            pairwise_comparisons=comparisons,
            n_scenarios=len(scenarios),
            n_runs=self.n_runs,
            conditions_tested=condition_types,
            outcome_stats=outcome_stats,
            backend_name=self.backend.name,
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
