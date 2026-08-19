"""AgentBackend: drive a real coding agent in a sandboxed workspace per run.

Orchestration only — the driver owns the agent CLI, the checks own pass/fail,
and the sandbox module owns isolation. Everything here is offline-testable
with a FakeDriver.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import timedelta
from pathlib import Path

from ...config import Config
from ...llm import LLMUsage
from ..checks import CheckContext, run_checks
from ..drivers.base import AgentDriver, RunPaths
from ..models import (
    AgentResponse,
    BehavioralScenario,
    Difficulty,
    EvalCondition,
    EvalResult,
    ScenarioBase,
    ScoreCard,
)
from ..sandbox import build_agent_env, copy_fixtures, render_prompt
from ..scenarios import load_behavioral_scenarios
from ..skills_install import InstalledSkill, install_skills
from ..trajectory import EventKind, compute_metrics
from .base import RunContext

SkillInstaller = Callable[[Path, RunPaths], list[InstalledSkill]]
#: Deletes DataRobot resources whose names carry the run's prefix.
Teardown = Callable[[Config, str], object]


class AgentBackend:
    """Execute behavioral scenarios by driving a coding agent end to end."""

    name = "agent"

    def __init__(
        self,
        config: Config,
        driver: AgentDriver,
        work_dir: Path,
        skill_installer: SkillInstaller | None = None,
        teardown: Teardown | None = None,
        keep_resources: bool = False,
        dr_client_factory: Callable[[], object] | None = None,
    ) -> None:
        self.config = config
        self.driver = driver
        self.work_dir = work_dir
        self._install_skills = skill_installer or install_skills
        self._teardown = teardown
        self.keep_resources = keep_resources
        self._dr_client_factory = dr_client_factory

    def load_scenarios(
        self, scenarios_dir: Path, difficulty_filter: Difficulty | None
    ) -> list[ScenarioBase]:
        return list(load_behavioral_scenarios(scenarios_dir, difficulty_filter))

    def execute(
        self, scenario: ScenarioBase, condition: EvalCondition, ctx: RunContext
    ) -> EvalResult:
        if not isinstance(scenario, BehavioralScenario):
            raise TypeError(
                f"AgentBackend expects behavioral scenarios, got {type(scenario).__name__}"
            )

        paths = RunPaths(run_dir=self.work_dir / ctx.run_id).create()
        copy_fixtures(scenario, paths)

        installed: list[InstalledSkill] = []
        if condition.skills_source is not None:
            installed = self._install_skills(condition.skills_source, paths)

        prompt = render_prompt(scenario, ctx.run_id)
        env = build_agent_env(paths, self.config, ctx.run_id)
        env.update({k: v.replace("{run_id}", ctx.run_id) for k, v in scenario.env.items()})

        raw = self.driver.run(
            paths.workspace,
            prompt,
            env,
            timedelta(minutes=scenario.timeout_minutes),
        )

        events = self.driver.normalize(raw)
        metrics = compute_metrics(events, raw.wall_seconds, self.driver.capabilities)

        try:
            outcome = run_checks(
                scenario.success_checks,
                CheckContext(
                    workspace=paths.workspace,
                    run_id=ctx.run_id,
                    config=self.config,
                    env=env,
                    dr_client_factory=self._dr_client_factory,
                ),
            )
        finally:
            if self._teardown is not None and not self.keep_resources:
                self._teardown(self.config, ctx.run_id)

        final_text = next(
            (str(e.detail.get("text", "")) for e in reversed(events) if e.kind == EventKind.TEXT),
            "",
        )

        self._write_artifacts(paths, scenario, condition, ctx, raw, metrics, outcome, installed)

        return EvalResult(
            scenario_id=scenario.id,
            condition=condition.condition_type,
            run_number=ctx.run_number,
            agent_response=AgentResponse(plan_text=final_text, usage=LLMUsage()),
            score_card=ScoreCard(),
            scoring_usage=LLMUsage(),
            outcome=outcome,
            trajectory=metrics,
            transcript_path=str(raw.transcript_path),
            run_id=ctx.run_id,
        )

    def _write_artifacts(
        self,
        paths: RunPaths,
        scenario: BehavioralScenario,
        condition: EvalCondition,
        ctx: RunContext,
        raw: object,
        metrics: object,
        outcome: object,
        installed: list[InstalledSkill],
    ) -> None:
        """Persist run metadata, normalized trajectory, and check results."""
        from dataclasses import asdict, is_dataclass

        def _as_dict(obj: object) -> object:
            if is_dataclass(obj) and not isinstance(obj, type):
                return asdict(obj)
            return str(obj)

        raw_meta = _as_dict(raw)
        if isinstance(raw_meta, dict):
            # Paths aren't JSON-serializable; keep them as strings.
            raw_meta = {k: str(v) if isinstance(v, Path) else v for k, v in raw_meta.items()}
        meta = {
            "run_id": ctx.run_id,
            "scenario_id": scenario.id,
            "condition": condition.condition_type.value,
            "run_number": ctx.run_number,
            "timeout_minutes": scenario.timeout_minutes,
            "skills_installed": [asdict(s) for s in installed],
            "transcript": raw_meta,
        }
        paths.meta.write_text(json.dumps(meta, indent=2, default=str) + "\n")
        paths.trajectory.write_text(json.dumps(_as_dict(metrics), indent=2, default=str) + "\n")
        paths.checks.write_text(json.dumps(_as_dict(outcome), indent=2, default=str) + "\n")
