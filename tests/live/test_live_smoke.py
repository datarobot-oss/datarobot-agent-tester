"""Live smoke: the full harness path against real opencode + the LLM Gateway.

Skipped unless ``DRAT_LIVE=1``. Requires the pinned opencode on PATH and
``DATAROBOT_API_TOKEN``/``DATAROBOT_ENDPOINT`` in the environment (a `.env`
works via the CLI, but this test reads the environment directly).

This exercises driver → sandbox → checks end to end with a cheap file-only
scenario (~1 minute, no DataRobot resources). The full train→deploy→predict
golden journey lives in datarobot-agent-skills and runs via
``dr-agent eval run-behavioral`` there.

Debugging knobs: ``DRAT_LIVE_KEEP=1`` keeps the run directory;
``DRAT_LIVE_VERSION_DRIFT=1`` tolerates a non-pinned opencode.
"""

import os
from pathlib import Path

import pytest

from dr_agents_tester.config import Config
from dr_agents_tester.eval.backends import AgentBackend, RunContext, make_run_id
from dr_agents_tester.eval.drivers.opencode import OpenCodeDriver
from dr_agents_tester.eval.models import (
    BehavioralScenario,
    CheckSpec,
    ConditionType,
    Difficulty,
    EvalCondition,
)

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(os.environ.get("DRAT_LIVE") != "1", reason="set DRAT_LIVE=1 to run"),
]


def test_file_scenario_end_to_end(tmp_path: Path) -> None:
    cfg = Config()
    cfg.validate()

    scenario = BehavioralScenario(
        id="live-smoke-hello",
        name="Live smoke: write a file",
        difficulty=Difficulty.EASY,
        prompt=(
            "Create a file named hello.txt in the current directory containing exactly "
            "the single line: hello from the smoke test"
        ),
        skills_under_test=["datarobot-setup"],
        success_checks=[
            CheckSpec(type="file_exists", params={"path": "hello.txt", "min_bytes": 5})
        ],
        timeout_minutes=5,
        source_dir=tmp_path,
    )

    keep = os.environ.get("DRAT_LIVE_KEEP") == "1"
    driver = OpenCodeDriver(
        allow_version_drift=os.environ.get("DRAT_LIVE_VERSION_DRIFT") == "1"
    )
    backend = AgentBackend(
        config=cfg,
        driver=driver,
        work_dir=tmp_path / "runs",
        keep_resources=keep,
    )
    condition = EvalCondition(condition_type=ConditionType.NO_SKILL)
    ctx = RunContext(run_number=1, run_id=make_run_id(scenario, condition, 1))

    result = backend.execute(scenario, condition, ctx)

    print(f"\nRun artifacts: {tmp_path / 'runs' / ctx.run_id}")
    assert result.trajectory is not None
    assert result.trajectory.num_turns and result.trajectory.num_turns >= 1
    assert result.trajectory.total_tokens and result.trajectory.total_tokens > 0
    assert result.outcome is not None, "checks did not run"
    assert result.outcome.outcome_pass, (
        f"live smoke failed: {[(c.check_type, c.evidence, c.error) for c in result.outcome.checks]}"
    )
