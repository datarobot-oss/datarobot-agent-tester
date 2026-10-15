"""Tests for the delivered-script SDK-conformance scanner.

The call-resolution core is covered in the skills repo; these tests pin the
tester-side contract instead — what gets scanned, what gets excluded, and the
promise that a bad workspace degrades to a report rather than an exception.

The one case that must never regress is `test_catches_the_round_2_residue`:
that exact defect shipped in a passing run because nothing measured it.
"""

from pathlib import Path

import pytest

pytest.importorskip("datarobot", reason="the scanner resolves calls against the real SDK")

from dr_agents_tester.eval.sdk_conformance import (  # noqa: E402
    MAX_SCRIPT_BYTES,
    scan_delivered_scripts,
)

# Verbatim from .behavioral-runs/drat-audit-skill-pr-r1-2026082714053390ba —
# the agent fixed this inline and left the delivered script holding it.
ROUND_2_RESIDUE = """
import datarobot as dr

dr.BatchPredictionJob.score_to_file(
    deployment=deployment,
    intake_file_path="./data/holdout.csv",
    output_file_path="./predictions.csv",
)
"""

CLEAN = """
import datarobot as dr

dr.BatchPredictionJob.score_to_file(deployment, "./in.csv", "./out.csv")
"""


def _workspace(tmp_path: Path) -> Path:
    ws = tmp_path / "workspace"
    ws.mkdir()
    return ws


def test_catches_the_round_2_residue(tmp_path: Path) -> None:
    ws = _workspace(tmp_path)
    (ws / "run_datarobot.py").write_text(ROUND_2_RESIDUE)

    scan = scan_delivered_scripts(ws)

    assert scan.scanned == ["run_datarobot.py"]
    assert scan.num_defects == 1
    (summary,) = scan.summaries()
    assert "run_datarobot.py" in summary
    assert "score_to_file" in summary
    assert "intake_path" in summary and "output_path" in summary


def test_correct_positional_call_is_clean(tmp_path: Path) -> None:
    ws = _workspace(tmp_path)
    (ws / "good.py").write_text(CLEAN)

    scan = scan_delivered_scripts(ws)

    assert scan.scanned == ["good.py"]
    assert scan.num_defects == 0
    assert scan.summaries() == []


def test_fixtures_are_excluded_but_agent_files_are_not(tmp_path: Path) -> None:
    """A fixture's defects are not the agent's residue; its own always are."""
    ws = _workspace(tmp_path)
    (ws / "fixture.py").write_text(ROUND_2_RESIDUE)
    (ws / "agent_wrote_this.py").write_text(ROUND_2_RESIDUE)

    scan = scan_delivered_scripts(ws, exclude=["fixture.py"])

    assert scan.scanned == ["agent_wrote_this.py"]
    assert scan.num_defects == 1


def test_exclude_tolerates_leading_slash(tmp_path: Path) -> None:
    ws = _workspace(tmp_path)
    (ws / "fixture.py").write_text(ROUND_2_RESIDUE)

    assert scan_delivered_scripts(ws, exclude=["/fixture.py"]).num_defects == 0


def test_nested_fixture_dest_is_excluded(tmp_path: Path) -> None:
    ws = _workspace(tmp_path)
    (ws / "data").mkdir()
    (ws / "data" / "seed.py").write_text(ROUND_2_RESIDUE)

    assert scan_delivered_scripts(ws, exclude=["data/seed.py"]).num_defects == 0


def test_unparseable_script_is_itself_a_defect(tmp_path: Path) -> None:
    """A script that does not parse certainly does not run."""
    ws = _workspace(tmp_path)
    (ws / "broken.py").write_text("def oops(:\n")

    scan = scan_delivered_scripts(ws)

    assert scan.num_defects == 1
    assert "unparseable-script" in scan.summaries()[0]


def test_vendored_and_cache_directories_are_skipped(tmp_path: Path) -> None:
    ws = _workspace(tmp_path)
    for noise in ("__pycache__", ".venv/lib", "node_modules"):
        (ws / noise).mkdir(parents=True)
        (ws / noise / "mod.py").write_text(ROUND_2_RESIDUE)

    scan = scan_delivered_scripts(ws)

    assert scan.scanned == []
    assert scan.num_defects == 0


def test_oversized_file_is_skipped_not_fatal(tmp_path: Path) -> None:
    ws = _workspace(tmp_path)
    (ws / "huge.py").write_text(ROUND_2_RESIDUE)

    scan = scan_delivered_scripts(ws, max_bytes=10)

    assert scan.scanned == []
    assert scan.num_defects == 0
    assert scan.skipped and "huge.py" in scan.skipped[0]


def test_default_size_cap_is_generous_enough_for_real_scripts() -> None:
    assert MAX_SCRIPT_BYTES > 100_000


def test_missing_workspace_reports_rather_than_raises(tmp_path: Path) -> None:
    """A metric must never cost a completed run its results."""
    scan = scan_delivered_scripts(tmp_path / "does-not-exist")

    assert scan.num_defects == 0
    assert scan.scanned == []


def test_non_sdk_code_is_ignored(tmp_path: Path) -> None:
    """Only datarobot calls are resolved — the toy LangGraph fixture must stay clean."""
    ws = _workspace(tmp_path)
    (ws / "agent.py").write_text(
        "from langgraph.graph import StateGraph\n"
        "g = StateGraph(dict)\n"
        "g.add_node('n', lambda s: s)\n"
        "g.compile(checkpointer=None, debug=True)\n"
    )

    scan = scan_delivered_scripts(ws)

    assert scan.scanned == ["agent.py"]
    assert scan.num_defects == 0
