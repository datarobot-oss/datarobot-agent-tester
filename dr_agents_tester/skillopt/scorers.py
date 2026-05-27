"""Pluggable scorers for SkillOpt eval rows.

Scorer is a Protocol so backends (mock-exec, live-DR, rubric, AST-only) can be
swapped freely. The composite scorer dispatches per row by ``supports()``.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from ..config import Config
from ..llm import call_llm
from .types import EvalRow, RowScore


class Scorer(Protocol):
    name: str

    def supports(self, row: EvalRow) -> bool: ...
    def score(self, row: EvalRow, agent_output: str) -> RowScore: ...


# ---------------------------------------------------------------------------
# Mock-exec scorer for code rows
# ---------------------------------------------------------------------------

# Sandbox runner: stubs the datarobot SDK so we can score any generated code
# by what it *would have called*, without hitting a real deployment.
_SANDBOX_RUNNER = r"""
import json, sys, types

_calls = []

def _make_recorder(qualname):
    def _rec(*args, **kwargs):
        _calls.append({
            "target": qualname,
            "args": [repr(a)[:200] for a in args],
            "kwargs": {k: repr(v)[:200] for k, v in kwargs.items()},
        })
        # Return a flexible mock so chained access works
        return _Mock(qualname + "()")
    return _rec

class _Mock:
    def __init__(self, name="mock"):
        self.__name = name
    def __getattr__(self, k):
        sub = _Mock(f"{self.__name}.{k}")
        setattr(self, k, sub)
        return sub
    def __call__(self, *a, **kw):
        _calls.append({
            "target": self.__name,
            "args": [repr(x)[:200] for x in a],
            "kwargs": {k: repr(v)[:200] for k, v in kw.items()},
        })
        return _Mock(self.__name + "()")
    def __iter__(self):
        return iter([])
    def __repr__(self):
        return f"<mock {self.__name}>"
    # DataFrame-ish
    @property
    def dataframe(self):
        return _Mock(self.__name + ".dataframe")
    def to_dict(self, *a, **kw):
        return {}

def _install(mod_name, attrs=None):
    m = types.ModuleType(mod_name)
    if attrs:
        for a in attrs:
            setattr(m, a, _make_recorder(f"{mod_name}.{a}"))
    sys.modules[mod_name] = m
    return m

# datarobot
dr = _install("datarobot", ["Client", "Deployment", "BatchPredictionJob", "Dataset",
                            "Project", "Model", "PredictionExplanations"])
# Make Deployment.get etc. return a Mock so chained calls record
dr.Deployment = _Mock("datarobot.Deployment")
dr.BatchPredictionJob = _Mock("datarobot.BatchPredictionJob")
dr.Dataset = _Mock("datarobot.Dataset")
dr.Project = _Mock("datarobot.Project")
dr.Model = _Mock("datarobot.Model")
dr.PredictionExplanations = _Mock("datarobot.PredictionExplanations")
dr.Client = _make_recorder("datarobot.Client")

# datarobot_predict.deployment
dp_mod = _install("datarobot_predict")
dep_mod = types.ModuleType("datarobot_predict.deployment")
dep_mod.predict = _make_recorder("datarobot_predict.deployment.predict")
sys.modules["datarobot_predict.deployment"] = dep_mod
dp_mod.deployment = dep_mod

# pandas — provide a minimal stub so DataFrame() constructions don't fail
try:
    import pandas  # noqa: F401
except Exception:
    pandas_mod = types.ModuleType("pandas")
    pandas_mod.DataFrame = _Mock("pandas.DataFrame")
    pandas_mod.read_csv = _make_recorder("pandas.read_csv")
    pandas_mod.read_json = _make_recorder("pandas.read_json")
    sys.modules["pandas"] = pandas_mod

# Run user code inside a throwaway temp dir so any real file writes
# (e.g. pandas .to_csv on a non-stubbed pandas) can't escape into the repo.
import os, tempfile
_sandbox_cwd = tempfile.mkdtemp(prefix="skillopt-sbx-")
os.chdir(_sandbox_cwd)

USER_CODE = __USER_CODE__

err_repr = None
try:
    exec(USER_CODE, {"__name__": "__main__"})
except SystemExit:
    pass
except Exception as e:
    err_repr = f"{type(e).__name__}: {e}"

print("\n__SKILLOPT_CALLS__")
print(json.dumps({"calls": _calls, "error": err_repr}))
"""


def _extract_code_blocks(text: str) -> list[tuple[str, str]]:
    """Return (lang, code) for each fenced block. Empty lang => unknown."""
    blocks = []
    for m in re.finditer(r"```([\w+-]*)\n(.*?)```", text, re.DOTALL):
        blocks.append((m.group(1).lower(), m.group(2)))
    return blocks


def _pick_runnable_code(agent_output: str) -> str:
    """Pick the first python-ish block, else the longest code block, else raw text."""
    blocks = _extract_code_blocks(agent_output)
    if not blocks:
        # Maybe agent answered with bare code
        if "import " in agent_output or "dr." in agent_output:
            return agent_output
        return ""
    py = [c for lang, c in blocks if lang in ("python", "py", "")]
    if py:
        return py[0]
    # Bash-only output: extract the python script reference if any, else return ""
    return blocks[0][1] if blocks else ""


def _pick_shell_block(agent_output: str) -> str:
    """Pick first bash/shell block (for CLI rows)."""
    for lang, code in _extract_code_blocks(agent_output):
        if lang in ("bash", "shell", "sh"):
            return code
    return ""


@dataclass
class MockExecScorer:
    """Score code rows by running them in a sandbox with a stubbed DR SDK.

    The expected fingerprint format:
        {
          "calls": [
            {"target": "datarobot_predict.deployment.predict",
             "kwargs_required": ["max_explanations"],
             "kwargs_values": {"max_explanations": "3"}   # repr-compared substring
            }
          ],
          "must_not_error": true,
          "must_import": ["datarobot"],            # optional
          "cli_contains": ["--max-explanations"]   # optional, checks shell block instead
        }
    """

    name: str = "mock_exec"
    timeout: float = 10.0

    def supports(self, row: EvalRow) -> bool:
        return row.type == "code"

    def score(self, row: EvalRow, agent_output: str) -> RowScore:
        expected = row.expected
        detail: dict = {}

        # CLI-style row: check the shell block for required substrings.
        if expected.get("cli_contains"):
            shell = _pick_shell_block(agent_output) or agent_output
            missing = [s for s in expected["cli_contains"] if s not in shell]
            score = 1.0 - (len(missing) / max(1, len(expected["cli_contains"])))
            detail = {"missing_cli_tokens": missing, "shell_snippet": shell[:300]}
            return RowScore(
                row_id=row.id,
                score=score,
                passed=not missing,
                detail=detail,
                agent_output=agent_output,
            )

        code = _pick_runnable_code(agent_output)
        if not code.strip():
            return RowScore(
                row_id=row.id,
                score=0.0,
                passed=False,
                detail={"error": "no_code_block_found"},
                agent_output=agent_output,
            )

        # Execute in subprocess sandbox
        runner = _SANDBOX_RUNNER.replace("__USER_CODE__", repr(code))
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
            f.write(runner)
            runner_path = f.name
        try:
            proc = subprocess.run(
                [sys.executable, runner_path],
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
        except subprocess.TimeoutExpired:
            return RowScore(
                row_id=row.id,
                score=0.0,
                passed=False,
                detail={"error": "timeout"},
                agent_output=agent_output,
            )
        finally:
            Path(runner_path).unlink(missing_ok=True)

        stdout = proc.stdout
        marker = "__SKILLOPT_CALLS__"
        if marker not in stdout:
            return RowScore(
                row_id=row.id,
                score=0.0,
                passed=False,
                detail={
                    "error": "sandbox_failed",
                    "stderr": proc.stderr[:500],
                    "stdout": stdout[:500],
                },
                agent_output=agent_output,
            )
        payload = json.loads(stdout.split(marker, 1)[1].strip())
        recorded = payload["calls"]
        err = payload["error"]

        # Scoring components
        components: list[float] = []
        notes: list[str] = []

        if expected.get("must_not_error", True):
            ok = err is None
            components.append(1.0 if ok else 0.0)
            if not ok:
                notes.append(f"exec_error: {err}")

        for must_import in expected.get("must_import", []):
            present = any(c["target"].split(".")[0] == must_import for c in recorded) or (
                f"import {must_import}" in code
            )
            components.append(1.0 if present else 0.0)
            if not present:
                notes.append(f"missing_import:{must_import}")

        for expected_call in expected.get("calls", []):
            target = expected_call["target"]
            matches = [c for c in recorded if c["target"] == target]
            if not matches:
                components.append(0.0)
                notes.append(f"missing_call:{target}")
                continue
            # Match best call by kwargs
            best = 0.0
            for m in matches:
                sub = 1.0
                for kw in expected_call.get("kwargs_required", []):
                    if kw not in m["kwargs"]:
                        sub -= 1.0 / max(1, len(expected_call.get("kwargs_required", [])))
                for kw, want_val in expected_call.get("kwargs_values", {}).items():
                    actual = m["kwargs"].get(kw, "")
                    if str(want_val) not in actual:
                        sub -= 0.5 / max(1, len(expected_call.get("kwargs_values", {})))
                best = max(best, max(0.0, sub))
            components.append(best)
            if best < 1.0:
                notes.append(f"weak_call_match:{target}={best:.2f}")

        score = sum(components) / max(1, len(components))
        detail = {
            "components": components,
            "notes": notes,
            "recorded_calls": recorded[:20],
            "exec_error": err,
        }
        return RowScore(
            row_id=row.id,
            score=round(score, 3),
            passed=score >= 0.8 and not notes,
            detail=detail,
            agent_output=agent_output,
        )


# ---------------------------------------------------------------------------
# Live-execution placeholder (interface only — implement when needed)
# ---------------------------------------------------------------------------


@dataclass
class LiveExecScorer:
    """Run generated code against a real DR deployment. Not yet implemented.

    Same interface as MockExecScorer — fill in by replacing the sandbox stubs
    with a real `datarobot.Client(...)` call and recording the round-trip
    result. Eval rows can include `live_deployment_id` to target a specific
    deployment.
    """

    name: str = "live_exec"

    def supports(self, row: EvalRow) -> bool:
        return row.type == "code" and "live_deployment_id" in row.expected

    def score(self, row: EvalRow, agent_output: str) -> RowScore:
        raise NotImplementedError(
            "LiveExecScorer is a placeholder. Implement by exec'ing the agent "
            "code with real datarobot SDK and asserting on the returned frame."
        )


# ---------------------------------------------------------------------------
# Rubric scorer for prose rows (LLM judge)
# ---------------------------------------------------------------------------

_RUBRIC_PROMPT = """\
You are grading an AI agent's answer against a rubric. Be strict but fair.

## Question the agent was asked
{prompt}

## Agent's answer
<answer>
{answer}
</answer>

## Rubric criteria (each is pass/fail)
{criteria_block}

## Source-of-truth context (the agent's answer must be consistent with this)
{source}

For EACH criterion, output one line: `PASS - <crit short>` or `FAIL - <crit short> - <why>`.
After all lines, output exactly one final line: `SCORE: <X>/<N>` where N is total criteria and X is number passed.
"""


@dataclass
class RubricScorer:
    """LLM-judge scorer for prose answers."""

    config: Config
    judge_model: str = ""
    name: str = "rubric"

    def supports(self, row: EvalRow) -> bool:
        return row.type == "rubric"

    def score(self, row: EvalRow, agent_output: str) -> RowScore:
        criteria = row.expected.get("criteria", [])
        if not criteria:
            return RowScore(row.id, 0.0, False, {"error": "no_criteria"}, agent_output)

        criteria_block = "\n".join(f"{i+1}. {c}" for i, c in enumerate(criteria))
        source = row.expected.get("source_context", row.source or "(none provided)")
        prompt = _RUBRIC_PROMPT.format(
            prompt=row.prompt,
            answer=agent_output,
            criteria_block=criteria_block,
            source=source,
        )
        model = self.judge_model or self.config.test_model
        raw = call_llm(prompt, model, self.config)

        # Parse SCORE: X/N
        m = re.search(r"SCORE:\s*(\d+)\s*/\s*(\d+)", raw)
        if m:
            x, n = int(m.group(1)), int(m.group(2))
            score = x / n if n else 0.0
        else:
            # Fall back to counting PASS/FAIL lines
            passes = len(re.findall(r"^PASS\b", raw, re.MULTILINE))
            fails = len(re.findall(r"^FAIL\b", raw, re.MULTILINE))
            total = passes + fails or 1
            score = passes / total

        return RowScore(
            row_id=row.id,
            score=round(score, 3),
            passed=score >= 0.8,
            detail={"judge_raw": raw[:2000]},
            agent_output=agent_output,
        )


# ---------------------------------------------------------------------------
# Composite scorer: dispatches per row
# ---------------------------------------------------------------------------


@dataclass
class CompositeScorer:
    scorers: list[Scorer]
    name: str = "composite"

    def supports(self, row: EvalRow) -> bool:
        return any(s.supports(row) for s in self.scorers)

    def score(self, row: EvalRow, agent_output: str) -> RowScore:
        for s in self.scorers:
            if s.supports(row):
                return s.score(row, agent_output)
        return RowScore(
            row_id=row.id,
            score=0.0,
            passed=False,
            detail={"error": f"no_scorer_for_type:{row.type}"},
            agent_output=agent_output,
        )
