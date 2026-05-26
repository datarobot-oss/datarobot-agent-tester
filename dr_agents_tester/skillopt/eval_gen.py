"""Generate hybrid eval rows for a skill by prompting an LLM grounded in the
skill content + supplemental docs.

Eval rows are JSONL — one row per line. See ``types.EvalRow`` for the schema.

Two row types:
  * code   — agent must emit Python or shell; scored by sandbox + call fingerprint
  * rubric — agent must explain; scored by LLM judge against criteria
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from ..config import Config
from ..llm import call_llm
from .types import EvalRow


_GEN_PROMPT = """\
You are generating an evaluation set for an AI coding skill. The skill below tells
agents how to do a specific kind of DataRobot task. Your job: produce N evaluation rows
that test whether an agent following this skill produces correct results.

# The skill under test
<skill>
{skill}
</skill>

# Supplemental DataRobot docs context
<docs>
{docs}
</docs>

# Row schema (JSON, one per line)

CODE row — agent must emit runnable Python:
{{
  "id": "code-001",
  "type": "code",
  "prompt": "<user-facing task that should produce a Python script>",
  "expected": {{
    "must_not_error": true,
    "must_import": ["datarobot"],                    // optional
    "calls": [
      {{
        "target": "datarobot_predict.deployment.predict",
        "kwargs_required": ["deployment", "max_explanations"],
        "kwargs_values": {{"max_explanations": "3", "explanation_algorithm": "shap"}}
      }}
    ]
  }},
  "source": "section: Prediction Explanations",
  "tags": ["shap", "explanations"]
}}

CODE row (CLI variant):
{{
  "id": "code-cli-001",
  "type": "code",
  "prompt": "<task that should produce a bash CLI command using scripts/make_prediction.py>",
  "expected": {{
    "cli_contains": ["scripts/make_prediction.py", "--max-explanations", "3"]
  }},
  "source": "section: CLI shortcut"
}}

RUBRIC row — agent must explain or recommend:
{{
  "id": "rubric-001",
  "type": "rubric",
  "prompt": "<conceptual question or recommendation request>",
  "expected": {{
    "criteria": [
      "answer mentions that SHAP requires deployment-time enablement",
      "answer recommends XEMP if SHAP is not enabled",
      "answer does NOT confuse deployment explanations with project-level PredictionExplanations"
    ],
    "source_context": "<short verbatim snippet from the skill or docs that the answer must agree with>"
  }},
  "source": "section: Common errors",
  "tags": ["shap", "xemp"]
}}

# Coverage requirements

Generate exactly {n} rows covering this matrix (try to balance):
- Real-time / single-row prediction (no explanations)
- Batch prediction from CSV / DataFrame
- Prediction explanations — SHAP request
- Prediction explanations — XEMP request
- Prediction explanations — threshold_high / threshold_low filtering
- Prediction explanations — passthrough_columns
- Prediction explanations — max_ngram_explanations (text models)
- Generating prediction-data templates
- Validating prediction data before scoring
- Common errors: "Prediction explanations not enabled", missing columns, wrong API
- When to use this skill vs datarobot-model-explainability (deployment vs project)
- CLI usage of scripts/make_prediction.py with various flag combinations
- Deployment.get_features / understanding deployment shape
- Edge cases: empty input, single-row JSON vs list-of-rows JSON

Mix roughly {code_pct}% CODE rows and {rubric_pct}% RUBRIC rows.

Make tasks realistic — phrased the way a DataRobot user would ask. Vary phrasing,
length, and specificity. For CODE rows, ensure the expected `calls` fingerprint is
something an agent following this skill should actually produce (look at the skill's
"Common patterns" / examples to know what's idiomatic).

# Output

Output ONE JSON object per line (JSONL). No markdown fences, no commentary. Just JSONL.
"""


def generate_rows(
    *,
    skill_text: str,
    docs_context: str,
    n: int,
    code_pct: int,
    rubric_pct: int,
    model: str,
    config: Config,
    chunk_size: int = 25,
) -> list[EvalRow]:
    """Generate eval rows in chunks (LLMs hold quality better at ~25/chunk)."""
    all_rows: list[EvalRow] = []
    remaining = n
    chunk_idx = 0
    while remaining > 0:
        this_chunk = min(chunk_size, remaining)
        chunk_idx += 1
        prompt = _GEN_PROMPT.format(
            skill=skill_text,
            docs=docs_context,
            n=this_chunk,
            code_pct=code_pct,
            rubric_pct=rubric_pct,
        )
        print(f"[skillopt] generating chunk {chunk_idx} ({this_chunk} rows)...")
        raw = call_llm(prompt, model, config)
        rows = _parse_jsonl(raw)
        # Reseed IDs to keep them unique across chunks
        for i, r in enumerate(rows):
            r.id = f"{r.type}-{chunk_idx:02d}-{i:03d}"
        all_rows.extend(rows)
        remaining -= len(rows) if rows else this_chunk
    return all_rows


def _parse_jsonl(raw: str) -> list[EvalRow]:
    # Strip any code fences if the model added them
    text = re.sub(r"^```\w*\n|```$", "", raw.strip(), flags=re.MULTILINE)
    rows: list[EvalRow] = []
    for line in text.splitlines():
        line = line.strip().rstrip(",")
        if not line or not line.startswith("{"):
            continue
        try:
            d = json.loads(line)
            rows.append(EvalRow.from_dict(d))
        except Exception as e:
            print(f"[skillopt] skipping unparseable row: {e}: {line[:120]}")
    return rows


def save_rows(rows: list[EvalRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r.to_dict()) + "\n")


def load_rows(path: Path) -> list[EvalRow]:
    rows: list[EvalRow] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(EvalRow.from_dict(json.loads(line)))
    return rows
