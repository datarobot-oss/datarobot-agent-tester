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
You are generating an evaluation set for an arbitrary AI coding/agent skill. A "skill" is a
markdown instruction file that tells an agent how to perform some recurring task. Your job:
read the skill, INFER what it teaches, and produce evaluation rows that test whether an agent
following this skill produces correct results.

# The skill under test
<skill>
{skill}
</skill>

# Supplemental reference context (may be empty)
<docs>
{docs}
</docs>

# Step 1 — analyze the skill (think, but don't output this)
Identify, from the skill text only:
- The concrete operations / APIs / functions / CLI commands it teaches (names, key arguments).
- The decisions/recommendations it expects an agent to make (when to use X vs Y, defaults,
  required preconditions, common errors it warns about).
- Any helper scripts / CLI entrypoints it references and their important flags.

# Step 2 — generate rows from THAT analysis

Two row types:

CODE row — the task should make the agent emit runnable code. Score is by executing the code
in a sandbox (every imported module is auto-mocked, so calls are recorded but nothing real
runs) and checking it called the right API with the right arguments:
{{
  "id": "code-001",
  "type": "code",
  "prompt": "<realistic user task that should produce a code script>",
  "expected": {{
    "must_not_error": true,
    "must_import": ["<top-level module the skill tells them to import>"],   // optional
    "calls": [
      {{
        "target": "<dotted.path.to.function_or_method the skill teaches>",
        "kwargs_required": ["<kwargs the skill shows for this call>"],      // optional
        "kwargs_values": {{"<kwarg>": "<expected value substring>"}}        // optional
      }}
    ]
  }},
  "source": "<which skill section this came from>",
  "tags": ["<topic>"]
}}

CODE row (CLI variant) — when the skill references a CLI/script entrypoint:
{{
  "id": "code-cli-001",
  "type": "code",
  "prompt": "<task that should produce a shell command using the skill's script>",
  "expected": {{ "cli_contains": ["<script path or command>", "<required flag>", "<value>"] }},
  "source": "<section>"
}}

RUBRIC row — the task is conceptual; the agent must explain/recommend. Scored by an LLM judge
against pass/fail criteria:
{{
  "id": "rubric-001",
  "type": "rubric",
  "prompt": "<conceptual question or recommendation request the skill should answer>",
  "expected": {{
    "criteria": [
      "<specific, checkable claim the answer must make, grounded in the skill>",
      "<a thing the answer must NOT get wrong>"
    ],
    "source_context": "<short verbatim snippet from the skill the answer must agree with>"
  }},
  "source": "<section>",
  "tags": ["<topic>"]
}}

# Coverage requirements

Generate exactly {n} rows. DERIVE the coverage from the skill you analyzed in Step 1 — there is
NO fixed topic list. Aim to cover, in rough proportion to how much the skill emphasizes them:
- Each distinct operation / API / CLI the skill teaches (the "happy path" for each).
- Important argument/flag combinations and non-default options the skill documents.
- The decisions/recommendations and "when to use X vs Y" guidance (rubric rows).
- Preconditions and common errors the skill explicitly warns about.
- A few realistic edge cases implied by the skill.

Mix roughly {code_pct}% CODE rows and {rubric_pct}% RUBRIC rows.

CRITICAL — only test what THIS skill actually teaches:
- Every `calls` target / `cli_contains` token must be an API, function, or flag the skill text
  actually shows or clearly implies. Do NOT invent APIs.
- Do NOT write expectations about the test harness, mock objects, undefined variables, or
  "sample/placeholder data" — test real-world correct usage only.
- Phrase prompts the way a real user of this skill's domain would ask. Vary phrasing and depth.

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
