"""SkillOpt optimizer: proposes a ranked pool of bounded edits to a skill.

Paper-faithful: the optimizer analyzes failures (and successes) and emits MANY
candidate add/delete/replace edits, ranked by expected impact with failures
prioritized. The loop then clips to the top L_t (the edit-count budget) and
applies them.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from ..config import Config
from ..llm import call_llm
from .types import Edit, RowScore


_OPT_PROMPT = """\
You are SkillOpt, a text-space optimizer for AI agent skills. Analyze the failing rollouts
below and propose a RANKED POOL of candidate edits to the skill. The training loop will apply
only the top {lt} of them this step (the edit-count budget), so order matters: put the edit
most likely to fix the most failures first.

# Current skill
<skill>
{skill}
</skill>

# Failing rollouts (worst-scoring tasks from the training split — your gradient signal)
{failures_block}

# Passing rollouts (what already works — do NOT regress these)
{successes_block}

# Rejected edits already tried this run (do NOT repropose these or trivial variants;
# AVOID locators that keep getting rejected — the signal isn't there)
{rejected_block}

# Cooldown locators (auto-rejected if you target them — pick a DIFFERENT section)
{cooldown_block}

# Constraints
- Propose between {lt} and {n_candidates} candidate edits, ranked best-first.
- Each edit is one of: add | delete | replace, and must be small and targeted (a sentence,
  a bullet, a code block — not a wholesale rewrite).
- For `replace`/`delete`, `old_text` must appear VERBATIM in the skill above (copy exactly).
- Prefer edits that fix a recurring failure pattern across multiple rollouts.
- Do not invent DataRobot APIs or flags — stick to what the skill shows or the failures imply.
- Edits should target DIFFERENT parts of the skill where possible (diversity beats piling onto
  one section).

# Output format
Respond with ONLY a fenced JSON array of edit objects, ranked best-first:
```json
[
  {{
    "op": "add" | "delete" | "replace",
    "locator": "<heading or short unique snippet identifying where the edit applies>",
    "old_text": "<exact substring from skill — required for delete/replace, empty for add>",
    "new_text": "<text to insert — required for add/replace, empty for delete>",
    "rationale": "<one sentence: which failure pattern this fixes>"
  }}
]
```
"""


def _format_failures(failures: list[tuple[str, RowScore]]) -> str:
    if not failures:
        return "(none — all training rollouts passed; propose robustness/clarity edits)"
    out = []
    for prompt, sc in failures[:8]:
        notes = sc.detail.get("notes") or sc.detail.get("error") or ""
        judge = sc.detail.get("judge_raw", "")
        snippet = (
            f"### row {sc.row_id} (score={sc.score})\n"
            f"prompt: {prompt[:280]}\n"
            f"agent_output: {sc.agent_output[:500]}\n"
            f"failure_notes: {notes}\n"
        )
        if judge:
            snippet += f"judge: {judge[:350]}\n"
        out.append(snippet)
    return "\n---\n".join(out)


def _format_successes(successes: list[tuple[str, RowScore]]) -> str:
    if not successes:
        return "(none)"
    return "\n".join(f"- row {sc.row_id}: {prompt[:120]}" for prompt, sc in successes[:8])


def _format_rejected(rejected: list[Edit]) -> str:
    if not rejected:
        return "(none yet)"
    return "\n".join(
        f"- [{e.op}] locator={e.locator[:80]!r} new={e.new_text[:100]!r}"
        for e in rejected[-10:]
    )


def propose_edits(
    *,
    skill: str,
    failures: list[tuple[str, RowScore]],
    successes: list[tuple[str, RowScore]],
    rejected: list[Edit],
    model: str,
    config: Config,
    lt: int,
    n_candidates: int,
    cooldown_locators: list[str] | None = None,
    max_edit_chars: int = 800,
) -> list[Edit]:
    """Return a ranked list of candidate edits (best-first)."""
    cooldown = cooldown_locators or []
    cooldown_block = "\n".join(f"- {loc!r}" for loc in cooldown) if cooldown else "(none)"
    prompt = _OPT_PROMPT.format(
        skill=skill,
        failures_block=_format_failures(failures),
        successes_block=_format_successes(successes),
        rejected_block=_format_rejected(rejected),
        cooldown_block=cooldown_block,
        lt=lt,
        n_candidates=n_candidates,
    )
    raw = call_llm(prompt, model, config)
    edits = _parse_edits(raw)
    # Secondary guardrail: cap each edit's new_text size (not the paper's LR
    # mechanism — just prevents pathological giant inserts).
    for e in edits:
        if len(e.new_text) > max_edit_chars:
            e.new_text = e.new_text[:max_edit_chars]
    return edits


def _parse_edits(raw: str) -> list[Edit]:
    # Prefer a fenced JSON array; fall back to the first bare [...] array.
    m = re.search(r"```json\s*(\[.*?\])\s*```", raw, re.DOTALL)
    if not m:
        m = re.search(r"(\[\s*\{.*\}\s*\])", raw, re.DOTALL)
    if not m:
        # Maybe the model returned a single object — wrap it.
        obj = re.search(r"(\{[^{}]*\"op\".*?\})", raw, re.DOTALL)
        if obj:
            data = [json.loads(obj.group(1))]
        else:
            raise ValueError(f"Optimizer returned no parseable edits. Raw:\n{raw[:1000]}")
    else:
        data = json.loads(m.group(1))
    edits: list[Edit] = []
    for d in data:
        if not isinstance(d, dict) or "op" not in d:
            continue
        edits.append(
            Edit(
                op=d["op"],
                locator=d.get("locator", ""),
                old_text=d.get("old_text", ""),
                new_text=d.get("new_text", ""),
                rationale=d.get("rationale", ""),
            )
        )
    return edits


def apply_edit(skill: str, edit: Edit) -> tuple[str, str]:
    """Apply one edit; return (new_skill, error_or_empty)."""
    if edit.op == "add":
        if not edit.new_text.strip():
            return skill, "add: empty new_text"
        if edit.locator and edit.locator in skill:
            idx = skill.index(edit.locator) + len(edit.locator)
            return skill[:idx] + "\n\n" + edit.new_text.strip() + "\n" + skill[idx:], ""
        return skill.rstrip() + "\n\n" + edit.new_text.strip() + "\n", ""

    if edit.op == "delete":
        if not edit.old_text or edit.old_text not in skill:
            return skill, f"delete: old_text not found ({edit.old_text[:80]!r})"
        return skill.replace(edit.old_text, "", 1), ""

    if edit.op == "replace":
        if not edit.old_text or edit.old_text not in skill:
            return skill, f"replace: old_text not found ({edit.old_text[:80]!r})"
        return skill.replace(edit.old_text, edit.new_text, 1), ""

    return skill, f"unknown op: {edit.op}"


def apply_edits(skill: str, edits: list[Edit]) -> tuple[str, list[Edit], list[tuple[Edit, str]]]:
    """Apply a list of edits sequentially.

    Returns (new_skill, applied_edits, failed_edits_with_reason). Edits that
    can't be located (e.g. because a prior edit changed the text) are skipped.
    """
    current = skill
    applied: list[Edit] = []
    failed: list[tuple[Edit, str]] = []
    for e in edits:
        candidate, err = apply_edit(current, e)
        if err:
            failed.append((e, err))
            continue
        current = candidate
        applied.append(e)
    return current, applied, failed
