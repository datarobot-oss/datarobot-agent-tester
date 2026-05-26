"""SkillOpt optimizer: proposes bounded edits to a skill document."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from ..config import Config
from ..llm import call_llm
from .types import Edit, RowScore


_OPT_PROMPT = """\
You are SkillOpt, a text-space optimizer for AI agent skills. Your job: propose ONE bounded
edit to the skill below so that an agent following it scores higher on the failing tasks.

# Current skill
<skill>
{skill}
</skill>

# Recent failing rollouts (worst-scoring tasks from the training split)
{failures_block}

# Rejected edits so far this run (do NOT propose these again or trivial variants;
# in particular, AVOID proposing further edits to the same locator if it has been
# rejected recently — that locator is not where the signal is)
{rejected_block}

# Constraints
- Output ONE edit: add | delete | replace.
- The edit must be small and targeted (a sentence, a bullet, a code example — not a rewrite).
- For `replace` and `delete`, `old_text` must appear VERBATIM in the skill above (copy exactly).
- Prefer edits that fix a recurring failure pattern across multiple rollouts.
- Do not invent DataRobot APIs or flags. Stick to what's already shown in the skill or is
  clearly implied by the failure detail.
- **Textual learning-rate budget (HARD LIMIT): `new_text` must be ≤ {lr_chars} characters.**
  This budget shrinks after every rejected edit and grows after every accepted edit; if the
  budget is small, propose a smaller / more surgical edit (a single line, a single phrase).

# Cooldown locators (DO NOT propose any edit whose `locator` matches or overlaps these — they
# have been rejected too many times recently and are auto-rejected if you target them again).
# Pick a DIFFERENT section of the skill to edit.
{cooldown_block}

# Output format
Respond with ONLY a fenced JSON block:
```json
{{
  "op": "add" | "delete" | "replace",
  "locator": "<heading or short unique snippet identifying where the edit applies>",
  "old_text": "<exact substring from skill — required for delete/replace, empty for add>",
  "new_text": "<text to insert — required for add/replace, empty for delete>",
  "rationale": "<one sentence: which failure pattern this fixes and how>"
}}
```
"""


def _format_failures(failures: list[tuple[str, RowScore]]) -> str:
    if not failures:
        return "(none — all training rollouts passed; suggest a robustness/clarity edit)"
    out = []
    for prompt, sc in failures[:6]:
        notes = sc.detail.get("notes") or sc.detail.get("error") or ""
        judge = sc.detail.get("judge_raw", "")
        snippet = (
            f"### row {sc.row_id} (score={sc.score})\n"
            f"prompt: {prompt[:300]}\n"
            f"agent_output: {sc.agent_output[:600]}\n"
            f"failure_notes: {notes}\n"
        )
        if judge:
            snippet += f"judge: {judge[:400]}\n"
        out.append(snippet)
    return "\n---\n".join(out)


def _format_rejected(rejected: list[Edit]) -> str:
    if not rejected:
        return "(none yet)"
    return "\n".join(
        f"- [{e.op}] locator={e.locator!r} new={e.new_text[:120]!r} (rejected: {e.rationale[:120]})"
        for e in rejected[-8:]
    )


def propose_edit(
    *,
    skill: str,
    failures: list[tuple[str, RowScore]],
    rejected: list[Edit],
    model: str,
    config: Config,
    lr_chars: int,
    cooldown_locators: list[str] | None = None,
) -> Edit:
    cooldown = cooldown_locators or []
    cooldown_block = (
        "\n".join(f"- {loc!r}" for loc in cooldown) if cooldown else "(none)"
    )
    prompt = _OPT_PROMPT.format(
        skill=skill,
        failures_block=_format_failures(failures),
        rejected_block=_format_rejected(rejected),
        lr_chars=lr_chars,
        cooldown_block=cooldown_block,
    )
    raw = call_llm(prompt, model, config)
    m = re.search(r"```json\s*(\{.*?\})\s*```", raw, re.DOTALL)
    if not m:
        # Last-ditch: try to find a bare json object
        m = re.search(r"(\{[^{}]*\"op\"[^{}]*\})", raw, re.DOTALL)
    if not m:
        raise ValueError(f"Optimizer did not return parseable JSON. Raw:\n{raw[:1000]}")
    data = json.loads(m.group(1))
    # Enforce the textual learning-rate budget by hard-truncating new_text.
    # The optimizer is asked to respect it but we don't trust it to.
    new_text = data.get("new_text", "")
    if len(new_text) > lr_chars:
        new_text = new_text[:lr_chars]
    return Edit(
        op=data["op"],
        locator=data.get("locator", ""),
        old_text=data.get("old_text", ""),
        new_text=new_text,
        rationale=data.get("rationale", ""),
    )


def apply_edit(skill: str, edit: Edit) -> tuple[str, str]:
    """Apply edit; return (new_skill, error_or_empty)."""
    if edit.op == "add":
        # Add after the locator (heading match) or at end if locator missing.
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
