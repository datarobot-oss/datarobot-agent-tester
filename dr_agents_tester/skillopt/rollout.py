"""Generate an agent's answer to a row's prompt, with the skill in context.

This is the 'frozen agent' from the SkillOpt paper — its only external state
is the current skill document.
"""

from __future__ import annotations

from ..config import Config
from ..llm import call_llm
from .types import EvalRow

_ROLLOUT_PROMPT = """\
You are an AI coding assistant working in a DataRobot context. You have been given the
following skill document — treat it as your operating manual for this kind of task. Follow
it as written. Prefer code from the skill over what you "know".

<skill>
{skill}
</skill>

## Task

{prompt}

## Output requirements

{output_hint}
"""


def _output_hint(row: EvalRow) -> str:
    if row.type == "code":
        if row.expected.get("cli_contains"):
            return (
                "Respond with a single ```bash code block containing the exact CLI "
                "command(s) the user should run. No prose before or after."
            )
        return (
            "Respond with a single ```python code block containing a complete, runnable "
            "script. Use the DataRobot SDK as described in the skill. No prose, no "
            "placeholder TODOs. Imports at the top."
        )
    return "Respond with a clear, concise prose answer (no code blocks needed)."


def rollout(skill: str, row: EvalRow, model: str, config: Config) -> str:
    prompt = _ROLLOUT_PROMPT.format(skill=skill, prompt=row.prompt, output_hint=_output_hint(row))
    return call_llm(prompt, model, config)
