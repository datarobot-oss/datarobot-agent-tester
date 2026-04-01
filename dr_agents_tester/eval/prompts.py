"""Prompt construction and response parsing for the evaluation framework."""

from __future__ import annotations

import json
import re

from .models import Scenario, ScoreCard


def build_agent_prompt(
    scenario: Scenario,
    repo_tree: str,
    agents_md_content: str | None = None,
) -> str:
    """Build the prompt sent to the agent LLM.

    The agent is asked to produce a detailed implementation plan (not code)
    for the given scenario, given the repo file tree and optionally an
    AGENTS.md file for additional context.
    """
    parts: list[str] = []

    parts.append(
        "You are an expert software engineer. You have been given a coding task "
        "for a repository. Your job is to produce a **detailed implementation plan** "
        "(NOT code). Your plan should cover:\n"
        "1. Which files need to be modified or created\n"
        "2. The approach you would take and why\n"
        "3. Patterns and conventions to follow from the existing codebase\n"
        "4. Potential pitfalls to avoid\n"
        "5. A step-by-step implementation order\n"
    )

    if agents_md_content:
        parts.append("## Repository Guide (AGENTS.md)\n")
        parts.append(agents_md_content)
        parts.append("")

    parts.append("## Repository File Tree\n")
    parts.append("```")
    parts.append(repo_tree)
    parts.append("```\n")

    parts.append("## Task\n")
    parts.append(scenario.prompt)
    parts.append("")

    parts.append(
        "## Instructions\n"
        "Produce a detailed implementation plan. Be specific about:\n"
        "- Exact file paths to modify or create\n"
        "- The technical approach and architecture decisions\n"
        "- Codebase patterns and conventions to follow\n"
        "- Common mistakes to avoid\n"
        "- Testing considerations\n\n"
        "Format your response as a clear, structured plan with sections."
    )

    return "\n".join(parts)


def build_scoring_prompt(
    scenario: Scenario,
    plan_text: str,
) -> str:
    """Build the prompt sent to the scoring LLM.

    The scorer evaluates the agent's plan against the scenario's ground truth
    and returns a JSON score card.
    """
    parts: list[str] = []

    parts.append(
        "You are an expert code reviewer evaluating an AI agent's implementation plan. "
        "Score the plan on five dimensions, each from 0.0 to 1.0.\n"
    )

    parts.append("## Scoring Dimensions\n")
    parts.append(
        "1. **file_identification** (weight 0.25): Did the agent identify the correct "
        "files to modify/create?\n"
        "2. **approach_correctness** (weight 0.30): Is the proposed approach technically "
        "correct and appropriate?\n"
        "3. **pattern_awareness** (weight 0.15): Does the plan follow codebase conventions "
        "and patterns?\n"
        "4. **pitfall_avoidance** (weight 0.15): Does the plan avoid known common mistakes?\n"
        "5. **completeness** (weight 0.15): Does the plan address all aspects of the task?\n"
    )

    parts.append("## Expected Ground Truth\n")
    parts.append(f"**Expected files:** {', '.join(scenario.expected_files)}\n")
    parts.append(f"**Expected approach:** {scenario.expected_approach}\n")
    parts.append(f"**Expected patterns:** {', '.join(scenario.expected_patterns)}\n")
    parts.append(f"**Common pitfalls to avoid:** {', '.join(scenario.common_pitfalls)}\n")
    parts.append(f"**Acceptance criteria:** {', '.join(scenario.acceptance_criteria)}\n")

    parts.append("## Agent's Plan\n")
    parts.append(plan_text)
    parts.append("")

    parts.append(
        "## Instructions\n"
        "Evaluate the plan against the ground truth above. Return your evaluation "
        "as a JSON object with exactly these fields:\n\n"
        "```json\n"
        "{\n"
        '  "file_identification": 0.0,\n'
        '  "approach_correctness": 0.0,\n'
        '  "pattern_awareness": 0.0,\n'
        '  "pitfall_avoidance": 0.0,\n'
        '  "completeness": 0.0,\n'
        '  "rationale": "Brief explanation of scores"\n'
        "}\n"
        "```\n\n"
        "Each score must be between 0.0 and 1.0. Be precise and fair."
    )

    return "\n".join(parts)


def parse_agent_response(raw: str) -> str:
    """Extract the plan text from the agent's raw response.

    Currently a pass-through but provides a hook for future parsing logic
    (e.g. if we ask the agent to wrap the plan in delimiters).
    """
    return raw.strip()


def parse_score_card_json(raw: str) -> ScoreCard:
    """Parse the scoring LLM's JSON response into a ScoreCard.

    Handles JSON embedded in markdown code fences.
    """
    from .scoring import compute_overall_score

    text = raw.strip()

    # Strip markdown code fences
    match = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
    if match:
        text = match.group(1).strip()

    # Try to find a JSON object in the text
    brace_match = re.search(r"\{.*\}", text, re.DOTALL)
    if brace_match:
        text = brace_match.group(0)

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return ScoreCard(rationale=f"Failed to parse scoring response: {raw[:200]}")

    card = ScoreCard(
        file_identification=_clamp(data.get("file_identification", 0.0)),
        approach_correctness=_clamp(data.get("approach_correctness", 0.0)),
        pattern_awareness=_clamp(data.get("pattern_awareness", 0.0)),
        pitfall_avoidance=_clamp(data.get("pitfall_avoidance", 0.0)),
        completeness=_clamp(data.get("completeness", 0.0)),
        rationale=str(data.get("rationale", "")),
    )
    card.overall = compute_overall_score(card)
    return card


def _clamp(value: object) -> float:
    """Clamp a value to [0.0, 1.0]."""
    try:
        v = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, v))
