"""Skill file testing and improvement via the DataRobot LLM gateway.

Skills are markdown files that define how an AI coding assistant should handle
a specific type of task (e.g. writing commit messages, reviewing PRs). This
module evaluates whether a skill prompt would produce high-quality, consistent
results and can apply LLM feedback to improve it.
"""

import re
import sys
from pathlib import Path

from .config import Config
from .llm import call_llm

REPORT_SUFFIX = ".skill-report.md"
NOTES_SUFFIX = ".skill-revision-notes.md"


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------


def _extract_allowed_tools(skill_content: str) -> list[str]:
    """Parse the allowed-tools list from YAML frontmatter, if present."""
    fm_match = re.match(r"^---\n(.*?\n)---\n", skill_content, re.DOTALL)
    if not fm_match:
        return []
    fm = fm_match.group(1)
    line_match = re.search(r"^allowed-tools:\s*(.+)$", fm, re.MULTILINE)
    if not line_match:
        return []
    raw = line_match.group(1).strip()
    # Handle inline list:  tool1, tool2, tool3
    return [t.strip() for t in raw.split(",") if t.strip()]


def _build_skill_test_prompt(skill_name: str, skill_content: str) -> str:
    allowed_tools = _extract_allowed_tools(skill_content)

    if allowed_tools:
        tool_list = "\n".join(f"  - {t}" for t in allowed_tools)
        tool_context = f"""
## Agent execution context

This skill will be loaded by an AI agent (e.g. Claude, Cursor, or a custom agent) that has
access to **only** the following tools — nothing else:

{tool_list}

The agent has NO shell access, NO ability to run CLI commands, NO filesystem access beyond
what those tools expose, and NO ability to install packages. It cannot browse the web, run
scripts, or call any API not covered by the tools above.

**Critically evaluate every step** in the skill against this tool list. If the skill instructs
the agent to do something that requires a capability not in the list above, that is a fatal
flaw — the agent will either stall, hallucinate a capability, or produce wrong results.
"""
    else:
        tool_context = ""

    return f"""\
You are an expert prompt engineer evaluating AI agent skills.

A "skill" is a markdown instruction file that tells an AI agent (such as Claude, Cursor, or a
custom coding agent) how to handle a specific task. The skill below is named "{skill_name}".

<skill>
{skill_content}
</skill>
{tool_context}
## Your task

Evaluate this skill from two perspectives:

### 1. As an AI agent following the skill
Mentally simulate using these instructions for 2-3 representative tasks this skill covers.
- Are the instructions clear enough to follow without ambiguity?
- Would two different agents following this skill produce consistent results?
- Are there edge cases the skill doesn't address that would leave an agent guessing?
- (If allowed-tools are declared) Can every step be completed with only the listed tools?

### 2. As a prompt engineer reviewing the skill
- Is the purpose of the skill stated clearly upfront?
- Are instructions specific and actionable, or vague and hand-wavy?
- Is there unnecessary complexity or padding that dilutes the key guidance?
- Does the skill define a clear output format where one is needed?

## Output format

Respond with a structured report in markdown:

### What works well
Specific strengths — instructions that are clear, well-scoped, or particularly effective.

### What is unclear or missing
Concrete gaps or ambiguities. For each one, suggest exactly what text would fix it.

### Suggested improvements
Specific rewrites or additions that would meaningfully improve the skill. Write the actual
text where possible, not just "add more detail."

### Overall verdict
One of: GOOD (minor tweaks only) / NEEDS WORK (several meaningful gaps) / INCOMPLETE
(an agent would produce inconsistent or poor results). One sentence explaining why.

## Verdict format

The verdict section is read by an automated test, so its shape matters:

- Reproduce the heading exactly as `### Overall verdict`. Do not add a document title
  above it or shift the report's heading levels.
- Start the line directly beneath the heading with the verdict label itself — nothing
  before it — then the explanation.
- Use the label verbatim: `GOOD`, `NEEDS WORK`, or `INCOMPLETE`.

For example:

### Overall verdict
NEEDS WORK — the skill leaves error handling undefined, so two agents would diverge.
"""


def _build_skill_improve_prompt(skill_name: str, skill_content: str, report: str) -> str:
    return f"""\
You are an expert prompt engineer. You wrote the following skill named "{skill_name}":

<skill>
{skill_content}
</skill>

A peer reviewer gave the following critique:

<critique>
{report}
</critique>

Your task:
1. For each critique point that is **valid and actionable**, incorporate the improvement.
2. For each critique point that is **wrong, already covered, or not relevant**, note it in the
   revision notes — do not add unnecessary content to the skill itself.
3. Output your response in exactly this format:

===SKILL===
<the complete, improved skill file — clean markdown only, no reviewer comments>
===REVISION_NOTES===
<a brief list of which critique points you rejected and why>

Preserve the existing tone and structure. Do not pad with unnecessary content.
"""


# ---------------------------------------------------------------------------
# Skills class
# ---------------------------------------------------------------------------


class Skills:
    """Test and improve AI coding skill files via the DataRobot LLM gateway.

    Example::

        from dr_agents_tester import Config, Skills
        from pathlib import Path

        cfg = Config()
        cfg.validate()
        skills = Skills(cfg)

        skills.test(Path("skills/commit.md"))             # evaluate and save report
        skills.improve(Path("skills/commit.md"))          # apply feedback
        skills.test_all(Path("skills/"))                  # test every skill in a directory
    """

    def __init__(self, config: Config) -> None:
        self.config = config

    def test(
        self,
        skill_path: Path,
        *,
        test_model: str | None = None,
    ) -> str:
        """Evaluate a skill file. Prints and saves a critique report.

        Returns the report text.
        """
        skill_path = Path(skill_path).resolve()
        if not skill_path.exists():
            print(f"❌ Skill file not found: {skill_path}", file=sys.stderr)
            sys.exit(1)

        use_model = test_model or self.config.test_model
        skill_name = skill_path.stem
        skill_content = skill_path.read_text(encoding="utf-8", errors="replace")

        print(f"🧪 Evaluating skill: {skill_path.name}")
        print(f"💬 Evaluating with [{use_model}]...")

        prompt = _build_skill_test_prompt(skill_name, skill_content)
        report = call_llm(prompt, use_model, self.config)

        report_path = skill_path.parent / (skill_path.stem + REPORT_SUFFIX)
        report_path.write_text(report)

        print(f"\n{'=' * 60}")
        print(f"SKILL TEST REPORT — {skill_path.name}")
        print(f"{'=' * 60}\n")
        print(report)
        print(f"\n💾 Report saved to {report_path} — run `improve` to apply feedback.")

        return report

    def improve(
        self,
        skill_path: Path,
        *,
        dry_run: bool = False,
        model: str | None = None,
    ) -> str:
        """Apply saved test report feedback to a skill file. Returns updated content."""
        skill_path = Path(skill_path).resolve()
        if not skill_path.exists():
            print(f"❌ Skill file not found: {skill_path}", file=sys.stderr)
            sys.exit(1)

        report_path = skill_path.parent / (skill_path.stem + REPORT_SUFFIX)
        if not report_path.exists():
            print(
                f"❌ No test report found at {report_path}. Run `test` first.",
                file=sys.stderr,
            )
            sys.exit(1)

        use_model = model or self.config.model
        skill_name = skill_path.stem
        skill_content = skill_path.read_text(encoding="utf-8", errors="replace")
        report = report_path.read_text()

        print(f"✏️  Improving skill: {skill_path.name}")
        print(f"💬 Improving with [{use_model}]...")

        raw = call_llm(
            _build_skill_improve_prompt(skill_name, skill_content, report),
            use_model,
            self.config,
        )

        if "===SKILL===" in raw and "===REVISION_NOTES===" in raw:
            skill_part = raw.split("===SKILL===")[1].split("===REVISION_NOTES===")[0].strip()
            notes_part = raw.split("===REVISION_NOTES===")[1].strip()
        else:
            skill_part = raw.strip()
            notes_part = "(Model did not produce separate revision notes.)"

        if notes_part and not dry_run:
            notes_path = skill_path.parent / (skill_path.stem + NOTES_SUFFIX)
            notes_path.write_text(notes_part)
            print(f"📝 Revision notes saved to {notes_path}")

        if dry_run:
            print(f"\n{'=' * 60}\nDRY RUN — would write to: {skill_path}\n{'=' * 60}\n")
            print(skill_part)
        else:
            skill_path.write_text(skill_part)
            print(f"✅ Written: {skill_path}")

        return skill_part

    def test_all(
        self,
        skills_dir: Path,
        *,
        test_model: str | None = None,
        glob: str = "*.md",
    ) -> dict[str, str]:
        """Evaluate every skill file matching glob in skills_dir.

        Returns a dict mapping skill filename -> report text.
        """
        skills_dir = Path(skills_dir).resolve()
        skill_files = sorted(skills_dir.glob(glob))

        # Exclude report and notes files generated by this tool
        skill_files = [
            f
            for f in skill_files
            if not f.name.endswith(REPORT_SUFFIX) and not f.name.endswith(NOTES_SUFFIX)
        ]

        if not skill_files:
            print(f"⚠️  No skill files found in {skills_dir}", file=sys.stderr)
            return {}

        results: dict[str, str] = {}
        for skill_path in skill_files:
            print(f"\n{'─' * 60}")
            results[skill_path.name] = self.test(skill_path, test_model=test_model)

        print(f"\n{'=' * 60}")
        print(f"Tested {len(results)} skill(s) in {skills_dir}")
        return results
