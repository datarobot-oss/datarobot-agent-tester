"""Tests for verdict parsing in the pytest plugin."""

import pytest

from dr_agents_tester.pytest_plugin import parse_verdict, verdict_passes

# Abridged from the real report that broke CI in datarobot-agent-skills: the
# model wrote the whole report one heading level up (`##` instead of `###`) and
# the word "incomplete" appears in the prose several times, including inside
# the verdict sentence itself.
H2_REPORT = """\
# Skill Evaluation Report: `datarobot-agent-assist`

---

## What is unclear or missing

### Gap 1: Missing reference files

> If missing, tell the user: "The skill reference file is missing — the skill
> installation may be incomplete." and stop.

### Gap 5: "Welcome menu reset" state changes are incomplete

The current rule is abstract.

---

## Overall Verdict

**NEEDS WORK** — The skill is architecturally sound, but has several concrete
gaps (undefined behavior for missing reference files, incomplete Windows
guards, and a stub Dress Rehearsal section) that would cause two agents to
produce meaningfully different behavior.
"""


class TestHeadingLevels:
    """The verdict must be found regardless of what heading level the LLM picks."""

    @pytest.mark.parametrize("hashes", ["#", "##", "###", "####", "#####", "######"])
    def test_any_heading_level(self, hashes: str) -> None:
        report = f"{hashes} Overall verdict\n\n**GOOD** — clear and complete.\n"
        assert parse_verdict(report) == "GOOD"

    def test_title_case_heading(self) -> None:
        assert parse_verdict("## Overall Verdict\n\nINCOMPLETE — too vague.\n") == "INCOMPLETE"

    def test_bold_heading_without_hashes(self) -> None:
        assert parse_verdict("**Overall verdict:**\n\nNEEDS WORK — a few gaps.\n") == "NEEDS WORK"

    def test_inline_verdict_on_heading_line(self) -> None:
        assert parse_verdict("### Overall verdict: NEEDS WORK\n") == "NEEDS WORK"

    def test_heading_with_trailing_text_falls_through_to_next_line(self) -> None:
        report = "## Overall Verdict (final)\n\n**NEEDS WORK** — a few gaps.\n"
        assert parse_verdict(report) == "NEEDS WORK"

    def test_blank_lines_between_heading_and_verdict(self) -> None:
        assert parse_verdict("## Overall verdict\n\n\n\nGOOD — solid.\n") == "GOOD"


class TestPrecedence:
    """The verdict label wins over stray keywords in the explanation."""

    def test_h2_report_from_ci_failure(self) -> None:
        # This is the regression: an `##`-level heading used to fall through to
        # a whole-report scan, which hit "incomplete" in the prose first.
        assert parse_verdict(H2_REPORT) == "NEEDS WORK"
        assert verdict_passes(H2_REPORT)

    def test_incomplete_in_explanation_does_not_win(self) -> None:
        report = "### Overall verdict\n\n**NEEDS WORK** — results may be incomplete.\n"
        assert parse_verdict(report) == "NEEDS WORK"

    def test_needs_work_in_explanation_does_not_win(self) -> None:
        report = "### Overall verdict\n\n**INCOMPLETE** — this skill needs work.\n"
        assert parse_verdict(report) == "INCOMPLETE"

    def test_good_in_explanation_does_not_win(self) -> None:
        report = "### Overall verdict\n\nNEEDS WORK — the structure is good but thin.\n"
        assert parse_verdict(report) == "NEEDS WORK"

    def test_hyphenated_needs_work(self) -> None:
        assert parse_verdict("### Overall verdict\n\n**NEEDS-WORK** — gaps.\n") == "NEEDS WORK"


class TestUnparseable:
    """A report with no verdict section must not be guessed at from prose."""

    def test_no_verdict_section_is_unknown(self) -> None:
        report = "# Report\n\nThe skill installation may be incomplete.\n"
        assert parse_verdict(report) == "UNKNOWN"
        assert verdict_passes(report)

    def test_empty_report_is_unknown(self) -> None:
        assert parse_verdict("") == "UNKNOWN"

    def test_bare_verdict_line_without_heading(self) -> None:
        report = "Some notes about the skill.\n\n**INCOMPLETE** — an agent would guess.\n"
        assert parse_verdict(report) == "INCOMPLETE"

    def test_heading_with_no_following_content(self) -> None:
        assert parse_verdict("### Overall verdict\n") == "UNKNOWN"


class TestFallbackIgnoresMarkdownStructure:
    """Only a line that *is* the verdict may satisfy the no-heading fallback.

    Headings and list items routinely start with a verdict word by coincidence
    ("### Incomplete Windows guards"), which is the same false-positive class
    the whole fix is about — it must not sneak back in via the fallback.
    """

    def test_gap_heading_starting_with_incomplete(self) -> None:
        report = "# Skill Review\n\n### Incomplete Windows guards\n\nThe guards are partial.\n"
        assert parse_verdict(report) == "UNKNOWN"

    @pytest.mark.parametrize("bullet", ["*", "-", "+"])
    def test_bullet_starting_with_a_verdict_word(self, bullet: str) -> None:
        report = f"# Skill Review\n\n## Strengths\n\n{bullet} Good clear purpose statement.\n"
        assert parse_verdict(report) == "UNKNOWN"

    def test_blockquote_is_not_a_verdict(self) -> None:
        # Reports quote suggested replacement text; it is not the verdict.
        report = '# Review\n\n> INCOMPLETE — say this if the file is missing.\n'
        assert parse_verdict(report) == "UNKNOWN"

    def test_emphasised_bare_verdict_still_matches(self) -> None:
        assert parse_verdict("Notes.\n\n**INCOMPLETE** — an agent would guess.\n") == "INCOMPLETE"
        assert parse_verdict("Notes.\n\n_GOOD_ — clear throughout.\n") == "GOOD"

    def test_bulleted_verdict_under_heading_still_matches(self) -> None:
        # The heading path is unaffected: once the section is found, the line
        # below it is the verdict even if the model bulleted it.
        assert parse_verdict("### Overall verdict\n\n- NEEDS WORK — gaps.\n") == "NEEDS WORK"


class TestVerdictPasses:
    @pytest.mark.parametrize(
        ("verdict", "expected"),
        [("GOOD", True), ("NEEDS WORK", True), ("INCOMPLETE", False)],
    )
    def test_only_incomplete_fails(self, verdict: str, expected: bool) -> None:
        report = f"### Overall verdict\n\n**{verdict}** — because reasons.\n"
        assert verdict_passes(report) is expected
