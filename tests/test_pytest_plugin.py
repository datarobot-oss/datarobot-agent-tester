"""Tests for the pytest plugin: verdict parsing and the skill quality gate.

The reports under ``tests/fixtures/verdict_reports/`` are captured verbatim from
real judge runs. They pin the formatting drift that broke the gate: the judge
promoting the verdict heading to H2, and the word "incomplete" appearing both in
body prose and inside the verdict sentence itself. Replaying them here is the
acceptance test for the parser — a live CI run cannot verify it, because the
outcome depends on which formatting the judge happens to pick.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from dr_agents_tester import pytest_plugin
from dr_agents_tester.pytest_plugin import parse_verdict, verdict_passes
from dr_agents_tester.skills import VERDICT_PREFIX, Skills, _build_skill_test_prompt

VERDICT_FIXTURES = Path(__file__).parent / "fixtures" / "verdict_reports"


# ---------------------------------------------------------------------------
# parse_verdict — captured reports
# ---------------------------------------------------------------------------


class TestParseVerdictCapturedReports:
    @pytest.mark.parametrize(
        "fixture_name",
        [
            # H1 title + H2 verdict heading; "incomplete" in body prose only
            "h2_needs_work_prose_incomplete.md",
            # same shape, but "incomplete" inside the verdict sentence itself
            "h2_needs_work_verdict_line_incomplete.md",
            # live run against the sentinel prompt: the verdict paragraph under
            # the heading carries no leading token, only the sentinel does
            "sentinel_needs_work.md",
        ],
    )
    def test_captured_needs_work_reports_parse_as_needs_work(
        self, fixture_name: str
    ) -> None:
        report = (VERDICT_FIXTURES / fixture_name).read_text(encoding="utf-8")
        assert parse_verdict(report) == "NEEDS WORK"
        assert verdict_passes(report) is True


# ---------------------------------------------------------------------------
# parse_verdict — sentinel line (primary)
# ---------------------------------------------------------------------------


class TestParseVerdictSentinel:
    @pytest.mark.parametrize(
        ("report", "expected"),
        [
            ("Some report body.\n\nVERDICT: GOOD", "GOOD"),
            ("Some report body.\n\nVERDICT: NEEDS WORK", "NEEDS WORK"),
            ("Some report body.\n\nVERDICT: INCOMPLETE", "INCOMPLETE"),
            # judges bold things
            ("body\n\n**VERDICT: NEEDS WORK**", "NEEDS WORK"),
            # trailing period / whitespace
            ("body\n\nVERDICT: GOOD.\n", "GOOD"),
            # case-insensitive
            ("body\n\nverdict: needs work", "NEEDS WORK"),
            # trailing explanation after a real separator is tolerated
            ("body\n\nVERDICT: GOOD — minor tweaks only", "GOOD"),
            ("body\n\nVERDICT: NEEDS WORK - see gaps above", "NEEDS WORK"),
        ],
    )
    def test_sentinel_line_is_parsed(self, report: str, expected: str) -> None:
        assert parse_verdict(report) == expected

    def test_ambiguous_continuation_is_not_a_verdict(self) -> None:
        # a bare continuation is ambiguity, not a separator; must retry
        assert parse_verdict("body\n\nVERDICT: GOOD or maybe NEEDS WORK") == "UNKNOWN"

    def test_sentinel_quoted_mid_report_cannot_stand_in_for_the_final_line(
        self,
    ) -> None:
        # a verdict-shaped line quoted in the body (skills under test define
        # report formats of their own) must not rescue a malformed final line
        report = (
            "The skill asks for\n"
            "    VERDICT: GOOD\n"
            "at the end of its own reports.\n\n"
            "More analysis here.\n"
            "Closing remarks.\n"
            "Final thoughts on structure.\n\n"
            "VERDICT: GOOOD\n"
        )
        assert parse_verdict(report) == "UNKNOWN"

    def test_sentinel_wins_over_heading(self) -> None:
        # the sentinel is the enforced contract; the heading section is the
        # human-readable artifact and may disagree after manual edits
        report = "### Overall verdict\nGOOD - fine.\n\nVERDICT: INCOMPLETE"
        assert parse_verdict(report) == "INCOMPLETE"

    def test_last_sentinel_wins(self) -> None:
        report = "VERDICT: GOOD\n\nmore analysis\n\nVERDICT: NEEDS WORK"
        assert parse_verdict(report) == "NEEDS WORK"

    def test_echoed_template_placeholder_is_not_a_verdict(self) -> None:
        # a judge that parrots the instruction instead of choosing must not
        # accidentally parse as GOOD (first token in the alternation)
        report = "body\n\nVERDICT: <GOOD or NEEDS WORK or INCOMPLETE>"
        assert parse_verdict(report) == "UNKNOWN"

    def test_verdict_token_mid_sentence_is_not_a_verdict(self) -> None:
        report = "We considered VERDICT: GOOD as an outcome but the report is unfinished."
        assert parse_verdict(report) == "UNKNOWN"


# ---------------------------------------------------------------------------
# parse_verdict — heading fallback (pre-sentinel reports)
# ---------------------------------------------------------------------------


class TestParseVerdictHeadingFallback:
    @pytest.mark.parametrize(
        ("report", "expected"),
        [
            # the exact legacy shape asserted elsewhere in this suite
            ("### Overall verdict\nGOOD - clear instructions.", "GOOD"),
            # any heading level
            ("# Overall Verdict\nGOOD", "GOOD"),
            ("## Overall Verdict\n\n**INCOMPLETE** — an agent would fail.", "INCOMPLETE"),
            # verdict inline on the heading line
            ("## Overall Verdict: **INCOMPLETE**", "INCOMPLETE"),
            # lowercase heading
            ("## overall verdict\nneeds work - meh", "NEEDS WORK"),
            # blockquoted verdict line
            ("## Overall Verdict\n> **NEEDS WORK** — gaps", "NEEDS WORK"),
            # a hyphenated skill name in the verdict line must not truncate it
            ("## Overall Verdict\n**GOOD** — datarobot-agent-assist is clear", "GOOD"),
            # no dash separator at all
            ("## Overall Verdict\nNEEDS WORK. Some parts are incomplete.", "NEEDS WORK"),
            # colon separator
            ("## Overall Verdict\n**GOOD**: no incomplete areas found", "GOOD"),
        ],
    )
    def test_heading_verdicts_parse(self, report: str, expected: str) -> None:
        assert parse_verdict(report) == expected

    def test_prose_mentioning_incomplete_cannot_poison_the_verdict(self) -> None:
        report = (
            "## Notes\n"
            "If missing, tell the user the skill installation is incomplete and stop.\n\n"
            "## Overall Verdict\n"
            "**GOOD** — fine\n"
        )
        assert parse_verdict(report) == "GOOD"

    def test_no_heading_and_no_sentinel_is_unknown(self) -> None:
        # the word "incomplete" in prose must not be scanned as a verdict
        report = "# Report\nThe setup instructions are incomplete in places."
        assert parse_verdict(report) == "UNKNOWN"

    def test_heading_with_no_token_on_verdict_line_is_unknown(self) -> None:
        report = "## Overall Verdict\nThe skill needs improvement but is serviceable."
        assert parse_verdict(report) == "UNKNOWN"

    def test_last_heading_wins_over_a_quoted_earlier_one(self) -> None:
        # judges quote skill text; a quoted verdict heading earlier in the
        # report must not shadow the real one at the end
        report = (
            "### Overall verdict\n"
            "GOOD - quoted from the skill's own report template\n\n"
            "More analysis.\n\n"
            "### Overall verdict\n"
            "INCOMPLETE - the real verdict\n"
        )
        assert parse_verdict(report) == "INCOMPLETE"


# ---------------------------------------------------------------------------
# verdict_passes
# ---------------------------------------------------------------------------


class TestVerdictPasses:
    def test_good_passes(self) -> None:
        assert verdict_passes("VERDICT: GOOD") is True

    def test_needs_work_passes(self) -> None:
        assert verdict_passes("VERDICT: NEEDS WORK") is True

    def test_incomplete_fails(self) -> None:
        assert verdict_passes("VERDICT: INCOMPLETE") is False

    def test_unknown_fails_closed(self) -> None:
        # a gate that cannot read the verdict must not silently pass
        assert verdict_passes("no verdict anywhere in this text") is False


# ---------------------------------------------------------------------------
# prompt <-> parser coupling
# ---------------------------------------------------------------------------


class TestPromptContract:
    def test_prompt_demands_the_sentinel_the_parser_reads(self) -> None:
        prompt = _build_skill_test_prompt("commit", "# skill body")
        assert VERDICT_PREFIX in prompt

    def test_prompt_verdict_heading_matches_the_fallback_pattern(self) -> None:
        # if the prompt's heading wording changes, the compat fallback dies
        # silently in production; fail here instead
        prompt = _build_skill_test_prompt("commit", "# skill body")
        assert pytest_plugin._HEADING_RE.search(prompt) is not None

    def test_prompt_itself_contains_no_parseable_verdict(self) -> None:
        # the prompt demonstrates the sentinel form; if the example ever
        # becomes parseable, a judge echoing the instructions verbatim would
        # produce a false verdict. the placeholder must stay unparseable
        prompt = _build_skill_test_prompt("commit", "# skill body")
        assert parse_verdict(prompt) == "UNKNOWN"


# ---------------------------------------------------------------------------
# the gate end-to-end (pytester)
# ---------------------------------------------------------------------------

_GATE_MODULE = """
from pathlib import Path
from dr_agents_tester.pytest_plugin import make_skill_e2e_test

pytest_generate_tests, test_skill_quality, _hash_store = make_skill_e2e_test(
    skills_dir=Path({skills_dir!r}),
    hash_file=Path({hash_file!r}),
)
"""

_PASSING_REPORT = "## Overall Verdict\n\n**NEEDS WORK** — gaps\n\nVERDICT: NEEDS WORK\n"
_INCOMPLETE_REPORT = "## Overall Verdict\n\n**INCOMPLETE** — broken\n\nVERDICT: INCOMPLETE\n"
_UNPARSEABLE_REPORT = "A thorough review. The setup section is incomplete in places.\n"


class TestSkillGate:
    @pytest.fixture(autouse=True)
    def _fake_credentials(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("DATAROBOT_API_TOKEN", "fake-token")
        monkeypatch.delenv("SKILLS_E2E_FORCE_ALL", raising=False)

    def _make_gate(self, pytester: pytest.Pytester) -> Path:
        skills_dir = pytester.path / "skills"
        (skills_dir / "demo").mkdir(parents=True)
        (skills_dir / "demo" / "SKILL.md").write_text("# demo skill\n")
        hash_file = pytester.path / "skill_hashes.json"
        pytester.makepyfile(
            test_gate=_GATE_MODULE.format(
                skills_dir=str(skills_dir), hash_file=str(hash_file)
            )
        )
        return hash_file

    def _fake_judge(
        self, monkeypatch: pytest.MonkeyPatch, reports: list[str]
    ) -> list[str]:
        calls: list[str] = []

        def fake_test(
            self: Skills, skill_path: Path, *, test_model: str | None = None
        ) -> str:
            calls.append(str(skill_path))
            return reports[min(len(calls), len(reports)) - 1]

        monkeypatch.setattr(Skills, "test", fake_test)
        return calls

    def test_passing_verdict_updates_hash_and_skips_next_run(
        self, pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        hash_file = self._make_gate(pytester)
        calls = self._fake_judge(monkeypatch, [_PASSING_REPORT])

        result = pytester.runpytest("test_gate.py")
        result.assert_outcomes(passed=1)
        assert len(calls) == 1
        assert "skills/demo/SKILL.md" in json.loads(hash_file.read_text())

        # unchanged skill: hash matches, no judge call
        result = pytester.runpytest("test_gate.py")
        result.assert_outcomes(skipped=1)
        assert len(calls) == 1

    def test_incomplete_verdict_fails_and_does_not_update_hash(
        self, pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        hash_file = self._make_gate(pytester)
        self._fake_judge(monkeypatch, [_INCOMPLETE_REPORT])

        result = pytester.runpytest("test_gate.py")
        result.assert_outcomes(failed=1)
        result.stdout.fnmatch_lines(["*is INCOMPLETE*"])
        assert json.loads(hash_file.read_text()) == {}

    def test_unparseable_verdict_retries_once_then_fails(
        self, pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        hash_file = self._make_gate(pytester)
        calls = self._fake_judge(monkeypatch, [_UNPARSEABLE_REPORT])

        result = pytester.runpytest("test_gate.py")
        result.assert_outcomes(failed=1)
        result.stdout.fnmatch_lines(["*did not emit a parseable verdict*"])
        assert len(calls) == 2  # one retry, then fail closed
        assert json.loads(hash_file.read_text()) == {}

    def test_unparseable_first_try_recovers_on_retry(
        self, pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        hash_file = self._make_gate(pytester)
        calls = self._fake_judge(monkeypatch, [_UNPARSEABLE_REPORT, _PASSING_REPORT])

        result = pytester.runpytest("test_gate.py")
        result.assert_outcomes(passed=1)
        assert len(calls) == 2
        assert "skills/demo/SKILL.md" in json.loads(hash_file.read_text())
