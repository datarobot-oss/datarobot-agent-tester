"""Tests for skill file testing and improvement."""

from pathlib import Path
from unittest.mock import patch

import pytest

from dr_agents_tester.skills import NOTES_SUFFIX, REPORT_SUFFIX, Skills


class TestSkillsTest:
    def test_test_saves_report(
        self,
        tmp_path: Path,
        fake_config: "Config",  # noqa: F821
    ) -> None:
        skill = tmp_path / "commit.md"
        skill.write_text("# Skill: Commit\nWrite good commits.")

        with patch("dr_agents_tester.skills.call_llm") as mock_llm:
            mock_llm.return_value = "### Overall verdict\nGOOD - clear instructions."
            s = Skills(fake_config)
            report = s.test(skill)

        assert "GOOD" in report
        report_path = tmp_path / ("commit" + REPORT_SUFFIX)
        assert report_path.exists()
        assert report_path.read_text() == report

    def test_test_missing_file_exits(
        self,
        tmp_path: Path,
        fake_config: "Config",  # noqa: F821
    ) -> None:
        s = Skills(fake_config)
        with pytest.raises(SystemExit):
            s.test(tmp_path / "nonexistent.md")

    def test_test_uses_test_model(
        self,
        tmp_path: Path,
        fake_config: "Config",  # noqa: F821
    ) -> None:
        skill = tmp_path / "review.md"
        skill.write_text("# Review skill")

        with patch("dr_agents_tester.skills.call_llm") as mock_llm:
            mock_llm.return_value = "GOOD"
            s = Skills(fake_config)
            s.test(skill, test_model="datarobot/custom-model")

        call_args = mock_llm.call_args
        assert call_args[0][1] == "datarobot/custom-model"


class TestSkillsImprove:
    def test_improve_writes_updated_skill(
        self,
        tmp_path: Path,
        fake_config: "Config",  # noqa: F821
    ) -> None:
        skill = tmp_path / "commit.md"
        skill.write_text("# Skill: Commit\nOld content.")
        report_path = tmp_path / ("commit" + REPORT_SUFFIX)
        report_path.write_text("NEEDS WORK: add examples")

        improved_content = "# Skill: Commit\nImproved content with examples."
        with patch("dr_agents_tester.skills.call_llm") as mock_llm:
            mock_llm.return_value = (
                f"===SKILL===\n{improved_content}\n"
                "===REVISION_NOTES===\nAdded examples as suggested."
            )
            s = Skills(fake_config)
            result = s.improve(skill)

        assert result == improved_content
        assert skill.read_text() == improved_content
        notes_path = tmp_path / ("commit" + NOTES_SUFFIX)
        assert notes_path.exists()

    def test_improve_dry_run_does_not_write(
        self,
        tmp_path: Path,
        fake_config: "Config",  # noqa: F821
    ) -> None:
        skill = tmp_path / "commit.md"
        original = "# Original content"
        skill.write_text(original)
        (tmp_path / ("commit" + REPORT_SUFFIX)).write_text("NEEDS WORK")

        with patch("dr_agents_tester.skills.call_llm") as mock_llm:
            mock_llm.return_value = "===SKILL===\n# New content\n===REVISION_NOTES===\nchanged"
            s = Skills(fake_config)
            s.improve(skill, dry_run=True)

        assert skill.read_text() == original

    def test_improve_missing_report_exits(
        self,
        tmp_path: Path,
        fake_config: "Config",  # noqa: F821
    ) -> None:
        skill = tmp_path / "commit.md"
        skill.write_text("# content")
        s = Skills(fake_config)
        with pytest.raises(SystemExit):
            s.improve(skill)

    def test_improve_handles_missing_delimiter(
        self,
        tmp_path: Path,
        fake_config: "Config",  # noqa: F821
    ) -> None:
        skill = tmp_path / "commit.md"
        skill.write_text("# content")
        (tmp_path / ("commit" + REPORT_SUFFIX)).write_text("feedback")

        with patch("dr_agents_tester.skills.call_llm") as mock_llm:
            mock_llm.return_value = "# Improved without delimiter"
            s = Skills(fake_config)
            result = s.improve(skill)

        assert result == "# Improved without delimiter"


class TestSkillsTestAll:
    def test_test_all_tests_each_md_file(
        self,
        tmp_path: Path,
        fake_config: "Config",  # noqa: F821
    ) -> None:
        (tmp_path / "commit.md").write_text("# Commit skill")
        (tmp_path / "review.md").write_text("# Review skill")
        # Report files should be excluded
        (tmp_path / ("commit" + REPORT_SUFFIX)).write_text("prior report")

        with patch("dr_agents_tester.skills.call_llm") as mock_llm:
            mock_llm.return_value = "GOOD"
            s = Skills(fake_config)
            results = s.test_all(tmp_path)

        assert "commit.md" in results
        assert "review.md" in results
        assert len(results) == 2  # report file excluded

    def test_test_all_empty_dir_returns_empty(
        self,
        tmp_path: Path,
        fake_config: "Config",  # noqa: F821
    ) -> None:
        s = Skills(fake_config)
        results = s.test_all(tmp_path)
        assert results == {}
