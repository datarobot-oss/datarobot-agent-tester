"""Tests for AGENTS.md generation, marker handling, and file management."""

from pathlib import Path
from unittest.mock import patch

from dr_agents_tester.agents_md import (
    MARKER_END,
    MARKER_START,
    AgentsMd,
    apply_generated_content,
    clean_generated,
    strip_markers,
)


class TestStripMarkers:
    def test_strips_start_and_end_markers(self) -> None:
        text = f"{MARKER_START}\n# Content\n{MARKER_END}"
        assert strip_markers(text) == "# Content"

    def test_strips_custom_content_comment(self) -> None:
        text = "<!-- Add custom content below this line. foo -->\n# Real content"
        result = strip_markers(text)
        assert "Add custom content" not in result
        assert "# Real content" in result

    def test_no_markers_returns_stripped(self) -> None:
        text = "# Plain content\n\nWith paragraphs"
        assert strip_markers(text) == text

    def test_empty_string(self) -> None:
        assert strip_markers("") == ""


class TestCleanGenerated:
    def test_strips_markdown_code_fence(self) -> None:
        text = "```markdown\n# Content\n```"
        assert clean_generated(text) == "# Content"

    def test_strips_plain_code_fence(self) -> None:
        text = "```\n# Content\n```"
        assert clean_generated(text) == "# Content"

    def test_strips_md_language_fence(self) -> None:
        text = "```md\n# Content\n```"
        assert clean_generated(text) == "# Content"

    def test_no_fence_returns_as_is(self) -> None:
        text = "# Content\nNo fences here."
        assert clean_generated(text) == text

    def test_strips_markers_from_llm_echo(self) -> None:
        text = f"{MARKER_START}\n# Content\n{MARKER_END}"
        assert clean_generated(text) == "# Content"


class TestApplyGeneratedContent:
    def test_wraps_new_file_with_markers(self) -> None:
        result = apply_generated_content("", "# Generated")
        assert MARKER_START in result
        assert MARKER_END in result
        assert "# Generated" in result

    def test_replaces_existing_marker_block(self) -> None:
        existing = f"{MARKER_START}\n# Old content\n{MARKER_END}\n\nCustom stuff"
        result = apply_generated_content(existing, "# New content")
        assert "# New content" in result
        assert "# Old content" not in result
        assert "Custom stuff" in result

    def test_preserves_custom_content_outside_markers(self) -> None:
        existing = (
            f"{MARKER_START}\n# Auto\n{MARKER_END}\n\n"
            "<!-- Add custom content below this line. -->\n\n# My custom section"
        )
        result = apply_generated_content(existing, "# Updated auto")
        assert "# My custom section" in result
        assert "# Updated auto" in result

    def test_existing_custom_content_appended_on_first_write(self) -> None:
        existing = "# Handwritten content"
        result = apply_generated_content(existing, "# Generated")
        assert "# Handwritten content" in result
        assert "# Generated" in result

    def test_uses_rindex_for_outermost_end_marker(self) -> None:
        # Edge case: nested marker in content (model echoed it back)
        existing = f"{MARKER_START}\n# Content with {MARKER_END} embedded\n{MARKER_END}"
        result = apply_generated_content(existing, "# Clean")
        assert result.count(MARKER_START) == 1
        assert result.count(MARKER_END) == 1


class TestAgentsMdGenerate:
    def test_generate_calls_llm_and_writes_file(
        self,
        tmp_path: Path,
        fake_config: "Config",  # noqa: F821
    ) -> None:
        (tmp_path / ".git").mkdir()

        with (
            patch("dr_agents_tester.agents_md.gather_context") as mock_ctx,
            patch("dr_agents_tester.agents_md.call_llm") as mock_llm,
        ):
            mock_ctx.return_value = {
                "target_dir": ".",
                "is_root": True,
                "tree": "  README.md",
                "file_contents": {},
                "copier_answers": {},
                "sibling_agents_md": {},
            }
            mock_llm.return_value = "# Generated AGENTS.md\n\nSome content."

            mgr = AgentsMd(fake_config)
            mgr.generate(tmp_path, tmp_path, no_copilot=True)

        agents_md = tmp_path / "AGENTS.md"
        assert agents_md.exists()
        content = agents_md.read_text()
        assert MARKER_START in content
        assert "# Generated AGENTS.md" in content

    def test_generate_dry_run_does_not_write(
        self,
        tmp_path: Path,
        fake_config: "Config",  # noqa: F821
    ) -> None:
        (tmp_path / ".git").mkdir()

        with (
            patch("dr_agents_tester.agents_md.gather_context") as mock_ctx,
            patch("dr_agents_tester.agents_md.call_llm") as mock_llm,
        ):
            mock_ctx.return_value = {
                "target_dir": ".",
                "is_root": True,
                "tree": "",
                "file_contents": {},
                "copier_answers": {},
                "sibling_agents_md": {},
            }
            mock_llm.return_value = "# Dry run content"

            mgr = AgentsMd(fake_config)
            mgr.generate(tmp_path, tmp_path, dry_run=True, no_copilot=True)

        assert not (tmp_path / "AGENTS.md").exists()


class TestAgentsMdTest:
    def test_test_saves_report(
        self,
        tmp_path: Path,
        fake_config: "Config",  # noqa: F821
    ) -> None:
        (tmp_path / ".git").mkdir()
        agents_md = tmp_path / "AGENTS.md"
        agents_md.write_text(f"{MARKER_START}\n# My project\n{MARKER_END}\n")

        with (
            patch("dr_agents_tester.agents_md.gather_context") as mock_ctx,
            patch("dr_agents_tester.agents_md.call_llm") as mock_llm,
        ):
            mock_ctx.return_value = {
                "target_dir": ".",
                "is_root": True,
                "tree": "",
                "file_contents": {},
                "copier_answers": {},
                "sibling_agents_md": {},
            }
            mock_llm.return_value = "### Overall verdict\nGOOD"

            mgr = AgentsMd(fake_config)
            report = mgr.test(tmp_path, tmp_path)

        assert "GOOD" in report
        assert (tmp_path / ".agents-md-report.md").exists()


class TestAgentsMdRevise:
    def test_revise_updates_agents_md(
        self,
        tmp_path: Path,
        fake_config: "Config",  # noqa: F821
    ) -> None:
        (tmp_path / ".git").mkdir()
        (tmp_path / "AGENTS.md").write_text(f"{MARKER_START}\n# Old\n{MARKER_END}\n")
        (tmp_path / ".agents-md-report.md").write_text("NEEDS WORK: add commands")

        with (
            patch("dr_agents_tester.agents_md.gather_context") as mock_ctx,
            patch("dr_agents_tester.agents_md.call_llm") as mock_llm,
        ):
            mock_ctx.return_value = {
                "target_dir": ".",
                "is_root": True,
                "tree": "",
                "file_contents": {},
                "copier_answers": {},
                "sibling_agents_md": {},
            }
            mock_llm.return_value = (
                "===AGENTS_MD===\n# Improved content\n===REVISION_NOTES===\nFixed commands."
            )

            mgr = AgentsMd(fake_config)
            mgr.revise(tmp_path, tmp_path, no_copilot=True)

        content = (tmp_path / "AGENTS.md").read_text()
        assert "# Improved content" in content
        assert (tmp_path / ".agents-md-revision-notes.md").exists()
