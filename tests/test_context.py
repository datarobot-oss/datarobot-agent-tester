"""Tests for context gathering (no LLM calls required)."""

import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from dr_agents_tester.context import (
    build_tree,
    find_repo_root,
    gather_context,
    git_tracked_files,
    read_safe,
)


class TestReadSafe:
    def test_reads_existing_file(self, tmp_path: Path) -> None:
        f = tmp_path / "hello.txt"
        f.write_text("hello world")
        assert read_safe(f) == "hello world"

    def test_returns_empty_on_missing_file(self, tmp_path: Path) -> None:
        assert read_safe(tmp_path / "nonexistent.txt") == ""

    def test_returns_empty_on_permission_error(self, tmp_path: Path) -> None:
        f = tmp_path / "locked.txt"
        f.write_text("secret")
        f.chmod(0o000)
        try:
            assert read_safe(f) == ""
        finally:
            f.chmod(0o644)


class TestBuildTree:
    def test_basic_flat_files(self) -> None:
        files = ["README.md", "pyproject.toml", "main.py"]
        tree = build_tree(files, None)
        assert "README.md" in tree
        assert "pyproject.toml" in tree

    def test_filters_by_subdir(self) -> None:
        files = ["core/main.py", "core/utils.py", "infra/deploy.py"]
        tree = build_tree(files, "core")
        assert "main.py" in tree
        assert "utils.py" in tree
        assert "deploy.py" not in tree

    def test_excludes_excluded_dirs(self) -> None:
        files = ["src/main.py", "__pycache__/cached.pyc", ".venv/lib.py"]
        tree = build_tree(files, None)
        assert "__pycache__" not in tree
        assert ".venv" not in tree

    def test_empty_file_list(self) -> None:
        assert build_tree([], None) == ""

    def test_truncates_deep_dirs(self) -> None:
        # More than 8 files in one directory should show truncation
        files = [f"pkg/file{i}.py" for i in range(15)]
        tree = build_tree(files, None)
        assert "more" in tree


class TestGitTrackedFiles:
    def test_returns_list_on_success(self, tmp_path: Path) -> None:
        mock_output = "README.md\nsrc/main.py\ntests/test_main.py\n"
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess(
                args=[], returncode=0, stdout=mock_output, stderr=""
            )
            result = git_tracked_files(tmp_path)
        assert result == ["README.md", "src/main.py", "tests/test_main.py"]

    def test_returns_empty_on_error(self, tmp_path: Path) -> None:
        with patch("subprocess.run", side_effect=FileNotFoundError):
            result = git_tracked_files(tmp_path)
        assert result == []

    def test_filters_empty_lines(self, tmp_path: Path) -> None:
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess(
                args=[], returncode=0, stdout="a.py\n\nb.py\n", stderr=""
            )
            result = git_tracked_files(tmp_path)
        assert "" not in result
        assert len(result) == 2


class TestFindRepoRoot:
    def test_finds_git_dir(self, tmp_path: Path) -> None:
        (tmp_path / ".git").mkdir()
        subdir = tmp_path / "src" / "pkg"
        subdir.mkdir(parents=True)
        assert find_repo_root(subdir) == tmp_path.resolve()

    def test_returns_start_when_no_git(self, tmp_path: Path) -> None:
        result = find_repo_root(tmp_path)
        # Should stop at filesystem root, not raise
        assert isinstance(result, Path)


class TestGatherContext:
    def test_root_context(self, tmp_path: Path) -> None:
        (tmp_path / ".git").mkdir()
        (tmp_path / "README.md").write_text("# My Project")
        (tmp_path / "pyproject.toml").write_text('[project]\nname = "test"')

        with patch("dr_agents_tester.context.git_tracked_files") as mock_git:
            mock_git.return_value = ["README.md", "pyproject.toml"]
            ctx = gather_context(tmp_path, tmp_path)

        assert ctx["is_root"] is True
        assert ctx["target_dir"] == "."
        assert "README.md" in ctx["file_contents"]  # type: ignore[operator]
        assert "pyproject.toml" in ctx["file_contents"]  # type: ignore[operator]

    def test_subdir_context(self, tmp_path: Path) -> None:
        (tmp_path / ".git").mkdir()
        subdir = tmp_path / "myservice"
        subdir.mkdir()
        (subdir / "README.md").write_text("# My Service")

        with patch("dr_agents_tester.context.git_tracked_files") as mock_git:
            mock_git.return_value = ["myservice/README.md"]
            ctx = gather_context(subdir, tmp_path)

        assert ctx["is_root"] is False
        assert ctx["target_dir"] == "myservice"

    def test_sibling_agents_md_collected(self, tmp_path: Path) -> None:
        (tmp_path / ".git").mkdir()
        subdir = tmp_path / "service_a"
        sibling = tmp_path / "service_b"
        subdir.mkdir()
        sibling.mkdir()
        (sibling / "AGENTS.md").write_text("# Service B")

        with patch("dr_agents_tester.context.git_tracked_files") as mock_git:
            mock_git.return_value = []
            ctx = gather_context(subdir, tmp_path)

        assert "service_b" in ctx["sibling_agents_md"]  # type: ignore[operator]
