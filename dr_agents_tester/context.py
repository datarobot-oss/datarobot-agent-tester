"""Project context gathering — file trees, priority file contents, copier answers."""

import subprocess
from pathlib import Path
from typing import TypedDict


class Context(TypedDict):
    target_dir: str
    is_root: bool
    tree: str
    file_contents: dict[str, str]
    copier_answers: dict[str, str]
    sibling_agents_md: dict[str, str]

PRIORITY_FILES = [
    "README.md",
    "AGENTS.md",
    "pyproject.toml",
    "package.json",
    "Taskfile.yml",
    "Taskfile.yaml",
    "copier.yml",
    "copier.yaml",
]

EXCLUDE_DIRS = {
    ".git",
    "__pycache__",
    "node_modules",
    ".venv",
    "dist",
    "build",
    ".mypy_cache",
    ".ruff_cache",
    "htmlcov",
    ".data",
    "tmp",
}

MAX_TREE_FILES = 80
MAX_ANSWERS_FILES = 5


def read_safe(path: Path) -> str:
    """Read a file, returning empty string on any error."""
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""


def git_tracked_files(repo_root: Path) -> list[str]:
    """Return git-tracked (and untracked-but-not-ignored) file paths relative to repo_root."""
    try:
        result = subprocess.run(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
            cwd=repo_root,
            capture_output=True,
            text=True,
        )
        return [f for f in result.stdout.splitlines() if f]
    except Exception:
        return []


def build_tree(files: list[str], target_subdir: str | None) -> str:
    """Build a compact directory-tree string from a list of relative paths."""
    if target_subdir:
        prefix = target_subdir.rstrip("/") + "/"
        files = [f[len(prefix) :] for f in files if f.startswith(prefix)]

    dirs: dict[str, list[str]] = {}
    for f in files:
        parts = f.split("/")
        top = parts[0] if len(parts) > 1 else ""
        dirs.setdefault(top, []).append(f)

    lines = []
    for d, members in sorted(dirs.items()):
        if d and d in EXCLUDE_DIRS:
            continue
        if d:
            lines.append(f"  {d}/")
            for m in sorted(members)[:8]:
                lines.append(f"    {m.split('/', 1)[-1]}")
            if len(members) > 8:
                lines.append(f"    ... ({len(members) - 8} more)")
        else:
            for m in sorted(members):
                lines.append(f"  {m}")

    return "\n".join(lines[:MAX_TREE_FILES])


def gather_context(target_dir: Path, repo_root: Path) -> Context:
    """Return structured context about a directory suitable for prompt construction.

    Args:
        target_dir: The directory to document (may equal repo_root).
        repo_root:  Root of the git repository.

    Returns:
        Dict with keys: target_dir, is_root, tree, file_contents,
        copier_answers, sibling_agents_md.
    """
    rel = str(target_dir.relative_to(repo_root)) if target_dir != repo_root else "."
    all_files = git_tracked_files(repo_root)
    tree = build_tree(all_files, None if rel == "." else rel)

    file_contents: dict[str, str] = {}
    for name in PRIORITY_FILES:
        candidate = target_dir / name
        if candidate.exists():
            content = read_safe(candidate)
            if content:
                file_contents[name] = content

    answers_dir = repo_root / ".datarobot" / "answers"
    answers: dict[str, str] = {}
    if answers_dir.exists():
        for yml in sorted(answers_dir.glob("*.yml"))[:MAX_ANSWERS_FILES]:
            content = read_safe(yml)
            if content:
                answers[yml.name] = content

    sibling_agents_md: dict[str, str] = {}
    if rel != ".":
        for sibling in sorted(repo_root.iterdir()):
            if sibling.is_dir() and sibling != target_dir and sibling.name not in EXCLUDE_DIRS:
                agents = sibling / "AGENTS.md"
                if agents.exists():
                    sibling_agents_md[sibling.name] = read_safe(agents)

    return {
        "target_dir": rel,
        "is_root": rel == ".",
        "tree": tree,
        "file_contents": file_contents,
        "copier_answers": answers,
        "sibling_agents_md": sibling_agents_md,
    }


def find_repo_root(start: Path) -> Path:
    """Walk up from start until a .git directory is found, or return start."""
    current = start.resolve()
    while not (current / ".git").exists() and current != current.parent:
        current = current.parent
    return current.resolve()
