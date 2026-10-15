"""Install skills into a run's isolated OpenCode config, the way users get them.

The published npm plugin's entire install step is a recursive copy of each
``skills/<name>/`` directory into ``~/.config/opencode/skills/`` — so copying
into the run's isolated HOME produces a byte-identical end state to a real
user install while staying deterministic and offline.

All skills from the source are installed (users install the whole plugin), so
behavioral runs also measure trigger *selectivity*: the agent must pick the
right skill among all of them.
"""

from __future__ import annotations

import hashlib
import shutil
from dataclasses import dataclass
from pathlib import Path

from .drivers.base import RunPaths


@dataclass
class InstalledSkill:
    """One installed skill plus a content digest proving which bytes ran."""

    name: str
    digest: str


def install_skills(skills_source: Path, paths: RunPaths) -> list[InstalledSkill]:
    """Copy every skill directory from ``skills_source`` into the run's config.

    Args:
        skills_source: Directory whose children are skill directories (each
            containing a SKILL.md, possibly nested sub-skills) — typically a
            checkout's ``skills/`` directory.
        paths: The run layout; skills land in
            ``<home>/.config/opencode/skills/<name>/``.

    Returns:
        The installed skills with content digests (recorded in run metadata).
    """
    if not skills_source.is_dir():
        raise FileNotFoundError(f"Skills source directory not found: {skills_source}")

    target_root = paths.home / ".config" / "opencode" / "skills"
    target_root.mkdir(parents=True, exist_ok=True)

    installed: list[InstalledSkill] = []
    for skill_dir in sorted(p for p in skills_source.iterdir() if p.is_dir()):
        if not any(skill_dir.rglob("SKILL.md")):
            continue
        target = target_root / skill_dir.name
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(skill_dir, target)
        installed.append(InstalledSkill(name=skill_dir.name, digest=_dir_digest(target)))

    if not installed:
        raise ValueError(f"No skill directories (containing SKILL.md) under {skills_source}")
    return installed


def _dir_digest(root: Path) -> str:
    """Stable content digest over every file in a skill directory."""
    h = hashlib.sha256()
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        h.update(str(path.relative_to(root)).encode())
        h.update(path.read_bytes())
    return h.hexdigest()[:16]
