"""Reusable pytest infrastructure for hash-cached skill end-to-end tests.

This module provides the building blocks for running LLM-based quality checks
against SKILL.md files in any repo.  It uses a file-backed MD5 hash store to
skip unchanged skills so CI only pays for what has actually changed.

Typical usage in a consuming repo's ``tests/e2e/test_skills_e2e.py``::

    from pathlib import Path
    from dr_agents_tester.pytest_plugin import make_skill_e2e_test
    from dotenv import load_dotenv

    load_dotenv(override=False)

    pytest_generate_tests, test_skill_quality = make_skill_e2e_test(
        skills_dir=Path(__file__).resolve().parents[2] / "skills",
        hash_file=Path(__file__).resolve().parent / "skill_hashes.json",
    )

The full hash-cache logic, LLM call, verdict parsing, parametrize hook, and
skip conditions all live here — the consuming file stays at ~10 lines.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Callable

try:
    import pytest
except ImportError as _pytest_import_error:  # pragma: no cover
    raise ImportError(
        "pytest is required to use dr_agents_tester.pytest_plugin. "
        "Install it with: pip install pytest"
    ) from _pytest_import_error

from .config import Config
from .skills import Skills

__all__ = [
    "HashCache",
    "parse_verdict",
    "verdict_passes",
    "register_skills_e2e",
    "make_skill_e2e_test",
]

_DEFAULT_TEST_MODEL = "datarobot/anthropic/claude-sonnet-4-6"
_TRANSIENT_ERROR_KEYWORDS = ("not found in catalog", "apiconnection", "connection", "timeout")


# ---------------------------------------------------------------------------
# HashCache
# ---------------------------------------------------------------------------


class HashCache:
    """File-backed MD5 hash store for SKILL.md files.

    The cache is loaded from a JSON file on construction and written back on
    :meth:`save`.  Callers are responsible for calling :meth:`save` when they
    are done — the intended pattern is to use the cache inside a pytest
    session-scoped fixture that yields and then saves on teardown.

    Example::

        cache = HashCache(Path("tests/e2e/skill_hashes.json"))
        if cache.is_changed(skill_path, key="skills/commit/SKILL.md"):
            # run the test …
            cache.update("skills/commit/SKILL.md", skill_path)
        cache.save()
    """

    def __init__(self, path: Path) -> None:
        self._path = path
        self._hashes: dict[str, str] = {}
        if path.exists():
            try:
                self._hashes = json.loads(path.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001 — corrupt file → start fresh
                self._hashes = {}

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def is_changed(self, skill_path: Path, key: str) -> bool:
        """Return True if the MD5 of *skill_path* differs from the stored hash.

        A missing key is treated as changed (always returns True).
        """
        return self._hashes.get(key) != _md5(skill_path)

    def update(self, key: str, skill_path: Path) -> None:
        """Store the current MD5 of *skill_path* under *key*."""
        self._hashes[key] = _md5(skill_path)

    def save(self) -> None:
        """Write the in-memory hash table back to the JSON file."""
        self._path.write_text(
            json.dumps(self._hashes, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


# ---------------------------------------------------------------------------
# Verdict helpers
# ---------------------------------------------------------------------------


def parse_verdict(report: str) -> str:
    """Extract the Overall Verdict token from a skill test report.

    Returns one of ``"GOOD"``, ``"NEEDS WORK"``, ``"INCOMPLETE"``, or
    ``"UNKNOWN"`` when the verdict line cannot be found.

    The LLM formats the verdict line as e.g. ``**NEEDS WORK** — explanation``.
    We only check the label portion (before any em-dash) to avoid false
    matches where the explanation text itself contains the word "incomplete".
    """
    match = re.search(
        r"###\s+Overall\s+verdict.*?\n(.+?)(?:\n|$)",
        report,
        re.IGNORECASE | re.DOTALL,
    )
    full_line = match.group(1).strip().upper() if match else report.upper()

    # Only inspect the label part — everything before the first em-dash or
    # regular dash separator so "NEEDS WORK — ...incomplete results..." doesn't
    # trigger a false INCOMPLETE match.
    label = re.split(r"\s*[—–-]\s*", full_line, maxsplit=1)[0].strip()

    for token in ("INCOMPLETE", "NEEDS WORK", "GOOD"):
        if token in label:
            return token

    # Fallback: search the whole line (handles unusual LLM formatting)
    for token in ("INCOMPLETE", "NEEDS WORK", "GOOD"):
        if token in full_line:
            return token

    return "UNKNOWN"


def verdict_passes(report: str) -> bool:
    """Return True when the skill verdict is not INCOMPLETE.

    A verdict of GOOD, NEEDS WORK, or UNKNOWN is considered passing because the
    skill is at least coherent enough for an agent to follow.  Only INCOMPLETE
    means the skill would produce inconsistent or poor results.
    """
    return parse_verdict(report) != "INCOMPLETE"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def _skill_files(skills_dir: Path) -> list[Path]:
    return sorted(skills_dir.glob("*/SKILL.md"))


# ---------------------------------------------------------------------------
# Core factory
# ---------------------------------------------------------------------------


def register_skills_e2e(
    skills_dir: Path,
    hash_file: Path,
    force_all_env: str = "SKILLS_E2E_FORCE_ALL",
    default_test_model: str = _DEFAULT_TEST_MODEL,
) -> tuple[Callable[..., None], Callable[..., None], Callable[..., None]]:
    """Build a ``(pytest_generate_tests, test_skill_quality, _hash_store)`` triple.

    Parameters
    ----------
    skills_dir:
        Root directory that contains per-skill sub-directories, each holding a
        ``SKILL.md`` file.  Example: ``repo_root / "skills"``.
    hash_file:
        Path to the JSON file used to persist MD5 hashes between runs.  The
        file is created automatically on the first run.
    force_all_env:
        Name of the environment variable that, when set to ``"1"``, ``"true"``,
        or ``"yes"``, disables the hash-skip logic and tests every skill.
        Defaults to ``"SKILLS_E2E_FORCE_ALL"``.
    default_test_model:
        LiteLLM model string used when ``AGENTS_MD_TEST_MODEL`` is not set in
        the environment.

    Returns
    -------
    tuple[Callable, Callable, Callable]
        A ``(pytest_generate_tests, test_skill_quality, _hash_store)`` triple
        ready to be assigned directly into a test module's namespace.  All
        three names must be assigned so that pytest can discover the
        session-scoped ``_hash_store`` fixture.
    """
    force_all = os.environ.get(force_all_env, "").lower() in ("1", "true", "yes")

    # -- session-scoped fixture: load hashes once, save after all tests -------

    @pytest.fixture(scope="session")
    def _hash_store() -> Any:
        cache = HashCache(hash_file)
        yield cache
        cache.save()

    # -- parametrize hook: one test per SKILL.md found ------------------------

    def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
        if "skill_path" in metafunc.fixturenames:
            skills = _skill_files(skills_dir)
            metafunc.parametrize(
                "skill_path",
                skills,
                ids=[p.parent.name for p in skills],
            )

    # -- the actual test function ---------------------------------------------

    def test_skill_quality(skill_path: Path, _hash_store: HashCache) -> None:
        skill_key = str(skill_path.relative_to(skills_dir.parent))
        if not force_all and not _hash_store.is_changed(skill_path, skill_key):
            pytest.skip(f"{skill_path.parent.name}: unchanged (hash match)")

        cfg = Config()
        try:
            cfg.validate()
        except ValueError as exc:
            pytest.skip(f"No DR credentials available — {exc}")

        test_model = os.environ.get("AGENTS_MD_TEST_MODEL", default_test_model)
        skills = Skills(cfg)
        try:
            report = skills.test(skill_path, test_model=test_model)
        except Exception as exc:
            err = str(exc).lower()
            if any(kw in err for kw in _TRANSIENT_ERROR_KEYWORDS):
                pytest.skip(f"LLM unavailable in this environment — {exc}")
            raise

        # Update hash only after a passing verdict so a failing skill is
        # re-evaluated on the next run rather than silently skipped.
        assert verdict_passes(report), (
            f"Skill '{skill_path.parent.name}' is INCOMPLETE.\n\n{report}"
        )
        _hash_store.update(skill_key, skill_path)

    # Return all three so the consuming module can assign them at module level,
    # which is required for pytest to discover the session-scoped _hash_store
    # fixture (pytest only finds fixtures in module scope / conftest.py).
    return pytest_generate_tests, test_skill_quality, _hash_store


def make_skill_e2e_test(
    skills_dir: Path,
    hash_file: Path,
    force_all_env: str = "SKILLS_E2E_FORCE_ALL",
    default_test_model: str = _DEFAULT_TEST_MODEL,
) -> tuple[Callable[..., None], Callable[..., None], Callable[..., None]]:
    """Convenience wrapper around :func:`register_skills_e2e`.

    Returns a fully configured ``(pytest_generate_tests, test_skill_quality,
    _hash_store)`` triple.  Assign **all three** names into your test module's
    global namespace so pytest can discover the session-scoped fixture::

        pytest_generate_tests, test_skill_quality, _hash_store = make_skill_e2e_test(
            skills_dir=Path(__file__).resolve().parents[2] / "skills",
            hash_file=Path(__file__).resolve().parent / "skill_hashes.json",
        )

    pytest will discover the returned callables by name.
    """
    return register_skills_e2e(
        skills_dir=skills_dir,
        hash_file=hash_file,
        force_all_env=force_all_env,
        default_test_model=default_test_model,
    )
