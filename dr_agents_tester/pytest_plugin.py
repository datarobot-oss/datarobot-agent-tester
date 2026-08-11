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
from .skills import VERDICT_PREFIX, Skills

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


_TOKEN_ALT = r"(INCOMPLETE|NEEDS\s+WORK|GOOD)"

# Primary: the mandatory sentinel line the test prompt demands as the report's
# final line, e.g. "VERDICT: NEEDS WORK".  A line only counts as a verdict when
# it carries exactly ONE token: the trailing text after a separator may be an
# explanation ("VERDICT: GOOD — minor tweaks") but never another token, so an
# echoed option list ("VERDICT: GOOD, NEEDS WORK, or INCOMPLETE") and an echoed
# template ("VERDICT: <GOOD or ...>") can never parse as a verdict.
_SENTINEL_PREFIX_RE = re.compile(rf"^[\s>*_`]*{re.escape(VERDICT_PREFIX)}", re.IGNORECASE)
_SENTINEL_LINE_RE = re.compile(
    rf"^[\s>*_`]*{re.escape(VERDICT_PREFIX)}\s*[*_`]*\s*{_TOKEN_ALT}"
    rf"[*_`]*\s*(?:[—–:.!;,-]\s*(?P<rest>.*))?$",
    re.IGNORECASE,
)
_TOKEN_ANYWHERE_RE = re.compile(_TOKEN_ALT, re.IGNORECASE)

# The sentinel is only honored near the end of the report.  A verdict-shaped
# line quoted mid-report (skills under test define report formats of their
# own) must not stand in for a missing or malformed final verdict line.
_SENTINEL_TAIL_LINES = 3

# Compat: pre-sentinel reports end in an "Overall verdict" heading.  Judges
# freely reformat — any heading level, any case, verdict inline or below.
_HEADING_RE = re.compile(
    r"^#{1,6}\s*Overall\s+Verdict\b[ \t]*:?[ \t]*(.*)$",
    re.IGNORECASE | re.MULTILINE,
)

# The verdict token must open its line (markdown wrapping tolerated).  Tokens
# buried mid-sentence are prose, not verdicts.
_LINE_TOKEN_RE = re.compile(rf"^[\s>*_`\-:]*{_TOKEN_ALT}\b", re.IGNORECASE)

# How many lines below the heading may hold the verdict (blank lines between
# the heading and the verdict are common).
_VERDICT_LOOKAHEAD_LINES = 5


def _normalize_token(token: str) -> str:
    return re.sub(r"\s+", " ", token.upper())


def parse_verdict(report: str) -> str:
    """Extract the verdict token from a skill test report.

    Returns one of ``"GOOD"``, ``"NEEDS WORK"``, ``"INCOMPLETE"``, or
    ``"UNKNOWN"`` when no verdict can be located.

    Resolution order:

    1. The ``VERDICT: <token>`` sentinel line, honored only within the last
       few non-empty lines of the report.  The last sentinel-shaped line is
       authoritative: it must carry exactly one verdict token, and a malformed
       one (garbage token, or a second token in the trailing text) returns
       UNKNOWN outright rather than falling back.  A judge that is off the
       sentinel contract gets a retry, not a guess.
    2. The ``Overall verdict`` heading, at any level, last occurrence: the
       token is read from the heading's own line or the first non-empty line
       below it, anchored at line start.  Kept for reports produced by the
       pre-sentinel prompt.  Last occurrence, because judges quote skill text,
       and a quoted heading earlier in the report must not shadow the real one.

    Body prose is never scanned.  The prompt itself lists INCOMPLETE as an
    option, so the word is effectively guaranteed to appear somewhere in the
    report text — scanning for it is how this parser produced false failures
    on reports whose actual verdict was NEEDS WORK.
    """
    tail_lines = [line for line in report.splitlines() if line.strip()]
    for line in reversed(tail_lines[-_SENTINEL_TAIL_LINES:]):
        if not _SENTINEL_PREFIX_RE.match(line):
            continue
        m = _SENTINEL_LINE_RE.match(line)
        if m and not _TOKEN_ANYWHERE_RE.search(m.group("rest") or ""):
            return _normalize_token(m.group(1))
        return "UNKNOWN"

    headings = list(_HEADING_RE.finditer(report))
    if headings:
        heading = headings[-1]
        candidates = [heading.group(1)]
        candidates += report[heading.end() :].splitlines()[1 : _VERDICT_LOOKAHEAD_LINES + 1]
        for line in candidates:
            if not line.strip():
                continue
            token = _LINE_TOKEN_RE.match(line)
            if token:
                return _normalize_token(token.group(1))
            break  # the first non-empty line is the verdict line; don't scan prose

    return "UNKNOWN"


def verdict_passes(report: str) -> bool:
    """Return True when the verdict affirmatively passes the gate.

    GOOD and NEEDS WORK pass; the skill is coherent enough for an agent to
    follow.  INCOMPLETE fails.  UNKNOWN — no parseable verdict — also fails:
    a gate that cannot read the verdict must not silently pass, otherwise a
    judge formatting drift disables the gate without anyone noticing.
    """
    return parse_verdict(report) in ("GOOD", "NEEDS WORK")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def _skill_files(skills_dir: Path) -> list[Path]:
    return sorted(skills_dir.glob("*/SKILL.md"))


def _judge_with_retry(skills: Skills, skill_path: Path, test_model: str) -> str:
    """Run the judge, re-sampling once if the report has no parseable verdict.

    Judges occasionally drop the verdict line; one fresh sample usually
    recovers.  The caller decides what a still-unparseable report means.
    """
    report = skills.test(skill_path, test_model=test_model)
    if parse_verdict(report) != "UNKNOWN":
        return report
    return skills.test(skill_path, test_model=test_model)


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
            report = _judge_with_retry(skills, skill_path, test_model)
        except Exception as exc:
            err = str(exc).lower()
            if any(kw in err for kw in _TRANSIENT_ERROR_KEYWORDS):
                pytest.skip(f"LLM unavailable in this environment — {exc}")
            raise

        verdict = parse_verdict(report)
        if verdict == "UNKNOWN":
            # Fail closed, loudly and distinctly: this is a judge output
            # contract breach, not a judgment on the skill's quality.
            pytest.fail(
                f"Judge did not emit a parseable verdict for skill "
                f"'{skill_path.parent.name}' (after one retry). The gate "
                f"cannot pass a report it cannot read.\n\n{report}"
            )

        # Update hash only after a passing verdict so a failing skill is
        # re-evaluated on the next run rather than silently skipped.
        assert verdict != "INCOMPLETE", (
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
