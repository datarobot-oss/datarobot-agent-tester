# Copyright (c) 2026 DataRobot, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resolve DataRobot SDK calls in agent-delivered scripts against the real SDK.

Behavioral runs pass on their ``success_checks``, which assert delivered
*state* — a project exists, a CSV has rows. Nothing asserts that the script the
agent leaves behind can run again. Both recorded rounds show the same residue:
the agent hits a bad SDK call, fixes it in a throwaway inline command, and
leaves the script in the workspace still broken. In round 2 the golden run
shipped ``run_datarobot.py`` with ``score_to_file(intake_file_path=…)`` against
the real ``intake_path``/``output_path`` — the user's artifact cannot run end to
end, and every check passed.

This module binds each ``dr.*`` call in the workspace's Python files against the
installed ``datarobot`` package, so wrong names and wrong arguments surface
without executing anything: no duplicate DataRobot resources, no second
AutoML run, no account required beyond the SDK already needed for checks.

Reported as the non-gating ``num_delivered_script_defects`` trajectory metric,
deliberately *not* a success check — making it gating would retroactively fail
runs that the recorded baseline and round 2 both counted as passes, and the
comparison across rounds is worth more than an immediate hard gate.

**Scope, so a zero is not over-read.** This resolves names and binds arguments.
It catches calls to things that do not exist and calls the SDK would reject on
arity or keyword names. It cannot catch a call that is *signature*-valid and
still fails: the baseline's own residue is the case in point —
``create_from_learning_model`` without ``default_prediction_server_id`` binds
cleanly because the SDK types that parameter ``Optional[str]``, and only
DataRobot Cloud's 422 reveals it is mandatory. Both baseline golden workspaces
therefore scan clean while shipping a script that cannot run. Read the metric as
"no SDK-signature defects", never as "the delivered script runs".

The call-resolution core is shared with the skills repo's
``tests/integration/sdk_conformance.py`` (PR #86), which applies the same
analysis to skill *sources*. That copy should import from here once the
skills-repo branches settle, rather than the two drifting apart.
"""

from __future__ import annotations

import ast
import importlib.metadata
import inspect
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

import datarobot as dr  # noqa: F401  (resolution target)

#: Reported in findings so a "does not exist" verdict names what it was checked
#: against. Read from package metadata rather than ``dr.__version__``, which the
#: SDK does not explicitly re-export.
try:
    SDK_VERSION = importlib.metadata.version("datarobot")
except importlib.metadata.PackageNotFoundError:  # pragma: no cover - installed by the extra
    SDK_VERSION = "unknown"

SDK_ALIASES = {"dr", "datarobot"}

# Nodes that introduce a new variable scope.  Type inference is per-scope so a
# name reused for different SDK types in sibling functions is not conflated.
SCOPE_NODES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)

# Methods assumed to return an instance of their own class, subject to the
# return-annotation check in _infer_var_types.  ``create`` is deliberately
# included but guarded: e.g. PredictionExplanations.create() returns a Job.
FACTORY_PREFIXES = ("create_from", "from_")
FACTORY_NAMES = ("get", "create")

# NOTE: deprecation detection is intentionally absent.  The SDK signals
# deprecations with a runtime decorator rather than docstring text, so a static
# check gives false confidence.  Importing under ``-W error::DeprecationWarning``
# is the right tool for that.


@dataclass(frozen=True)
class Finding:
    """A single SDK conformance violation."""

    file: str
    line: int
    kind: str
    message: str

    def __str__(self) -> str:
        return f"{self.file}:{self.line} [{self.kind}] {self.message}"


def _dotted_name(node: ast.AST) -> str | None:
    """Flatten an attribute chain (``dr.Project.start``) into a dotted string."""
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return ".".join(reversed(parts))
    return None


def _resolve(attr_path: list[str]) -> tuple[Any, bool]:
    """Walk an attribute path from the datarobot module."""
    obj: Any = dr
    for part in attr_path:
        if not hasattr(obj, part):
            return None, False
        obj = getattr(obj, part)
    return obj, True


def _scope_nodes(scope: ast.AST) -> Iterator[ast.AST]:
    """Yield nodes belonging to ``scope``, without entering nested scopes."""
    for child in ast.iter_child_nodes(scope):
        if isinstance(child, SCOPE_NODES):
            continue
        yield child
        yield from _scope_nodes(child)


def _nested_scopes(scope: ast.AST) -> Iterator[ast.AST]:
    """Yield the scopes directly nested inside ``scope`` (not their children)."""
    for child in ast.iter_child_nodes(scope):
        if isinstance(child, SCOPE_NODES):
            yield child
        else:
            yield from _nested_scopes(child)


def _infer_var_types(nodes: Iterable[ast.AST], aliases: set[str]) -> dict[str, type]:
    """Map ``x = dr.Foo.get(...)`` to ``x: Foo`` so instance calls can be checked."""
    var_types: dict[str, type] = {}
    for node in nodes:
        if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Call):
            continue
        name = _dotted_name(node.value.func)
        if not name:
            continue
        parts = name.split(".")
        if parts[0] not in aliases or len(parts) < 3:
            continue
        method = parts[-1]
        if not (method in FACTORY_NAMES or method.startswith(FACTORY_PREFIXES)):
            continue
        cls, ok = _resolve(parts[1:-1])
        if not (ok and inspect.isclass(cls)):
            continue
        # Only infer when the factory really returns its own class.
        try:
            annotation = str(inspect.signature(getattr(cls, method)).return_annotation)
        except (ValueError, TypeError):
            annotation = ""
        if annotation and "inspect._empty" not in annotation and cls.__name__ not in annotation:
            continue
        for target in node.targets:
            if isinstance(target, ast.Name):
                var_types[target.id] = cls
    return var_types


def _check_signature(
    call: ast.Call, target: Any, display: str, drop_self: bool
) -> list[tuple[str, str]]:
    """Return (kind, message) pairs for signature violations."""
    out: list[tuple[str, str]] = []
    if not callable(target):
        return out
    try:
        signature = inspect.signature(target)
    except (ValueError, TypeError):
        return out

    params = list(signature.parameters.values())
    if drop_self and params and params[0].name in ("self", "cls"):
        params = params[1:]

    names = {p.name for p in params}
    accepts_kwargs = any(p.kind is p.VAR_KEYWORD for p in params)
    accepts_varargs = any(p.kind is p.VAR_POSITIONAL for p in params)
    passed_kw = {kw.arg for kw in call.keywords if kw.arg}

    unknown = passed_kw - names
    if unknown and not accepts_kwargs:
        out.append(
            (
                "bad-kwarg",
                f"`{display}()` got unexpected keyword(s) {sorted(unknown)}; "
                f"valid: {sorted(names)[:9]}",
            )
        )

    # A named keyword is checkable even alongside a splat, but arity is not:
    # ``f(**opts)`` may well supply every required parameter.
    splatted = any(isinstance(arg, ast.Starred) for arg in call.args) or any(
        kw.arg is None for kw in call.keywords
    )
    if accepts_varargs or splatted:
        return out

    # Positional parameters are satisfied either by position or by name, so the
    # index into `required` is compared against how many positionals were passed.
    required = [
        p
        for p in params
        if p.default is inspect.Parameter.empty
        and p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
    ]
    n_positional = len(call.args)
    missing = [
        p.name for i, p in enumerate(required) if p.name not in passed_kw and i >= n_positional
    ]
    # Keyword-only parameters can only ever be supplied by name, so position is
    # irrelevant: absent from the call keywords means absent. e.g. omitting
    # connector_type from Connector.create(*, connector_type) is a TypeError.
    missing += [
        p.name
        for p in params
        if p.kind is p.KEYWORD_ONLY
        and p.default is inspect.Parameter.empty
        and p.name not in passed_kw
    ]
    if missing and missing[0] in ("self", "cls"):
        # Reached through the class rather than an instance, so the receiver is
        # unbound.  Reporting a literal "self" as missing only confuses.
        out.append(
            (
                "instance-method-on-class",
                f"`{display}()` is an instance method reached through the class, "
                "so there is no receiver; call it on an instance",
            )
        )
    elif missing:
        out.append(("missing-arg", f"`{display}()` missing required arg(s) {missing}"))
    return out


def _check_call(
    call: ast.Call,
    aliases: set[str],
    var_types: dict[str, type],
    rel_path: str,
    line_offset: int,
) -> list[Finding]:
    name = _dotted_name(call.func)
    if not name:
        return []
    parts = name.split(".")
    line = call.lineno + line_offset
    found: list[Finding] = []

    def note(kind: str, message: str) -> None:
        found.append(Finding(rel_path, line, kind, message))

    classmethod_via_instance = False

    if parts[0] in aliases:
        # dr.Foo.bar(...) — resolvable directly against the module.
        target, ok = _resolve(parts[1:])
        if not ok:
            note(
                "missing-symbol",
                f"`{name}` does not exist in datarobot {SDK_VERSION}",
            )
            return found
        drop_self = False
    elif len(parts) == 2 and parts[0] in var_types:
        # instance.bar(...) — the descriptor kind decides how args bind.
        cls = var_types[parts[0]]
        if not hasattr(cls, parts[1]):
            note("missing-method", f"`{cls.__name__}.{parts[1]}` does not exist")
            return found
        static = inspect.getattr_static(cls, parts[1], None)
        target = getattr(cls, parts[1])
        # getattr() pre-binds cls for classmethods but leaves `self` on plain
        # methods, so only the latter needs its first parameter dropped.
        drop_self = not isinstance(static, (classmethod, staticmethod))
        classmethod_via_instance = isinstance(static, classmethod)
    else:
        return found

    problems = _check_signature(call, target, name, drop_self)

    # Reaching a classmethod through an instance is legal and works when the
    # arguments are supplied, so it is only worth reporting as the explanation
    # for an arity failure.
    if classmethod_via_instance and any(kind == "missing-arg" for kind, _ in problems):
        note(
            "classmethod-on-instance",
            f"`{name}()` calls a CLASSMETHOD on an instance "
            f"({var_types[parts[0]].__name__}.{parts[1]}) — required args are "
            "not auto-filled",
        )

    for kind, message in problems:
        note(kind, message)
    return found


def _analyse_source(source: str, rel_path: str, line_offset: int = 0) -> tuple[list[Finding], bool]:
    """Analyse one code chunk. Returns (findings, parsed_ok)."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return [], False

    aliases = set(SDK_ALIASES)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "datarobot" and alias.asname:
                    aliases.add(alias.asname)

    def walk_scope(scope: ast.AST, inherited: dict[str, type]) -> list[Finding]:
        nodes = list(_scope_nodes(scope))
        # Names bound here shadow anything inherited from an enclosing scope.
        var_types = {**inherited, **_infer_var_types(nodes, aliases)}
        findings = [
            finding
            for node in nodes
            if isinstance(node, ast.Call)
            for finding in _check_call(node, aliases, var_types, rel_path, line_offset)
        ]
        for nested in _nested_scopes(scope):
            findings.extend(walk_scope(nested, var_types))
        return findings

    return walk_scope(tree, {}), True


# --------------------------------------------------------------------------
# Delivered-script scanning (the tester-side entry point)
# --------------------------------------------------------------------------

#: Directories never worth scanning inside a run workspace.
_SKIP_DIRS = frozenset({"__pycache__", ".venv", "venv", "node_modules", ".git", ".pytest_cache"})

#: A single agent script larger than this is skipped rather than parsed — a
#: runaway artifact must not stall the post-run metric.
MAX_SCRIPT_BYTES = 2 * 1024 * 1024


@dataclass(frozen=True)
class ScriptScan:
    """Result of scanning a workspace's agent-authored Python files."""

    findings: list[Finding]
    scanned: list[str]
    skipped: list[str]

    @property
    def num_defects(self) -> int:
        return len(self.findings)

    def summaries(self) -> list[str]:
        """One human-readable line per finding, for trajectory.json."""
        return [str(f) for f in self.findings]


def scan_delivered_scripts(
    workspace: Path, exclude: Iterable[str] = (), max_bytes: int = MAX_SCRIPT_BYTES
) -> ScriptScan:
    """Bind every ``dr.*`` call in the workspace's Python files against the SDK.

    ``exclude`` holds workspace-relative paths that were copied in as fixtures:
    the agent did not write them, so their defects are not its residue. A file
    the agent *edited* still counts — the scenario handed it over to be changed.

    Never raises: a workspace that cannot be read, or a file that cannot be
    parsed, is reported through the returned scan rather than failing the run.
    A syntax error is itself a defect worth reporting — a script that does not
    parse certainly does not run.
    """
    excluded = {e.strip("/") for e in exclude}
    findings: list[Finding] = []
    scanned: list[str] = []
    skipped: list[str] = []

    try:
        candidates = sorted(workspace.rglob("*.py"))
    except OSError as exc:
        return ScriptScan([], [], [f"workspace unreadable: {exc}"])

    for path in candidates:
        try:
            rel = str(path.relative_to(workspace))
        except ValueError:  # pragma: no cover - rglob results are always under root
            continue
        parts = set(Path(rel).parts)
        if parts & _SKIP_DIRS:
            continue
        if rel in excluded:
            continue
        try:
            size = path.stat().st_size
            if size > max_bytes:
                skipped.append(f"{rel} ({size}B > {max_bytes})")
                continue
            source = path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            skipped.append(f"{rel} (unreadable: {exc})")
            continue

        scanned.append(rel)
        try:
            chunk, parsed = _analyse_source(source, rel)
        except Exception as exc:  # analyser bug must not fail the run
            skipped.append(f"{rel} (analyser error: {type(exc).__name__}: {exc})")
            continue
        if not parsed:
            findings.append(Finding(rel, 1, "unparseable-script", "file does not parse"))
            continue
        findings.extend(chunk)

    return ScriptScan(findings, scanned, skipped)
