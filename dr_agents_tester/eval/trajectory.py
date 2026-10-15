"""Trajectory normalization: per-driver event streams → common events → metrics.

Per-driver adapters convert raw transcripts into :class:`TrajectoryEvent`
streams; :func:`compute_metrics` derives efficiency metrics from the common
stream, so scoring never knows which agent ran (design doc §4.3). Unknown
event types are preserved verbatim and counted — never dropped, never fatal —
so an agent-CLI version bump degrades measurement instead of crashing runs.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from enum import Enum

from .models import TrajectoryMetrics

# The built-in OpenCode tool through which skills load; a call to it *is* the
# observable skill-trigger event (verified in the driver spikes).
SKILL_TOOL_NAME = "skill"


class EventKind(str, Enum):
    """Normalized event vocabulary shared by all drivers."""

    TURN_START = "turn_start"
    TEXT = "text"
    TOOL_CALL = "tool_call"
    SKILL_TRIGGERED = "skill_triggered"
    TURN_FINISH = "turn_finish"
    ERROR = "error"
    DONE = "done"
    UNKNOWN = "unknown"


@dataclass
class TrajectoryEvent:
    """One normalized trajectory event.

    ``raw`` always preserves the original driver event so the archive loses
    nothing when the normalized view is too coarse.
    """

    kind: EventKind
    seq: int
    turn: int | None = None
    name: str | None = None
    ok: bool | None = None
    detail: dict[str, object] = field(default_factory=dict)
    raw: dict[str, object] = field(default_factory=dict)


#: Metric names a fully-instrumented driver can populate. Drivers declare a
#: subset; compute_metrics reports None for anything outside it.
ALL_CAPABILITIES: frozenset[str] = frozenset(
    {"tokens", "cost", "turns", "tool_calls", "errors", "skill_trigger"}
)


def compute_metrics(
    events: Sequence[TrajectoryEvent],
    wall_seconds: float | None = None,
    capabilities: frozenset[str] = ALL_CAPABILITIES,
) -> TrajectoryMetrics:
    """Derive driver-agnostic efficiency metrics from a normalized event stream.

    Metrics outside ``capabilities`` are reported as ``None`` ("unavailable
    from this driver") rather than fake zeros. An empty stream reports
    everything (except wall time) as unavailable.
    """
    metrics = TrajectoryMetrics(wall_seconds=wall_seconds)
    metrics.num_unknown_events = sum(1 for e in events if e.kind == EventKind.UNKNOWN)

    if not events:
        return metrics

    if "turns" in capabilities:
        metrics.num_turns = sum(1 for e in events if e.kind == EventKind.TURN_START)

    if "tool_calls" in capabilities:
        tool_calls = [e for e in events if e.kind == EventKind.TOOL_CALL]
        metrics.num_tool_calls = len(tool_calls)

    if "errors" in capabilities:
        error_events = [
            e
            for e in events
            if e.kind == EventKind.ERROR or (e.kind == EventKind.TOOL_CALL and e.ok is False)
        ]
        metrics.num_errors = len(error_events)
        metrics.num_retries = _count_retries(events)

    if "skill_trigger" in capabilities:
        triggers = [e for e in events if e.kind == EventKind.SKILL_TRIGGERED]
        metrics.skill_triggered = bool(triggers)
        metrics.skills_used = list(dict.fromkeys(e.name for e in triggers if e.name))
        metrics.skill_first_turn = triggers[0].turn if triggers else None

    if "tokens" in capabilities:
        totals = {"input": 0, "output": 0, "cache_read": 0, "cache_write": 0}
        saw_tokens = False
        for e in events:
            if e.kind != EventKind.TURN_FINISH:
                continue
            tokens = e.detail.get("tokens")
            if not isinstance(tokens, dict):
                continue
            saw_tokens = True
            totals["input"] += int(tokens.get("input", 0) or 0)
            totals["output"] += int(tokens.get("output", 0) or 0)
            cache = tokens.get("cache")
            if isinstance(cache, dict):
                totals["cache_read"] += int(cache.get("read", 0) or 0)
                totals["cache_write"] += int(cache.get("write", 0) or 0)
        if saw_tokens:
            metrics.input_tokens = totals["input"]
            metrics.output_tokens = totals["output"]
            metrics.cache_read_tokens = totals["cache_read"]
            metrics.cache_write_tokens = totals["cache_write"]
            metrics.total_tokens = totals["input"] + totals["output"]

    if "cost" in capabilities:
        costs = [
            float(c)
            for e in events
            if e.kind == EventKind.TURN_FINISH
            and isinstance(c := e.detail.get("cost"), (int, float))
        ]
        if costs:
            metrics.cost = round(sum(costs), 6)

    return metrics


def _count_retries(events: Sequence[TrajectoryEvent]) -> int:
    """Count tool calls that repeat the same tool immediately after it errored."""
    retries = 0
    last_failed_tool: str | None = None
    for e in events:
        if e.kind != EventKind.TOOL_CALL:
            continue
        if last_failed_tool is not None and e.name == last_failed_tool:
            retries += 1
        last_failed_tool = e.name if e.ok is False else None
    return retries


class OpenCodeAdapter:
    """Normalize `opencode run --format json` JSONL streams.

    Raw event shape (verified against opencode 1.17):
    ``{"type": <underscored>, "timestamp": ..., "part": {"type": <hyphenated>, ...}}``
    with types ``step_start`` / ``text`` / ``tool_use`` / ``step_finish``.
    Tool state lives at ``part.state``: status ``completed`` or ``error``,
    ``input``, ``output``, ``metadata`` (bash exit code at ``metadata.exit``),
    or ``error`` string when status is ``error``. Skill loading is a
    ``tool_use`` of the built-in ``skill`` tool with the skill name at
    ``state.input.name``.
    """

    CAPABILITIES: frozenset[str] = ALL_CAPABILITIES

    def parse(self, raw_lines: Iterable[str]) -> list[TrajectoryEvent]:
        events: list[TrajectoryEvent] = []
        turn = 0
        seq = 0
        for line in raw_lines:
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError:
                events.append(
                    TrajectoryEvent(
                        kind=EventKind.ERROR,
                        seq=seq,
                        turn=turn or None,
                        detail={"unparseable_line": line[:200]},
                    )
                )
                seq += 1
                continue
            if not isinstance(raw, dict):
                raw = {"value": raw}

            for event in self._classify(raw, turn):
                event.seq = seq
                seq += 1
                if event.kind == EventKind.TURN_START:
                    turn += 1
                    event.turn = turn
                events.append(event)
        return events

    def _classify(self, raw: dict[str, object], turn: int) -> list[TrajectoryEvent]:
        """Map one raw opencode event to normalized event(s)."""
        etype = raw.get("type")
        part = raw.get("part")
        part = part if isinstance(part, dict) else {}
        current_turn = turn or None

        if etype == "step_start":
            return [TrajectoryEvent(kind=EventKind.TURN_START, seq=0, raw=raw)]

        if etype == "text":
            text = str(part.get("text", ""))
            return [
                TrajectoryEvent(
                    kind=EventKind.TEXT,
                    seq=0,
                    turn=current_turn,
                    detail={"text": text},
                    raw=raw,
                )
            ]

        if etype == "step_finish":
            detail: dict[str, object] = {}
            if isinstance(part.get("tokens"), dict):
                detail["tokens"] = part["tokens"]
            if isinstance(part.get("cost"), (int, float)):
                detail["cost"] = part["cost"]
            return [
                TrajectoryEvent(
                    kind=EventKind.TURN_FINISH, seq=0, turn=current_turn, detail=detail, raw=raw
                )
            ]

        if etype == "tool_use":
            return self._classify_tool_use(raw, part, current_turn)

        return [TrajectoryEvent(kind=EventKind.UNKNOWN, seq=0, turn=current_turn, raw=raw)]

    def _classify_tool_use(
        self, raw: dict[str, object], part: dict[str, object], turn: int | None
    ) -> list[TrajectoryEvent]:
        tool = str(part.get("tool", ""))
        state = part.get("state")
        state = state if isinstance(state, dict) else {}
        status = state.get("status")
        input_data = state.get("input")
        input_data = input_data if isinstance(input_data, dict) else {}

        ok: bool | None
        if status == "error":
            ok = False
        elif status == "completed":
            # A completed bash call whose command exited non-zero counts as an
            # error signal for efficiency metrics.
            metadata = state.get("metadata")
            exit_code = metadata.get("exit") if isinstance(metadata, dict) else None
            ok = not (isinstance(exit_code, int) and exit_code != 0)
        else:
            ok = None

        detail: dict[str, object] = {"status": status}
        if status == "error":
            detail["error"] = state.get("error")

        events = [
            TrajectoryEvent(
                kind=EventKind.TOOL_CALL,
                seq=0,
                turn=turn,
                name=tool,
                ok=ok,
                detail=detail,
                raw=raw,
            )
        ]
        if tool == SKILL_TOOL_NAME and status == "completed":
            skill_name = str(input_data.get("name", ""))
            events.append(
                TrajectoryEvent(
                    kind=EventKind.SKILL_TRIGGERED,
                    seq=0,
                    turn=turn,
                    name=skill_name or None,
                    detail={"skill": skill_name},
                    raw=raw,
                )
            )
        return events
