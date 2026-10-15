"""Tests for trajectory normalization against real captured opencode streams."""

from pathlib import Path

from dr_agents_tester.eval.trajectory import (
    ALL_CAPABILITIES,
    EventKind,
    OpenCodeAdapter,
    TrajectoryEvent,
    compute_metrics,
)

FIXTURES = Path(__file__).parent / "fixtures" / "opencode"


def _parse(fixture: str) -> list[TrajectoryEvent]:
    return OpenCodeAdapter().parse((FIXTURES / fixture).read_text().splitlines())


class TestOpenCodeAdapterOnRealStreams:
    def test_simple_text_stream(self) -> None:
        events = _parse("simple_text.jsonl")
        kinds = [e.kind for e in events]
        assert kinds == [EventKind.TURN_START, EventKind.TEXT, EventKind.TURN_FINISH]
        metrics = compute_metrics(events, wall_seconds=3.0)
        assert metrics.num_turns == 1
        assert metrics.num_tool_calls == 0
        assert metrics.input_tokens == 7983
        assert metrics.output_tokens == 4
        assert metrics.total_tokens == 7987
        assert metrics.skill_triggered is False

    def test_tool_events_stream(self) -> None:
        """write/read/bash journey: 4 steps, 3 tool calls, bash exit 3 counts as error."""
        events = _parse("tool_events.jsonl")
        tool_calls = [e for e in events if e.kind == EventKind.TOOL_CALL]
        assert [t.name for t in tool_calls] == ["write", "read", "bash"]
        # write/read completed ok; the bash command exited 3 → ok False
        assert [t.ok for t in tool_calls] == [True, True, False]

        metrics = compute_metrics(events)
        assert metrics.num_turns == 4
        assert metrics.num_tool_calls == 3
        assert metrics.num_errors == 1
        assert metrics.num_retries == 0
        assert metrics.skill_triggered is False
        assert metrics.total_tokens is not None and metrics.total_tokens > 0

    def test_skill_trigger_stream(self) -> None:
        events = _parse("skill_trigger.jsonl")
        triggers = [e for e in events if e.kind == EventKind.SKILL_TRIGGERED]
        assert len(triggers) == 1
        assert triggers[0].name == "probe-skill"

        metrics = compute_metrics(events)
        assert metrics.skill_triggered is True
        assert metrics.skills_used == ["probe-skill"]
        assert metrics.skill_first_turn == 1
        # The skill tool invocation is also a tool call
        assert metrics.num_tool_calls == 1

    def test_tool_error_stream(self) -> None:
        """Rejected read (status=error) counts as a failed tool call."""
        events = _parse("tool_error.jsonl")
        tool_calls = [e for e in events if e.kind == EventKind.TOOL_CALL]
        assert [t.name for t in tool_calls] == ["read"]
        assert tool_calls[0].ok is False
        assert "rejected" in str(tool_calls[0].detail.get("error", "")).lower()

        metrics = compute_metrics(events)
        assert metrics.num_errors == 1


class TestAdapterRobustness:
    def test_unknown_event_types_preserved(self) -> None:
        lines = [
            '{"type": "step_start", "part": {}}',
            '{"type": "totally_new_event", "part": {"whatever": 1}}',
            '{"type": "step_finish", "part": {"tokens": {"input": 5, "output": 2}}}',
        ]
        events = OpenCodeAdapter().parse(lines)
        unknown = [e for e in events if e.kind == EventKind.UNKNOWN]
        assert len(unknown) == 1
        assert unknown[0].raw["type"] == "totally_new_event"
        assert compute_metrics(events).num_unknown_events == 1

    def test_garbage_lines_become_error_events_not_crashes(self) -> None:
        lines = ["not json at all", '{"type": "text", "part": {"text": "hi"}}', "{broken"]
        events = OpenCodeAdapter().parse(lines)
        assert len(events) == 3
        assert [e.kind for e in events] == [EventKind.ERROR, EventKind.TEXT, EventKind.ERROR]

    def test_empty_stream_reports_unavailable(self) -> None:
        metrics = compute_metrics([], wall_seconds=12.0)
        assert metrics.wall_seconds == 12.0
        assert metrics.num_turns is None
        assert metrics.total_tokens is None
        assert metrics.skill_triggered is None

    def test_truncated_stream_still_normalizes(self) -> None:
        """A timeout-killed stream (no final step_finish) still yields metrics."""
        lines = [
            '{"type": "step_start", "part": {}}',
            '{"type": "tool_use", "part": {"tool": "bash", "state": {"status": "running", "input": {"command": "sleep 999"}}}}',
        ]
        events = OpenCodeAdapter().parse(lines)
        metrics = compute_metrics(events, wall_seconds=1800.0)
        assert metrics.num_turns == 1
        assert metrics.num_tool_calls == 1
        # In-flight tool call: ok is unknown, not an error
        assert metrics.num_errors == 0
        assert metrics.total_tokens is None


class TestCapabilities:
    def test_restricted_capabilities_report_none(self) -> None:
        events = _parse("tool_events.jsonl")
        metrics = compute_metrics(events, capabilities=frozenset({"turns"}))
        assert metrics.num_turns == 4
        assert metrics.num_tool_calls is None
        assert metrics.total_tokens is None
        assert metrics.skill_triggered is None
        assert metrics.available() == ["num_turns"]

    def test_all_capabilities_constant(self) -> None:
        assert "tokens" in ALL_CAPABILITIES and "skill_trigger" in ALL_CAPABILITIES


class TestRetryDetection:
    def _tool(self, name: str, ok: bool, seq: int) -> TrajectoryEvent:
        return TrajectoryEvent(kind=EventKind.TOOL_CALL, seq=seq, name=name, ok=ok)

    def test_same_tool_after_error_is_retry(self) -> None:
        events = [
            self._tool("bash", False, 0),
            self._tool("bash", True, 1),
        ]
        assert compute_metrics(events).num_retries == 1

    def test_different_tool_after_error_is_not_retry(self) -> None:
        events = [
            self._tool("bash", False, 0),
            self._tool("read", True, 1),
        ]
        assert compute_metrics(events).num_retries == 0

    def test_repeat_without_error_is_not_retry(self) -> None:
        events = [
            self._tool("read", True, 0),
            self._tool("read", True, 1),
        ]
        assert compute_metrics(events).num_retries == 0
