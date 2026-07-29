"""Invariant checks for Vel-native stream events.

These checks complement pinned event-sequence baselines. Baselines catch exact
ordering drift; this helper catches whole classes of invalid streams, especially
tool calls that open but never reach a terminal output/error event.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence


Event = Mapping[str, Any]

TERMINAL_EVENTS = {"finish", "error", "abort"}
TEXT_START_EVENTS = {"text-start", "text_start"}
TEXT_END_EVENTS = {"text-end", "text_end"}
REASONING_START_EVENTS = {"reasoning-start", "reasoning_start"}
REASONING_END_EVENTS = {"reasoning-end", "reasoning_end"}
TOOL_OPEN_EVENTS = {
    "tool-input-available",
    "tool_input_available",
    "tool-call",
    "tool_call",
}
TOOL_TERMINAL_EVENTS = {
    "tool-output-available",
    "tool_output_available",
    "tool-output-error",
    "tool_output_error",
    "tool-result",
    "tool_result",
}


@dataclass
class StreamInvariantResult:
    ok: bool
    errors: list[str] = field(default_factory=list)

    def assert_ok(self) -> None:
        if not self.ok:
            raise AssertionError("\n".join(self.errors))


def assert_stream_invariants(events: Iterable[Event]) -> None:
    """Assert that a Vel event stream is balanced and terminally well-formed."""

    check_stream_invariants(events).assert_ok()


def check_stream_invariants(events: Iterable[Event]) -> StreamInvariantResult:
    items = list(events)
    errors: list[str] = []

    starts = [idx for idx, event in enumerate(items) if _event_type(event) == "start"]
    if len(starts) != 1:
        errors.append(f"expected exactly one start event, saw {len(starts)}")
    elif starts[0] != 0:
        errors.append(f"expected first event to be start, saw {_event_type(items[0]) or 'none'}")

    terminal_indexes = [
        idx for idx, event in enumerate(items) if _event_type(event) in TERMINAL_EVENTS
    ]
    if len(terminal_indexes) != 1:
        errors.append(f"expected exactly one terminal event, saw {len(terminal_indexes)}")
    elif terminal_indexes[0] != len(items) - 1:
        after = [_event_type(event) or "unknown" for event in items[terminal_indexes[0] + 1 :]]
        errors.append(f"events appeared after terminal event: {after}")

    _check_balanced_blocks(
        items,
        start_types=TEXT_START_EVENTS,
        end_types=TEXT_END_EVENTS,
        id_names=("id", "textId", "text_id"),
        label="text",
        errors=errors,
    )
    _check_balanced_blocks(
        items,
        start_types=REASONING_START_EVENTS,
        end_types=REASONING_END_EVENTS,
        id_names=("id", "reasoningId", "reasoning_id"),
        label="reasoning",
        errors=errors,
    )
    _check_tools(items, errors)
    _check_steps(items, errors)

    return StreamInvariantResult(ok=not errors, errors=errors)


def _event_type(event: Event) -> str:
    value = event.get("type")
    return value if isinstance(value, str) else ""


def _first_string(event: Event, names: Sequence[str]) -> str | None:
    for name in names:
        value = event.get(name)
        if isinstance(value, str) and value:
            return value
    return None


def _check_balanced_blocks(
    events: Sequence[Event],
    *,
    start_types: set[str],
    end_types: set[str],
    id_names: Sequence[str],
    label: str,
    errors: list[str],
) -> None:
    open_ids: set[str] = set()

    for event in events:
        event_type = _event_type(event)
        if event_type in start_types:
            block_id = _first_string(event, id_names)
            if block_id is None:
                errors.append(f"{event_type} missing {label} id")
                continue
            if block_id in open_ids:
                errors.append(f"{label} block {block_id} started twice")
            open_ids.add(block_id)
        elif event_type in end_types:
            block_id = _first_string(event, id_names)
            if block_id is None:
                errors.append(f"{event_type} missing {label} id")
                continue
            if block_id not in open_ids:
                errors.append(f"{label} block {block_id} ended without a start")
            else:
                open_ids.remove(block_id)

    for block_id in sorted(open_ids):
        errors.append(f"{label} block {block_id} did not end")


def _check_tools(events: Sequence[Event], errors: list[str]) -> None:
    opened: set[str] = set()
    terminal_counts: dict[str, int] = {}

    for event in events:
        event_type = _event_type(event)
        if event_type in TOOL_OPEN_EVENTS:
            tool_call_id = _first_string(event, ("toolCallId", "tool_call_id", "id"))
            if tool_call_id is None:
                errors.append(f"{event_type} missing tool call id")
                continue
            opened.add(tool_call_id)
        elif event_type in TOOL_TERMINAL_EVENTS:
            tool_call_id = _first_string(event, ("toolCallId", "tool_call_id", "id"))
            if tool_call_id is None:
                errors.append(f"{event_type} missing tool call id")
                continue
            terminal_counts[tool_call_id] = terminal_counts.get(tool_call_id, 0) + 1

    for tool_call_id in sorted(opened):
        count = terminal_counts.get(tool_call_id, 0)
        if count == 0:
            errors.append(f"tool call {tool_call_id} did not produce output or error")
        elif count > 1:
            errors.append(f"tool call {tool_call_id} produced {count} terminal events")

    for tool_call_id, count in sorted(terminal_counts.items()):
        if tool_call_id not in opened:
            errors.append(f"tool call {tool_call_id} produced terminal event without input")


def _check_steps(events: Sequence[Event], errors: list[str]) -> None:
    open_steps = 0
    for event in events:
        event_type = _event_type(event)
        if event_type in {"start-step", "start_step"}:
            open_steps += 1
        elif event_type in {"finish-step", "finish_step"}:
            if open_steps == 0:
                errors.append("finish-step appeared without a matching start-step")
            else:
                open_steps -= 1

    if open_steps:
        errors.append(f"expected balanced start-step/finish-step events, {open_steps} still open")
