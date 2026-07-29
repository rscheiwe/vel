"""Cancellation, and the shape a cancelled stream has to keep.

Vel had none: no signal on `run_stream`, no `abort` event, `RunManager` holding
tasks it never cancelled, and a `'cancelled'` status that nothing ever wrote. A
cancelled run simply stopped mid-stream — no terminal event, blocks left open,
in-flight tools left as spinners.

The invariant these tests exist to hold, taken from the abort bugs harness-agent
actually shipped and had to fix:

    close every open text block
    -> close every open reasoning block
    -> tool-output-error for every tool that started but never returned
    -> finish-step for any open step
    -> abort
    -> finish

Cancelling is not an error. `abort` says someone stopped it; `error` continues
to mean the run failed. A client renders those differently, and a cancelled run
reported as an error reads as a bug in the agent.
"""
from __future__ import annotations

import asyncio
from typing import Any, Dict, List

import pytest

from vel import Agent, ToolSpec
from vel.events import (
    FinishMessageEvent,
    ReasoningDeltaEvent,
    ReasoningEndEvent,
    ReasoningStartEvent,
    TextDeltaEvent,
    TextEndEvent,
    TextStartEvent,
    ToolInputAvailableEvent,
)
from vel.providers import BaseProvider

from tests.helpers.invariants import assert_stream_invariants


class SlowProvider(BaseProvider):
    """Emits many deltas, so a cancel can land in the middle of a block."""

    name = 'scripted'

    def __init__(self, script: List[List[Any]]):
        self._script = list(script)

    async def stream(self, messages, model, tools, generation_config=None):
        batch = self._script.pop(0) if self._script else [
            TextStartEvent(block_id='e'), TextDeltaEvent(block_id='e', delta='done'),
            TextEndEvent(block_id='e'), FinishMessageEvent(finish_reason='stop'),
        ]
        for ev in batch:
            await asyncio.sleep(0.01)
            yield ev

    async def generate(self, messages, model, tools, generation_config=None):
        return {}


def _long_text() -> List[Any]:
    return (
        [TextStartEvent(block_id='b')]
        + [TextDeltaEvent(block_id='b', delta=f'chunk{i} ') for i in range(20)]
        + [TextEndEvent(block_id='b'), FinishMessageEvent(finish_reason='stop')]
    )


def _long_reasoning() -> List[Any]:
    return (
        [ReasoningStartEvent(block_id='r')]
        + [ReasoningDeltaEvent(block_id='r', delta=f'think{i} ') for i in range(20)]
        + [ReasoningEndEvent(block_id='r'), FinishMessageEvent(finish_reason='stop')]
    )


async def slow_tool(x: str = '', ctx: dict = None) -> dict:
    await asyncio.sleep(5)
    return {'never': 'reached'}


def _tool_step() -> List[Any]:
    return [
        ToolInputAvailableEvent(tool_call_id='tc1', tool_name='slow_tool', input={'x': 'a'}),
        FinishMessageEvent(finish_reason='tool_calls'),
    ]


def _agent(script) -> tuple[Agent, asyncio.Event]:
    agent = Agent(
        id='cancel',
        model={'provider': 'scripted', 'model': 'm'},
        tools=[ToolSpec.from_function(slow_tool, name='slow_tool')],
    )
    agent._custom_provider = SlowProvider(script)
    return agent, asyncio.Event()


async def _cancel_after(agent, token, *, stop_on: str, nth: int = 3) -> List[Dict[str, Any]]:
    """Consume until the nth `stop_on` event, then cancel and drain."""
    events: List[Dict[str, Any]] = []
    seen = 0
    async for event in agent.run_stream({'message': 'hi'}, cancel_token=token):
        events.append(event)
        if event['type'] == stop_on:
            seen += 1
            if seen >= nth:
                token.set()
    return events


def _types(events) -> List[str]:
    return [e['type'] for e in events]


@pytest.mark.asyncio
async def test_cancel_mid_text_closes_the_block_then_aborts():
    agent, token = _agent([_long_text()])
    events = await _cancel_after(agent, token, stop_on='text-delta')
    types = _types(events)

    assert types[-2:] == ['abort', 'finish']
    assert types.count('text-start') == types.count('text-end') == 1
    assert types.index('text-end') < types.index('abort')
    assert 'error' not in types, 'a cancel is an abort, not a failure'
    assert_stream_invariants(events)


@pytest.mark.asyncio
async def test_cancel_mid_reasoning_closes_the_block():
    agent, token = _agent([_long_reasoning()])
    events = await _cancel_after(agent, token, stop_on='reasoning-delta')
    types = _types(events)

    assert types[-2:] == ['abort', 'finish']
    assert types.count('reasoning-start') == types.count('reasoning-end') == 1
    assert_stream_invariants(events)


@pytest.mark.asyncio
async def test_cancel_with_a_tool_in_flight_closes_the_tool():
    """The spinner case: a tool announced but never returned.

    Cancelling has to give that call a terminal event keyed to the same id, or
    the client renders it as still running on a run that has already stopped.
    """
    agent, token = _agent([_tool_step(), _long_text()])
    events: List[Dict[str, Any]] = []
    async for event in agent.run_stream({'message': 'hi'}, cancel_token=token):
        events.append(event)
        if event['type'] == 'tool-input-available':
            token.set()

    types = _types(events)
    errors = [e for e in events if e['type'] == 'tool-output-error']
    assert len(errors) == 1 and errors[0]['toolCallId'] == 'tc1'
    assert types[-2:] == ['abort', 'finish']
    assert_stream_invariants(events)


@pytest.mark.asyncio
async def test_cancel_balances_open_steps():
    agent, token = _agent([_long_text()])
    events = await _cancel_after(agent, token, stop_on='text-delta')
    types = _types(events)
    assert types.count('start-step') == types.count('finish-step')


@pytest.mark.asyncio
async def test_nothing_is_emitted_after_finish():
    agent, token = _agent([_long_text()])
    events = await _cancel_after(agent, token, stop_on='text-delta')
    assert _types(events)[-1] == 'finish'
    assert _types(events).count('finish') == 1


@pytest.mark.asyncio
async def test_an_uncancelled_run_is_unaffected():
    """The token is opt-in; not setting it changes nothing."""
    agent, token = _agent([_long_text()])
    events = [e async for e in agent.run_stream({'message': 'hi'}, cancel_token=token)]
    types = _types(events)
    assert 'abort' not in types
    assert types[-1] == 'finish'
    assert_stream_invariants(events)


@pytest.mark.asyncio
async def test_no_cancel_token_at_all_still_works():
    agent, _ = _agent([_long_text()])
    events = [e async for e in agent.run_stream({'message': 'hi'})]
    assert 'abort' not in _types(events)
    assert_stream_invariants(events)


@pytest.mark.asyncio
async def test_abort_wire_shape_is_minimal():
    """Strict clients reject unknown sibling keys and discard the stream."""
    agent, token = _agent([_long_text()])
    events = await _cancel_after(agent, token, stop_on='text-delta')
    abort = next(e for e in events if e['type'] == 'abort')
    assert set(abort.keys()) <= {'type', 'reason'}
