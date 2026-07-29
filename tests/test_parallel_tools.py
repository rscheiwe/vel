"""Opt-in concurrent tool execution.

Vel has always run a step's tool calls one after another, so callers are
entitled to assume ordered side effects. Concurrency is therefore opt-in at two
levels — the tool declares `parallel_safe`, and the agent enables
`policies={'tool_execution': 'parallel'}` — and a batch runs concurrently only
if every call in it qualifies.

The assertion that matters is wall-clock. The event sequence is identical
whether tools run serially or concurrently, because the model announces every
call before any executes; that is why the parity eval measured Vel at 63.4s and
pi at 31.1s on the same transcript. Timing is the only observable difference,
so timing is what these tests check.
"""
from __future__ import annotations

import asyncio
import time
from typing import Any, Dict, List

import pytest

from vel import Agent, ToolSpec
from vel.events import (
    FinishMessageEvent,
    TextDeltaEvent,
    TextEndEvent,
    TextStartEvent,
    ToolInputAvailableEvent,
)
from vel.providers import BaseProvider

from tests.helpers.invariants import assert_stream_invariants

DELAY = 0.30
CALLS: List[str] = []


@pytest.fixture(autouse=True)
def _reset():
    CALLS.clear()


class ScriptedProvider(BaseProvider):
    name = 'scripted'

    def __init__(self, script: List[List[Any]]):
        self._script = list(script)

    async def stream(self, messages, model, tools, generation_config=None):
        batch = self._script.pop(0) if self._script else [
            TextStartEvent(block_id='e'), TextDeltaEvent(block_id='e', delta='done'),
            TextEndEvent(block_id='e'), FinishMessageEvent(finish_reason='stop'),
        ]
        for ev in batch:
            yield ev

    async def generate(self, messages, model, tools, generation_config=None):
        return {}


def _text(t: str) -> List[Any]:
    return [TextStartEvent(block_id='b'), TextDeltaEvent(block_id='b', delta=t),
            TextEndEvent(block_id='b'), FinishMessageEvent(finish_reason='stop')]


def _two_tools() -> List[Any]:
    """Two calls in ONE step — the only shape where parallelism is observable."""
    return [
        ToolInputAvailableEvent(tool_call_id='tc1', tool_name='slow_a', input={'x': 'a'}),
        ToolInputAvailableEvent(tool_call_id='tc2', tool_name='slow_b', input={'x': 'b'}),
        FinishMessageEvent(finish_reason='tool_calls'),
    ]


async def slow_a(x: str = '', ctx: dict = None) -> dict:
    await asyncio.sleep(DELAY)
    CALLS.append('a')
    return {'ran': 'a'}


async def slow_b(x: str = '', ctx: dict = None) -> dict:
    await asyncio.sleep(DELAY)
    CALLS.append('b')
    return {'ran': 'b'}


def _tools(parallel_safe: bool, second_safe: bool | None = None) -> List[ToolSpec]:
    if second_safe is None:
        second_safe = parallel_safe
    return [
        ToolSpec.from_function(slow_a, name='slow_a', parallel_safe=parallel_safe),
        ToolSpec.from_function(slow_b, name='slow_b', parallel_safe=second_safe),
    ]


async def _timed_run(tools, policies) -> tuple[float, List[Dict[str, Any]]]:
    agent = Agent(id='par', model={'provider': 'scripted', 'model': 'm'},
                  tools=tools, policies=policies)
    agent._custom_provider = ScriptedProvider([_two_tools(), _text('done')])
    started = time.monotonic()
    events = [e async for e in agent.run_stream({'message': 'hi'})]
    return time.monotonic() - started, events


PARALLEL = {'max_steps': 24, 'tool_execution': 'parallel'}
DEFAULT = {'max_steps': 24}


@pytest.mark.asyncio
async def test_parallel_batch_takes_about_the_slowest_not_the_sum():
    elapsed, events = await _timed_run(_tools(parallel_safe=True), PARALLEL)
    assert elapsed < DELAY * 1.8, f'{elapsed:.2f}s suggests serial execution'
    assert len(CALLS) == 2
    assert_stream_invariants(events)


@pytest.mark.asyncio
async def test_default_policy_is_still_serial():
    """The whole point of opt-in: existing callers see no change."""
    elapsed, events = await _timed_run(_tools(parallel_safe=True), DEFAULT)
    assert elapsed >= DELAY * 1.8, f'{elapsed:.2f}s suggests it ran concurrently'
    assert CALLS == ['a', 'b'], 'serial execution must preserve call order'


@pytest.mark.asyncio
async def test_policy_on_but_tools_not_marked_is_serial():
    elapsed, _ = await _timed_run(_tools(parallel_safe=False), PARALLEL)
    assert elapsed >= DELAY * 1.8
    assert CALLS == ['a', 'b']


@pytest.mark.asyncio
async def test_one_unsafe_tool_makes_the_whole_batch_serial():
    """All-or-nothing. A serial tool must not observe a concurrent one's writes
    at an unpredictable point, so partitioning the batch is not offered."""
    elapsed, _ = await _timed_run(_tools(parallel_safe=True, second_safe=False), PARALLEL)
    assert elapsed >= DELAY * 1.8
    assert CALLS == ['a', 'b']


@pytest.mark.asyncio
async def test_event_sequence_is_identical_either_way():
    """Timing is the only observable difference — the contract is unchanged."""
    _, parallel_events = await _timed_run(_tools(parallel_safe=True), PARALLEL)
    CALLS.clear()
    _, serial_events = await _timed_run(_tools(parallel_safe=True), DEFAULT)
    assert [e['type'] for e in parallel_events] == [e['type'] for e in serial_events]


@pytest.mark.asyncio
async def test_a_failure_in_a_parallel_batch_still_recovers():
    """A concurrent failure must take the same path as a serial one."""
    async def boom(x: str = '', ctx: dict = None) -> dict:
        raise ValueError('parallel boom')

    tools = [
        ToolSpec.from_function(boom, name='slow_a', parallel_safe=True),
        ToolSpec.from_function(slow_b, name='slow_b', parallel_safe=True),
    ]
    _, events = await _timed_run(tools, PARALLEL)

    errors = [e for e in events if e['type'] == 'tool-output-error']
    assert len(errors) == 1 and errors[0]['toolCallId'] == 'tc1'
    assert 'parallel boom' in errors[0]['errorText']
    # The other tool still succeeded, and the run still answered.
    assert any(e['type'] == 'tool-output-available' for e in events)
    assert events[-1]['type'] == 'finish'
    assert_stream_invariants(events)


@pytest.mark.asyncio
async def test_approval_callback_forces_serial_execution():
    """A decision has to be made before the side effect, not after it."""
    agent = Agent(id='par', model={'provider': 'scripted', 'model': 'm'},
                  tools=_tools(parallel_safe=True), policies=PARALLEL)
    agent._custom_provider = ScriptedProvider([_two_tools(), _text('done')])

    async def approve(tool_name, args, tool_call_id):
        return True

    agent._tool_approval_callback = approve
    started = time.monotonic()
    [e async for e in agent.run_stream({'message': 'hi'})]
    assert time.monotonic() - started >= DELAY * 1.8
