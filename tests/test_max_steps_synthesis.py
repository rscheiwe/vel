"""The max-steps path must emit the same event vocabulary as every other path.

`_synthesize_final` runs when a run exhausts `max_steps`: one final tool-less
model call to produce an answer. It used to forward `finish-message`, which no
other path does — and `finish-message` is not a member of the AI SDK UI Message
Stream union at all (verified against ai@6.0.149: zero occurrences).

A strict client parses each chunk against that union, so an unknown `type`
matches no member and fails with invalid_union, discarding the whole stream.
The practical effect was that any run long enough to hit `max_steps` broke the
browser, and only that kind of run — which is exactly the sort of bug that
survives happy-path testing.
"""
from __future__ import annotations

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

# Every part type the AI SDK UI Message Stream accepts. `finish-message` is
# deliberately absent — that is the point of this module.
AI_SDK_PART_TYPES = {
    'start', 'text-start', 'text-delta', 'text-end',
    'reasoning-start', 'reasoning-delta', 'reasoning-end',
    'tool-input-start', 'tool-input-delta', 'tool-input-available', 'tool-input-error',
    'tool-output-available', 'tool-output-error', 'tool-output-denied',
    'source-url', 'source-document', 'file',
    'start-step', 'finish-step', 'start', 'finish', 'abort',
    'error', 'message-metadata',
}


class LoopingProvider(BaseProvider):
    """Always asks for another tool call, so the run runs out of steps."""

    name = 'scripted'

    def __init__(self) -> None:
        self.calls = 0

    async def stream(self, messages, model, tools, generation_config=None):
        self.calls += 1
        # `_synthesize_final` calls the provider with tools=[]; that final call
        # must answer rather than loop, or nothing terminates.
        if not tools:
            yield TextStartEvent(block_id='f')
            yield TextDeltaEvent(block_id='f', delta='synthesized')
            yield TextEndEvent(block_id='f')
            yield FinishMessageEvent(finish_reason='stop')
            return

        yield ToolInputAvailableEvent(
            tool_call_id=f'c{self.calls}', tool_name='echo', input={'x': 'a'}
        )
        yield FinishMessageEvent(finish_reason='tool_calls')

    async def generate(self, messages, model, tools, generation_config=None):
        return {}


async def echo(x: str = '', ctx: dict = None) -> dict:
    return {'echo': x}


async def _run_until_max_steps() -> List[Dict[str, Any]]:
    agent = Agent(
        id='maxsteps',
        model={'provider': 'scripted', 'model': 'm'},
        tools=[ToolSpec.from_function(echo, name='echo')],
        policies={'max_steps': 2},
    )
    agent._custom_provider = LoopingProvider()
    return [e async for e in agent.run_stream({'message': 'hi'})]


@pytest.mark.asyncio
async def test_max_steps_path_emits_only_ai_sdk_part_types():
    events = await _run_until_max_steps()
    unknown = sorted({
        e['type'] for e in events
        if e['type'] not in AI_SDK_PART_TYPES and not e['type'].startswith('data-')
    })
    assert unknown == [], f'these would be rejected by a strict client: {unknown}'


@pytest.mark.asyncio
async def test_finish_message_is_never_forwarded():
    events = await _run_until_max_steps()
    assert not any(e['type'] == 'finish-message' for e in events)


@pytest.mark.asyncio
async def test_max_steps_run_still_produces_an_answer_and_terminates():
    events = await _run_until_max_steps()
    deltas = ''.join(e.get('delta', '') for e in events if e['type'] == 'text-delta')
    assert deltas == 'synthesized'
    assert events[-1]['type'] == 'finish'


@pytest.mark.asyncio
async def test_max_steps_stream_satisfies_the_invariants():
    assert_stream_invariants(await _run_until_max_steps())
