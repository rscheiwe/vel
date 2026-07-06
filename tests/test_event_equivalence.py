"""Event-equivalence tripwire — the sacred-path guard for the loop consolidation.

The non-harness single-agent path must emit a byte-identical V5 stream-event
sequence across every refactor milestone (ontql-ui consumes these events). These
baselines were captured on `main` before the loop-consolidation work; any change
to the base event stream must be intentional and update these snapshots in the
same commit.
"""
from __future__ import annotations

from typing import Any, Dict, List

import pytest

from vel import Agent, ToolSpec
from vel.events import (
    TextStartEvent,
    TextDeltaEvent,
    TextEndEvent,
    ToolInputAvailableEvent,
    FinishMessageEvent,
)
from vel.providers import BaseProvider


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


def _tool(cid: str, name: str, args: Dict[str, Any]) -> List[Any]:
    return [ToolInputAvailableEvent(tool_call_id=cid, tool_name=name, input=args),
            FinishMessageEvent(finish_reason='tool_calls')]


async def echo(x: str = '', ctx: dict = None) -> dict:
    return {'echo': x}


async def _types(script: List[List[Any]]) -> List[str]:
    agent = Agent(
        id='eq',
        model={'provider': 'scripted', 'model': 'm'},
        tools=[ToolSpec.from_function(echo, name='echo')],
    )
    agent._custom_provider = ScriptedProvider(script)
    return [e.get('type') async for e in agent.run_stream({'message': 'hi'})]


# Baselines captured on `main` (pre loop-consolidation). Do not edit without a
# deliberate, reviewed change to the base event stream.
NO_TOOL = ['start', 'start-step', 'text-start', 'text-delta', 'text-end', 'finish-step', 'finish']
ONE_TOOL = ['start', 'start-step', 'tool-input-available', 'tool-output-available',
            'finish-step', 'start-step', 'text-start', 'text-delta', 'text-end',
            'finish-step', 'finish']
MULTI = ['start', 'start-step', 'tool-input-available', 'tool-output-available',
         'finish-step', 'start-step', 'tool-input-available', 'tool-output-available',
         'finish-step', 'start-step', 'text-start', 'text-delta', 'text-end',
         'finish-step', 'finish']


@pytest.mark.asyncio
async def test_no_tool_text_run_event_sequence():
    assert await _types([_text('hello')]) == NO_TOOL


@pytest.mark.asyncio
async def test_single_tool_run_event_sequence():
    assert await _types([_tool('c1', 'echo', {'x': 'a'}), _text('done')]) == ONE_TOOL


@pytest.mark.asyncio
async def test_multi_step_run_event_sequence():
    assert await _types([
        _tool('c1', 'echo', {'x': 'a'}),
        _tool('c2', 'echo', {'x': 'b'}),
        _text('done'),
    ]) == MULTI


@pytest.mark.asyncio
async def test_event_payloads_preserved():
    """Beyond types: text deltas and tool identities must survive too."""
    agent = Agent(
        id='eq',
        model={'provider': 'scripted', 'model': 'm'},
        tools=[ToolSpec.from_function(echo, name='echo')],
    )
    agent._custom_provider = ScriptedProvider([_tool('c1', 'echo', {'x': 'a'}), _text('hi there')])
    events = [e async for e in agent.run_stream({'message': 'hi'})]

    tool_in = next(e for e in events if e['type'] == 'tool-input-available')
    assert tool_in['toolName'] == 'echo' and tool_in['toolCallId'] == 'c1'
    tool_out = next(e for e in events if e['type'] == 'tool-output-available')
    assert tool_out['output'] == {'echo': 'a'}
    deltas = ''.join(e.get('delta', '') for e in events if e['type'] == 'text-delta')
    assert deltas == 'hi there'
