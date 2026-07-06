"""M1: the shared turn atom _stream_llm_call — one LLM call, normalized events.

Serves the loop/reflection engine: forwards events, can re-tag text as reasoning,
and reports results via the mutable `out` dict.
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
    ResponseMetadataEvent,
    FinishMessageEvent,
)
from vel.providers import BaseProvider


class OneShotProvider(BaseProvider):
    name = 'oneshot'

    def __init__(self, events: List[Any]):
        self._events = events

    async def stream(self, messages, model, tools, generation_config=None):
        for ev in self._events:
            yield ev

    async def generate(self, messages, model, tools, generation_config=None):
        return {}


async def echo(x: str = '', ctx: dict = None) -> dict:
    return {'echo': x}


def _agent(events):
    a = Agent(id='atom', model={'provider': 'oneshot', 'model': 'm'},
              tools=[ToolSpec.from_function(echo, name='echo')])
    a._custom_provider = OneShotProvider(events)
    return a


@pytest.mark.asyncio
async def test_emit_as_text_forwards_and_collects():
    events = [
        TextStartEvent(block_id='b'),
        TextDeltaEvent(block_id='b', delta='hello '),
        TextDeltaEvent(block_id='b', delta='world'),
        TextEndEvent(block_id='b'),
        ResponseMetadataEvent(usage={'total_tokens': 11}),
        FinishMessageEvent(finish_reason='stop'),
    ]
    agent = _agent(events)
    out: Dict[str, Any] = {}
    types = [e.get('type') async for e in agent._stream_llm_call([{'role': 'user', 'content': 'hi'}], out=out)]

    assert 'text-start' in types and 'text-delta' in types and 'text-end' in types
    assert 'reasoning-delta' not in types
    assert out['text'] == 'hello world'
    assert out['usage'] == {'total_tokens': 11}
    assert out['finish_reason'] == 'stop'


@pytest.mark.asyncio
async def test_emit_as_reasoning_retags_text():
    events = [
        TextStartEvent(block_id='b'),
        TextDeltaEvent(block_id='b', delta='thinking...'),
        TextEndEvent(block_id='b'),
        FinishMessageEvent(finish_reason='stop'),
    ]
    agent = _agent(events)
    out: Dict[str, Any] = {}
    collected = [e async for e in agent._stream_llm_call(
        [{'role': 'user', 'content': 'hi'}], out=out, emit_as='reasoning')]
    types = [e.get('type') for e in collected]

    # Text is re-tagged as reasoning; no raw text-* leaks.
    assert types == ['reasoning-start', 'reasoning-delta', 'reasoning-end']
    assert all(e.get('type', '').startswith('reasoning') for e in collected)
    # The reasoning content is still collected as text for the phase state.
    assert out['text'] == 'thinking...'


@pytest.mark.asyncio
async def test_collects_tool_calls():
    events = [
        ToolInputAvailableEvent(tool_call_id='c1', tool_name='echo', input={'x': 'a'}),
        FinishMessageEvent(finish_reason='tool_calls'),
    ]
    agent = _agent(events)
    out: Dict[str, Any] = {}
    _ = [e async for e in agent._stream_llm_call([{'role': 'user', 'content': 'hi'}], out=out)]
    assert out['tool_calls'] == [{'tool_call_id': 'c1', 'tool_name': 'echo', 'input': {'x': 'a'}}]
    assert out['finish_reason'] == 'tool_calls'
