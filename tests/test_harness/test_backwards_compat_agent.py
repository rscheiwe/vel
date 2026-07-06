"""Backwards-compatibility contract (§8) — agent/loop perspective.

These are executable assertions of the HARD GATE: Harness Mode must be purely
additive. With no harness config, run_stream must behave exactly as before.
"""
from __future__ import annotations

import inspect
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
        batch = self._script.pop(0) if self._script else []
        for ev in batch:
            yield ev

    async def generate(self, messages, model, tools, generation_config=None):
        return {'done': True}


def _text(t: str) -> List[Any]:
    return [TextStartEvent(block_id='b'), TextDeltaEvent(block_id='b', delta=t),
            TextEndEvent(block_id='b'), FinishMessageEvent(finish_reason='stop')]


def _tool(cid: str, name: str, args: Dict[str, Any]) -> List[Any]:
    return [ToolInputAvailableEvent(tool_call_id=cid, tool_name=name, input=args),
            FinishMessageEvent(finish_reason='tool_calls')]


async def echo(q: str = '', ctx: dict = None) -> dict:
    return {'echo': q}


async def _collect(agen) -> List[Dict[str, Any]]:
    return [ev async for ev in agen]


def _agent(provider, with_tool=False, **kw):
    tools = [ToolSpec.from_function(echo)] if with_tool else None
    a = Agent(id='bc', model={'provider': 'scripted', 'model': 'm'}, tools=tools, **kw)
    a._custom_provider = provider
    return a


# §8.2 — no signature changes: only new optional kwargs / methods.
def test_run_stream_signature_only_adds_optional_kwargs():
    sig = inspect.signature(Agent.run_stream)
    assert 'harness' in sig.parameters
    assert sig.parameters['harness'].default is None
    # the original parameters are still present and optional
    for p in ('input', 'session_id', 'generation_config', 'rlm', 'thinking'):
        assert p in sig.parameters


def test_init_harness_kwarg_defaults_none():
    sig = inspect.signature(Agent.__init__)
    assert 'harness' in sig.parameters
    assert sig.parameters['harness'].default is None


def test_resume_method_exists():
    assert inspect.iscoroutinefunction(Agent.resume) or inspect.isasyncgenfunction(Agent.resume)


# §8.1 — default-off: golden non-harness event sequence (text path).
@pytest.mark.asyncio
async def test_golden_text_sequence_unchanged():
    provider = ScriptedProvider([_text('hello')])
    agent = _agent(provider)
    events = await _collect(agent.run_stream({'message': 'hi'}))
    types = [e['type'] for e in events]
    assert types[0] == 'start'
    assert types[1] == 'start-step'
    assert 'text-start' in types
    assert 'text-delta' in types
    assert 'text-end' in types
    assert types[-2] == 'finish-step'
    assert types[-1] == 'finish'
    # §8.3 — no harness events leak into a non-harness run
    assert not any(t.startswith('data-harness-') for t in types)


# §8.1 — default-off: golden non-harness event sequence (tool path).
@pytest.mark.asyncio
async def test_golden_tool_sequence_unchanged():
    provider = ScriptedProvider([_tool('c1', 'echo', {'q': 'x'}), _text('done')])
    agent = _agent(provider, with_tool=True)
    events = await _collect(agent.run_stream({'message': 'use echo'}))
    types = [e['type'] for e in events]
    assert 'tool-input-available' in types
    assert 'tool-output-available' in types
    assert types[-1] == 'finish'
    assert not any(t.startswith('data-harness-') for t in types)


# §8.6 — inline approval callback preserved when harness is off.
@pytest.mark.asyncio
async def test_inline_tool_approval_callback_still_works():
    seen = {}

    async def approve_cb(name, args, call_id):
        seen['called'] = (name, call_id)
        return False  # deny

    provider = ScriptedProvider([_tool('c1', 'echo', {'q': 'x'}), _text('done')])
    agent = _agent(provider, with_tool=True, tool_approval_callback=approve_cb)
    events = await _collect(agent.run_stream({'message': 'use echo'}))
    assert seen.get('called') == ('echo', 'c1')
    # denied tool returns an error result, not the real output
    out = next(e for e in events if e['type'] == 'tool-output-available')
    assert 'error' in out['output']


# §8.3 — harness on still emits all the normal event shapes (additive only).
@pytest.mark.asyncio
async def test_harness_on_is_superset_of_normal_events():
    provider_off = ScriptedProvider([_text('hi')])
    agent_off = _agent(provider_off)
    off = [e['type'] for e in await _collect(agent_off.run_stream({'message': 'x'}))]

    provider_on = ScriptedProvider([_text('hi')])
    agent_on = _agent(provider_on, harness={'enabled': True})
    on = [e['type'] for e in await _collect(agent_on.run_stream({'message': 'x'}))]

    # every non-harness event type from the off-run also appears in the on-run
    on_non_harness = [t for t in on if not t.startswith('data-harness-')]
    assert on_non_harness == off
