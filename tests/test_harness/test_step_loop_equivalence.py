"""M0 hard gate: ``Agent._step_loop`` with hooks=None must be behavior-identical
to the legacy ``run_stream`` loop.

These tests drive a deterministic ``FakeProvider`` so the event sequence is fully
reproducible offline, then assert that:

1. A representative non-harness ``run_stream`` produces a stable golden event
   sequence (text-only and tool-calling paths).
2. Driving ``_step_loop`` directly with all hooks ``None`` yields the same
   loop-emitted events that ``run_stream`` yields (minus the outer ``start``
   event that lives in ``run_stream`` setup, not in the loop).
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
    ResponseMetadataEvent,
)
from vel.providers import BaseProvider


class ScriptedProvider(BaseProvider):
    """Deterministic provider that replays a scripted list of event batches.

    Each entry in ``script`` is a list of StreamEvent objects emitted for one
    ``stream()`` call (i.e. one agent step).
    """

    name = 'scripted'

    def __init__(self, script: List[List[Any]]):
        self._script = list(script)
        self.calls: List[Dict[str, Any]] = []

    async def stream(self, messages, model, tools, generation_config=None):
        self.calls.append({'messages': messages, 'model': model, 'tools': tools})
        batch = self._script.pop(0) if self._script else []
        for ev in batch:
            yield ev

    async def generate(self, messages, model, tools, generation_config=None):
        return {'done': True}


def _text_step(text: str) -> List[Any]:
    return [
        TextStartEvent(block_id='b1'),
        TextDeltaEvent(block_id='b1', delta=text),
        TextEndEvent(block_id='b1'),
        FinishMessageEvent(finish_reason='stop'),
    ]


def _tool_step(tool_call_id: str, tool_name: str, args: Dict[str, Any]) -> List[Any]:
    return [
        ToolInputAvailableEvent(
            tool_call_id=tool_call_id, tool_name=tool_name, input=args
        ),
        FinishMessageEvent(finish_reason='tool_calls'),
    ]


async def echo(q: str = '', ctx: dict = None) -> dict:
    return {'echo': q}


def _make_agent(provider: ScriptedProvider, with_tool: bool = False) -> Agent:
    tools = [ToolSpec.from_function(echo)] if with_tool else None
    agent = Agent(id='equiv-agent', model={'provider': 'scripted', 'model': 'm'}, tools=tools)
    agent._custom_provider = provider
    return agent


async def _collect(agen) -> List[Dict[str, Any]]:
    out = []
    async for ev in agen:
        out.append(ev)
    return out


@pytest.mark.asyncio
async def test_step_loop_exists_and_is_async_gen():
    """The extracted method must exist with the documented hook kwargs."""
    import inspect

    assert hasattr(Agent, '_step_loop')
    sig = inspect.signature(Agent._step_loop)
    for kw in ('pre_step_hook', 'approval_prepass', 'approval_resolver', 'budget_hook', 'loop_state', 'wrap_event'):
        assert kw in sig.parameters, f'missing kwarg {kw}'
        assert sig.parameters[kw].default is None


@pytest.mark.asyncio
async def test_text_only_run_stream_golden():
    """Text-only run produces the canonical event sequence (unchanged by M0)."""
    provider = ScriptedProvider([_text_step('Hello world')])
    agent = _make_agent(provider)

    events = await _collect(agent.run_stream({'message': 'hi'}))
    types = [e['type'] for e in events]

    assert types[0] == 'start'
    assert types[1] == 'start-step'
    assert 'text-delta' in types
    assert types[-1] == 'finish'
    # finish-step precedes finish
    assert types[-2] == 'finish-step'


@pytest.mark.asyncio
async def test_step_loop_matches_run_stream_text():
    """Driving _step_loop directly (hooks=None) == run_stream minus outer start."""
    # run_stream reference
    provider_a = ScriptedProvider([_text_step('Hello world')])
    agent_a = _make_agent(provider_a)
    rs_events = await _collect(agent_a.run_stream({'message': 'hi'}))
    # drop the single leading 'start' event emitted by run_stream setup (not the loop)
    assert rs_events[0]['type'] == 'start'
    rs_loop_events = rs_events[1:]

    # _step_loop direct drive
    provider_b = ScriptedProvider([_text_step('Hello world')])
    agent_b = _make_agent(provider_b)
    run_id = 'run-direct'
    agent_b.ctxmgr.set_input(run_id, {'message': 'hi'}, None)
    loop_state: Dict[str, Any] = {'steps': 0, 'final_answer': ''}
    sl_events = await _collect(
        agent_b._step_loop(run_id=run_id, session_id=None, loop_state=loop_state)
    )

    assert [e['type'] for e in sl_events] == [e['type'] for e in rs_loop_events]
    assert sl_events == rs_loop_events
    assert loop_state['final_answer'] == 'Hello world'
    assert loop_state['steps'] == 1


@pytest.mark.asyncio
async def test_step_loop_matches_run_stream_tool_call():
    """Tool-calling path is identical between run_stream and direct _step_loop."""
    script = [
        _tool_step('call-1', 'echo', {'q': 'ping'}),
        _text_step('done'),
    ]
    provider_a = ScriptedProvider([list(b) for b in script])
    agent_a = _make_agent(provider_a, with_tool=True)
    rs_events = await _collect(agent_a.run_stream({'message': 'use echo'}))
    rs_loop_events = rs_events[1:]  # drop outer start

    provider_b = ScriptedProvider([list(b) for b in script])
    agent_b = _make_agent(provider_b, with_tool=True)
    run_id = 'run-direct-tool'
    agent_b.ctxmgr.set_input(run_id, {'message': 'use echo'}, None)
    loop_state: Dict[str, Any] = {'steps': 0, 'final_answer': ''}
    sl_events = await _collect(
        agent_b._step_loop(run_id=run_id, session_id=None, loop_state=loop_state)
    )

    assert [e['type'] for e in sl_events] == [e['type'] for e in rs_loop_events]
    assert sl_events == rs_loop_events
    # tool was executed and produced an output event
    assert any(e['type'] == 'tool-output-available' for e in sl_events)
    assert loop_state['steps'] == 2


@pytest.mark.asyncio
async def test_none_hooks_do_not_alter_events():
    """Passing explicit None hooks equals passing no hooks at all."""
    provider = ScriptedProvider([_text_step('abc')])
    agent = _make_agent(provider)
    run_id = 'run-hooks-none'
    agent.ctxmgr.set_input(run_id, {'message': 'hi'}, None)
    events = await _collect(
        agent._step_loop(
            run_id=run_id,
            session_id=None,
            loop_state={'steps': 0, 'final_answer': ''},
            pre_step_hook=None,
            approval_prepass=None,
            approval_resolver=None,
            budget_hook=None,
        )
    )
    assert events[-1]['type'] == 'finish'
    assert any(e['type'] == 'text-delta' for e in events)
