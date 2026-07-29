"""A raising tool must close its own call and let the run continue.

Before this, a tool exception emitted a global `error` and terminated. Two
distinct failures, and both are covered here because fixing only one leaves a
visible bug:

- the tool call never reached a terminal event, so clients left the tool part
  open forever (the spinner that never resolves)
- the model never saw the failure, so it could not retry or explain

`_run_tool_calls`'s `except Exception` branch had **no test at all** before this
file, so the characterization test below pins what the new behavior is rather
than what it was.
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


async def boom(x: str = '', ctx: dict = None) -> dict:
    raise ValueError('tool blew up')


async def echo(x: str = '', ctx: dict = None) -> dict:
    return {'echo': x}


def _agent(tools) -> Agent:
    return Agent(id='err', model={'provider': 'scripted', 'model': 'm'}, tools=tools)


async def _run(script, tools) -> List[Dict[str, Any]]:
    agent = _agent(tools)
    agent._custom_provider = ScriptedProvider(script)
    return [e async for e in agent.run_stream({'message': 'hi'})]


BOOM = ToolSpec.from_function(boom, name='boom')
ECHO = ToolSpec.from_function(echo, name='echo')


@pytest.mark.asyncio
async def test_raising_tool_emits_tool_output_error_for_its_own_id():
    events = await _run([_tool('c1', 'boom', {'x': 'a'}), _text('recovered')], [BOOM])

    errors = [e for e in events if e['type'] == 'tool-output-error']
    assert len(errors) == 1
    assert errors[0]['toolCallId'] == 'c1'
    assert 'tool blew up' in errors[0]['errorText']


@pytest.mark.asyncio
async def test_raising_tool_does_not_emit_a_global_error():
    """A contained tool failure is not a stream failure.

    Downstream shims treat a global `error` as the whole turn failing — sophee
    sets `streamFailed` on it — so leaking one here would misreport a recovered
    run as broken.
    """
    events = await _run([_tool('c1', 'boom', {'x': 'a'}), _text('recovered')], [BOOM])
    assert not any(e['type'] == 'error' for e in events)


@pytest.mark.asyncio
async def test_run_continues_to_an_answer_after_a_tool_raises():
    events = await _run([_tool('c1', 'boom', {'x': 'a'}), _text('recovered')], [BOOM])
    types = [e['type'] for e in events]

    assert 'text-start' in types, 'the model never got a chance to answer'
    deltas = ''.join(e.get('delta', '') for e in events if e['type'] == 'text-delta')
    assert deltas == 'recovered'
    assert types[-1] == 'finish'


@pytest.mark.asyncio
async def test_failure_is_fed_back_to_the_model_as_a_tool_result():
    """Without the tool-result message the transcript is malformed.

    The assistant message announcing the call would have no matching
    `role: 'tool'` reply, which both OpenAI and Anthropic reject on a resumed
    transcript — and the model would have no idea the tool failed.
    """
    agent = _agent([BOOM])
    agent._custom_provider = ScriptedProvider([_tool('c1', 'boom', {'x': 'a'}), _text('ok')])
    [e async for e in agent.run_stream({'message': 'hi'}, session_id='s1')]

    messages = agent.ctxmgr._by_session['s1']
    tool_replies = [m for m in messages if m.get('role') == 'tool']
    assert len(tool_replies) == 1
    assert tool_replies[0]['tool_call_id'] == 'c1'
    assert 'tool blew up' in tool_replies[0]['content']

    announced = [m for m in messages if m.get('role') == 'assistant' and m.get('tool_calls')]
    assert len(announced) == 1, 'the announcing assistant message must still be there'


@pytest.mark.asyncio
async def test_stream_satisfies_the_invariants():
    """Chiefly: every opened tool call reaches exactly one terminal event."""
    events = await _run([_tool('c1', 'boom', {'x': 'a'}), _text('recovered')], [BOOM])
    assert_stream_invariants(events)


@pytest.mark.asyncio
async def test_a_healthy_tool_is_unaffected():
    events = await _run([_tool('c1', 'echo', {'x': 'a'}), _text('done')], [ECHO])
    types = [e['type'] for e in events]
    assert 'tool-output-available' in types
    assert 'tool-output-error' not in types
    assert_stream_invariants(events)


@pytest.mark.asyncio
async def test_denied_tools_are_not_repainted_as_errors():
    """User denial keeps its existing shape: tool-output-available + {'error': …}.

    Four other test files assert that shape (approval policy, suspend/resume,
    backwards-compat, durable reflection). Denial is a decision, not a crash, so
    it deliberately does not become `tool-output-error`.
    """
    agent = _agent([ECHO])
    agent._custom_provider = ScriptedProvider([_tool('c1', 'echo', {'x': 'a'}), _text('ok')])

    async def deny(tool_name, args, tool_call_id):
        return False

    agent._tool_approval_callback = deny
    events = [e async for e in agent.run_stream({'message': 'hi'})]

    outputs = [e for e in events if e['type'] == 'tool-output-available']
    assert len(outputs) == 1
    assert 'denied' in outputs[0]['output']['error']
    assert not any(e['type'] == 'tool-output-error' for e in events)
