"""Regression tests for vel behaviour found while building mesh on top of vel.

1. Run-scoped system messages (instruction, output_type schema) must not
   accumulate in session history.
2. Structured-output runs must not stream their raw JSON as text by default.
3. Ending on an error must close the open step.
4. Tool results must reach the model as JSON, not a Python repr.
5. ``ToolSpec.from_function`` tools must receive ``ctx`` when they declare it.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

import pytest
from pydantic import BaseModel

from vel import Agent, ToolSpec
from vel.core.structured_output import StructuredOutputPolicy
from vel.events import (
    ErrorEvent,
    FinishMessageEvent,
    TextDeltaEvent,
    TextEndEvent,
    TextStartEvent,
    ToolInputAvailableEvent,
)
from vel.providers import BaseProvider


class ScriptedProvider(BaseProvider):
    """Replays one batch of provider events per call; snapshots what it was sent."""

    name = 'scripted'

    def __init__(self, script: List[List[Any]]):
        self._script = list(script)
        self.calls: List[List[Dict[str, Any]]] = []

    async def stream(self, messages, model, tools, generation_config=None):
        self.calls.append([dict(m) for m in messages])
        for event in self._script.pop(0):
            if isinstance(event, Exception):
                raise event
            yield event

    async def generate(self, messages, model, tools, generation_config=None):
        """Non-streaming calls replay the same script: text turns become answers."""
        self.calls.append([dict(m) for m in messages])
        batch = self._script.pop(0)
        text = ''.join(e.delta for e in batch if isinstance(e, TextDeltaEvent))
        return {'done': True, 'answer': text}


def _text(text: str) -> List[Any]:
    return [TextStartEvent(block_id='b'), TextDeltaEvent(block_id='b', delta=text),
            TextEndEvent(block_id='b'), FinishMessageEvent(finish_reason='stop')]


def _tool(call_id: str, name: str, args: Dict[str, Any]) -> List[Any]:
    return [ToolInputAvailableEvent(tool_call_id=call_id, tool_name=name, input=args),
            FinishMessageEvent(finish_reason='tool_calls')]


def _agent(script, **kwargs) -> Agent:
    agent = Agent(id='t', model={'provider': 'scripted', 'model': 'm'}, **kwargs)
    agent._custom_provider = ScriptedProvider(script)
    return agent


async def _collect(agen) -> List[Dict[str, Any]]:
    return [event async for event in agen]


class Slots(BaseModel):
    vertical: Optional[str] = None
    budget: Optional[float] = None


SLOTS_JSON = json.dumps({'vertical': 'Retail', 'budget': 30000})


# 1. Run-scoped system messages ---------------------------------------------

@pytest.mark.asyncio
async def test_instruction_is_sent_once_per_call_in_a_session():
    """Also covers run(): the non-streaming path now sends system_prompt too."""
    agent = _agent([_text('one'), _text('two'), _text('three')], instruction='Be brief.')

    for turn in ('hi', 'again', 'third'):
        await _collect(agent.run_stream({'message': turn}, session_id='s'))

    for call in agent._custom_provider.calls:
        assert [m['content'] for m in call if m['role'] == 'system'] == ['Be brief.']
    history = agent.ctxmgr.get_session_context('s')
    assert [m['role'] for m in history] == ['user', 'assistant'] * 3


@pytest.mark.asyncio
async def test_schema_prompt_is_sent_once_per_call_in_a_session():
    agent = _agent([_text(SLOTS_JSON), _text(SLOTS_JSON)], output_type=Slots,
                   instruction='Extract slots.')

    for turn in ('brief one', 'brief two'):
        await agent.run({'message': turn}, session_id='s')

    second = agent._custom_provider.calls[1]
    systems = [m['content'] for m in second if m['role'] == 'system']
    assert len(systems) == 2  # one schema prompt, one instruction
    assert systems[1] == 'Extract slots.'
    assert all(m['role'] != 'system' for m in agent.ctxmgr.get_session_context('s'))


# 2. Structured-output text --------------------------------------------------

@pytest.mark.asyncio
async def test_structured_output_stream_hides_json_text_by_default():
    agent = _agent([_text(SLOTS_JSON)], output_type=Slots)

    events = await _collect(agent.run_stream({'message': 'brief'}))

    types = [e['type'] for e in events]
    assert not [t for t in types if t.startswith('text-')]
    complete = [e for e in events if e['type'] == 'data-object-complete']
    assert complete[0]['data']['object'] == {'vertical': 'Retail', 'budget': 30000.0}


@pytest.mark.asyncio
async def test_structured_output_text_can_be_streamed_on_request():
    agent = _agent([_text(SLOTS_JSON)], output_type=Slots,
                   structured_output_policy=StructuredOutputPolicy(stream_text=True))

    events = await _collect(agent.run_stream({'message': 'brief'}))

    assert [e['delta'] for e in events if e['type'] == 'text-delta'] == [SLOTS_JSON]
    assert any(e['type'] == 'data-object-complete' for e in events)


# 3. Errors close the step ---------------------------------------------------

def _assert_step_closed_before_error(events):
    types = [e['type'] for e in events]
    error_at = types.index('error')
    before = types[:error_at]
    assert before.count('start-step') == before.count('finish-step'), types


@pytest.mark.xfail(strict=True, reason='pending fix 3')
@pytest.mark.asyncio
async def test_provider_exception_closes_the_step_then_raises():
    agent = _agent([[RuntimeError('provider returned 503')]])
    events = []

    with pytest.raises(RuntimeError, match='503'):
        async for event in agent.run_stream({'message': 'hi'}):
            events.append(event)

    _assert_step_closed_before_error(events)


@pytest.mark.xfail(strict=True, reason='pending fix 3')
@pytest.mark.asyncio
async def test_provider_error_event_closes_the_step():
    agent = _agent([[ErrorEvent(error='rate limited')]])

    events = await _collect(agent.run_stream({'message': 'hi'}))

    _assert_step_closed_before_error(events)
    assert events[-1]['type'] == 'finish'


# 4. JSON tool results -------------------------------------------------------

@pytest.mark.xfail(strict=True, reason='pending fix 4')
@pytest.mark.asyncio
async def test_tool_results_reach_the_model_as_json():
    async def lookup(product_id: str) -> dict:
        """Look up a product."""
        return {'product_id': product_id, 'cpm': 15.0, 'in_plan': True, 'tags': ['ctr']}

    agent = _agent([_tool('c1', 'lookup', {'product_id': 'P003'}), _text('done')],
                   tools=[ToolSpec.from_function(lookup)])

    await _collect(agent.run_stream({'message': 'look up P003'}))

    tool_message = next(m for m in agent._custom_provider.calls[1] if m['role'] == 'tool')
    assert json.loads(tool_message['content']) == {
        'product_id': 'P003', 'cpm': 15.0, 'in_plan': True, 'tags': ['ctr'],
    }


@pytest.mark.xfail(strict=True, reason='pending fix 4')
def test_non_json_tool_results_fall_back_to_str():
    from vel.core.context import ContextManager

    ctx = ContextManager()
    ctx.set_input('r', {'message': 'hi'})
    marker = object()
    ctx.append_tool_result('r', 'tool', {'value': marker}, tool_call_id='c1')

    content = ctx.messages_for_llm('r')[-1]['content']
    assert json.loads(content) == {'value': str(marker)}


# 5. from_function ctx -------------------------------------------------------

@pytest.mark.xfail(strict=True, reason='pending fix 5')
@pytest.mark.asyncio
@pytest.mark.parametrize('param', ['ctx', 'context', '_context'])
async def test_from_function_tools_receive_ctx(param):
    seen: Dict[str, Any] = {}
    namespace: Dict[str, Any] = {'seen': seen}
    exec(
        f'async def inspect_ctx(product_id: str, {param}: dict) -> dict:\n'
        f'    """Inspect ctx."""\n'
        f'    seen["ctx"] = {param}\n'
        f'    return {{"ok": True}}\n',
        namespace,
    )
    tool = ToolSpec.from_function(namespace['inspect_ctx'])
    agent = _agent([_tool('c1', 'inspect_ctx', {'product_id': 'P1'}), _text('done')],
                   tools=[tool], tool_context={'state': {'budget': 30000}})

    await _collect(agent.run_stream({'message': 'go'}))

    assert seen['ctx'] == {'state': {'budget': 30000}}
    assert param not in tool.input_schema.get('properties', {})


@pytest.mark.xfail(strict=True, reason='pending fix 5')
@pytest.mark.asyncio
async def test_from_function_ctx_with_default_none_is_filled():
    seen: Dict[str, Any] = {}

    def lookup(product_id: str, ctx: dict = None) -> dict:
        """Look up."""
        seen['ctx'] = ctx
        return {}

    tool = ToolSpec.from_function(lookup)

    await tool.run({'product_id': 'P1'}, {'user': 'u1'})

    assert seen['ctx'] == {'user': 'u1'}


@pytest.mark.asyncio
async def test_run_sends_the_system_prompt():
    """run() used to call the provider without system_prompt; run_stream() sent it."""
    agent = _agent([_text('ok')], system_prompt="You are Kargo's media assistant.")

    await agent.run({'message': 'hi'})

    first = agent._custom_provider.calls[0]
    assert first[0] == {'role': 'system', 'content': "You are Kargo's media assistant."}
