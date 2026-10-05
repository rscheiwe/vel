"""Per-run tool control: tool_choice, ToolSpec(enabled=...), and stop-on-tool history.

- generation_config['tool_choice'] reaches the wire in each provider's format.
- A forcing choice ('required' or a named function) applies until a tool has
  run in the turn, then calls use 'auto' (policies['tool_choice_scope']='run'
  keeps it forced).
- ToolSpec(enabled=...) also applies to tools passed to the agent directly.
- Stopping the loop on a tool keeps the history valid for the next turn.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List

import httpx
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
from vel.providers.anthropic import AnthropicProvider
from vel.providers.openai import OpenAIProvider, OpenAIResponsesProvider

TOOLS = {
    'run_workflow': {'input': {'type': 'object', 'properties': {}}, 'description': 'Run it.'},
    'lookup': {'input': {'type': 'object', 'properties': {}}, 'description': 'Look up.'},
}


# ---------------------------------------------------------------- wire format

class _Stop(Exception):
    pass


async def _captured_payload(provider, method: str, config: Dict[str, Any], tools=TOOLS):
    """Run one provider call against a fake httpx client; return the JSON body."""
    captured: Dict[str, Any] = {}

    class FakeClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def post(self, url, **kwargs):
            captured.update(kwargs.get('json') or {})
            raise _Stop()

        def stream(self, method, url, **kwargs):
            captured.update(kwargs.get('json') or {})
            raise _Stop()

    original = httpx.AsyncClient
    httpx.AsyncClient = lambda *a, **k: FakeClient()
    try:
        call = getattr(provider, method)
        messages = [{'role': 'user', 'content': 'hi'}]
        try:
            if method == 'stream':
                async for _ in call(messages, 'm', tools, config):
                    pass
            else:
                await call(messages, 'm', tools, config)
        except _Stop:
            pass
    finally:
        httpx.AsyncClient = original
    return captured


CHAT_CASES = [
    ({}, 'auto'),
    ({'tool_choice': 'required'}, 'required'),
    ({'tool_choice': 'none'}, 'none'),
    ({'tool_choice': {'type': 'function', 'name': 'run_workflow'}},
     {'type': 'function', 'function': {'name': 'run_workflow'}}),
    ({'tool_choice': {'type': 'function', 'function': {'name': 'run_workflow'}}},
     {'type': 'function', 'function': {'name': 'run_workflow'}}),
]


@pytest.mark.asyncio
@pytest.mark.parametrize('method', ['stream', 'generate'])
@pytest.mark.parametrize('config, expected', CHAT_CASES)
async def test_openai_chat_completions_tool_choice(method, config, expected):
    payload = await _captured_payload(OpenAIProvider(api_key='test'), method, config)

    assert payload['tool_choice'] == expected


@pytest.mark.asyncio
async def test_openai_responses_api_forced_tool_choice_is_flat():
    payload = await _captured_payload(
        OpenAIResponsesProvider(api_key='test'), 'stream',
        {'tool_choice': {'type': 'function', 'name': 'run_workflow'}},
    )

    assert payload['tool_choice'] == {'type': 'function', 'name': 'run_workflow'}


@pytest.mark.asyncio
async def test_no_tools_means_no_tool_choice():
    payload = await _captured_payload(
        OpenAIProvider(api_key='test'), 'generate', {'tool_choice': 'required'}, tools={}
    )

    assert 'tool_choice' not in payload


@pytest.mark.asyncio
@pytest.mark.parametrize('choice, match', [
    ('sometimes', 'tool_choice must be one of'),
    ({'type': 'function', 'name': 'missing_tool'}, 'not among the offered tools'),
])
async def test_invalid_tool_choice_is_rejected(choice, match):
    with pytest.raises(ValueError, match=match):
        await _captured_payload(OpenAIProvider(api_key='test'), 'generate', {'tool_choice': choice})


@pytest.mark.asyncio
@pytest.mark.parametrize('config, expected', [
    ({}, None),
    ({'tool_choice': 'auto'}, {'type': 'auto'}),
    ({'tool_choice': 'required'}, {'type': 'any'}),
    ({'tool_choice': 'none'}, {'type': 'none'}),
    ({'tool_choice': {'type': 'function', 'name': 'run_workflow'}},
     {'type': 'tool', 'name': 'run_workflow'}),
])
async def test_anthropic_tool_choice(config, expected):
    payload = await _captured_payload(AnthropicProvider(api_key='test'), 'stream', config)

    assert payload.get('tool_choice') == expected


# ---------------------------------------------------------------- agent loop

class RecordingProvider(BaseProvider):
    """Scripted provider that records the generation_config of every call."""

    name = 'scripted'

    def __init__(self, script: List[List[Any]]):
        self._script = list(script)
        self.configs: List[Dict[str, Any]] = []
        self.calls: List[List[Dict[str, Any]]] = []

    async def stream(self, messages, model, tools, generation_config=None):
        self.configs.append(dict(generation_config or {}))
        self.calls.append([dict(m) for m in messages])
        for event in self._script.pop(0):
            yield event

    async def generate(self, messages, model, tools, generation_config=None):
        self.configs.append(dict(generation_config or {}))
        self.calls.append([dict(m) for m in messages])
        batch = self._script.pop(0)
        call = next((e for e in batch if isinstance(e, ToolInputAvailableEvent)), None)
        if call is not None:
            return {'tool': call.tool_name, 'args': call.input}
        return {'done': True, 'answer': ''.join(e.delta for e in batch if isinstance(e, TextDeltaEvent))}


def _text(text: str) -> List[Any]:
    return [TextStartEvent(block_id='b'), TextDeltaEvent(block_id='b', delta=text),
            TextEndEvent(block_id='b'), FinishMessageEvent(finish_reason='stop')]


def _tool(call_id: str, name: str) -> List[Any]:
    return [ToolInputAvailableEvent(tool_call_id=call_id, tool_name=name, input={}),
            FinishMessageEvent(finish_reason='tool_calls')]


def run_workflow() -> dict:
    """Run the workflow."""
    return {'status': 'started'}


def _agent(script, **kwargs) -> Agent:
    agent = Agent(id='t', model={'provider': 'scripted', 'model': 'm'}, **kwargs)
    agent._custom_provider = RecordingProvider(script)
    return agent


FORCED = {'tool_choice': {'type': 'function', 'name': 'run_workflow'}}


@pytest.mark.asyncio
async def test_forced_choice_applies_until_a_tool_has_run():
    agent = _agent([_tool('c1', 'run_workflow'), _text('Started.')],
                   tools=[ToolSpec.from_function(run_workflow)], generation_config=FORCED)

    [e async for e in agent.run_stream({'message': 'go'})]

    configs = agent._custom_provider.configs
    assert configs[0]['tool_choice'] == FORCED['tool_choice']
    assert configs[1]['tool_choice'] == 'auto'


@pytest.mark.asyncio
async def test_forced_choice_scope_run_keeps_it_forced():
    agent = _agent([_tool('c1', 'run_workflow'), _tool('c2', 'run_workflow'), _text('Done.')],
                   tools=[ToolSpec.from_function(run_workflow)], generation_config=FORCED,
                   policies={'tool_choice_scope': 'run', 'max_steps': 2})

    [e async for e in agent.run_stream({'message': 'go'})]

    assert [c['tool_choice'] for c in agent._custom_provider.configs[:2]] == [FORCED['tool_choice']] * 2


@pytest.mark.asyncio
async def test_forced_choice_is_scoped_in_non_streaming_run_too():
    agent = _agent([_tool('c1', 'run_workflow'), _text('Started.')],
                   tools=[ToolSpec.from_function(run_workflow)], generation_config=FORCED)

    await agent.run({'message': 'go'})

    assert [c['tool_choice'] for c in agent._custom_provider.configs] == [FORCED['tool_choice'], 'auto']


@pytest.mark.asyncio
async def test_none_is_not_relaxed():
    agent = _agent([_text('Just chatting.')], tools=[ToolSpec.from_function(run_workflow)],
                   generation_config={'tool_choice': 'none'})

    [e async for e in agent.run_stream({'message': 'hi'})]

    assert agent._custom_provider.configs[0]['tool_choice'] == 'none'


# ---------------------------------------------------------------- enabled

def test_enabled_applies_to_tools_passed_directly():
    state = {'ready': False}
    tool = ToolSpec.from_function(run_workflow, enabled=lambda ctx: ctx['state']['ready'])
    agent = Agent(id='t', model={'provider': 'scripted', 'model': 'm'},
                  tools=[tool], tool_context={'state': state})

    assert 'run_workflow' not in agent._get_tool_schemas()
    with pytest.raises(KeyError, match='not enabled'):
        agent._get_tool('run_workflow')

    state['ready'] = True
    assert 'run_workflow' in agent._get_tool_schemas()
    assert agent._get_tool('run_workflow') is tool


# ---------------------------------------------------------------- stop-on-tool history

@pytest.mark.asyncio
async def test_stopping_on_a_tool_keeps_session_history_valid():
    agent = _agent([_tool('c1', 'run_workflow'), _text('Here are the results.')],
                   tools=[ToolSpec.from_function(run_workflow)],
                   policies={'tool_behavior': {'run_workflow': {'stop_on_first_use': True}}})

    first = [e async for e in agent.run_stream({'message': 'go'}, session_id='s')]
    second = [e async for e in agent.run_stream({'message': 'and then?'}, session_id='s')]

    assert first[-1]['type'] == 'finish'
    assert second[-1]['type'] == 'finish'
    second_call = agent._custom_provider.calls[1]
    call_ids = [tc['id'] for m in second_call for tc in (m.get('tool_calls') or [])]
    result_ids = [m['tool_call_id'] for m in second_call if m.get('role') == 'tool']
    assert call_ids == ['c1']
    assert result_ids == ['c1'], second_call
    assert json.loads(next(m['content'] for m in second_call if m.get('role') == 'tool')) == {
        'status': 'started'
    }


@pytest.mark.asyncio
async def test_stopping_on_a_tool_closes_the_steps_other_calls():
    def lookup() -> dict:
        """Look up."""
        return {'found': True}

    batch = [
        ToolInputAvailableEvent(tool_call_id='c1', tool_name='run_workflow', input={}),
        ToolInputAvailableEvent(tool_call_id='c2', tool_name='lookup', input={}),
        FinishMessageEvent(finish_reason='tool_calls'),
    ]
    agent = _agent([batch, _text('Next turn.')],
                   tools=[ToolSpec.from_function(run_workflow), ToolSpec.from_function(lookup)],
                   policies={'tool_behavior': {'run_workflow': {'stop_on_first_use': True}}})

    [e async for e in agent.run_stream({'message': 'go'}, session_id='s')]
    [e async for e in agent.run_stream({'message': 'next'}, session_id='s')]

    tool_messages = [m for m in agent._custom_provider.calls[1] if m.get('role') == 'tool']
    assert [m['tool_call_id'] for m in tool_messages] == ['c1', 'c2']
    assert 'not run' in json.loads(tool_messages[1]['content'])['skipped']
