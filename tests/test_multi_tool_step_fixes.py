"""Two defects that a multi-tool step already had, before parallelism.

Both were reachable whenever a model asked for more than one tool in a single
step. Opt-in parallel execution does not cause either — it makes multi-tool
steps common enough that they stop being theoretical.
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
from vel.core.tool_behavior import ToolUseBehavior
from vel.providers import BaseProvider
from vel.providers.message_translator import translate_to_anthropic


# ---------------------------------------------------------------------------
# 1. Anthropic requires every tool_result for one assistant turn in ONE message
# ---------------------------------------------------------------------------

def test_multiple_tool_results_merge_into_one_user_message():
    """One message per result is a malformed Anthropic request.

    The API pairs `tool_result` blocks with the preceding assistant turn's
    `tool_use` blocks; splitting them across messages is rejected.
    """
    messages = [
        {'role': 'user', 'content': 'do both'},
        {'role': 'assistant', 'content': None, 'tool_calls': [
            {'id': 'c1', 'type': 'function', 'function': {'name': 'a', 'arguments': '{}'}},
            {'id': 'c2', 'type': 'function', 'function': {'name': 'b', 'arguments': '{}'}},
        ]},
        {'role': 'tool', 'tool_call_id': 'c1', 'content': 'first'},
        {'role': 'tool', 'tool_call_id': 'c2', 'content': 'second'},
    ]

    _system, out = translate_to_anthropic(messages)
    result_messages = [
        m for m in out
        if m['role'] == 'user'
        and isinstance(m['content'], list)
        and any(b.get('type') == 'tool_result' for b in m['content'])
    ]

    assert len(result_messages) == 1, 'tool results must not be split across messages'
    ids = [b['tool_use_id'] for b in result_messages[0]['content']]
    assert ids == ['c1', 'c2'], 'both results present, in order'


def test_a_single_tool_result_is_unchanged():
    messages = [
        {'role': 'assistant', 'content': None, 'tool_calls': [
            {'id': 'c1', 'type': 'function', 'function': {'name': 'a', 'arguments': '{}'}},
        ]},
        {'role': 'tool', 'tool_call_id': 'c1', 'content': 'only'},
    ]
    _system, out = translate_to_anthropic(messages)
    blocks = [b for m in out if isinstance(m.get('content'), list)
              for b in m['content'] if b.get('type') == 'tool_result']
    assert len(blocks) == 1


def test_results_separated_by_another_message_are_not_merged():
    """Merging is only correct for results that belong to the same turn."""
    messages = [
        {'role': 'tool', 'tool_call_id': 'c1', 'content': 'first'},
        {'role': 'user', 'content': 'interruption'},
        {'role': 'tool', 'tool_call_id': 'c2', 'content': 'second'},
    ]
    _system, out = translate_to_anthropic(messages)
    result_messages = [
        m for m in out
        if isinstance(m.get('content'), list)
        and any(b.get('type') == 'tool_result' for b in m['content'])
    ]
    assert len(result_messages) == 2


# ---------------------------------------------------------------------------
# 2. stop_on_first_tool must actually stop the loop
# ---------------------------------------------------------------------------

class CountingProvider(BaseProvider):
    """Counts model calls, so an extra one after `finish` is visible."""

    name = 'scripted'

    def __init__(self) -> None:
        self.calls = 0

    async def stream(self, messages, model, tools, generation_config=None):
        self.calls += 1
        if self.calls == 1:
            yield ToolInputAvailableEvent(tool_call_id='c1', tool_name='echo', input={'x': 'a'})
            yield FinishMessageEvent(finish_reason='tool_calls')
            return
        yield TextStartEvent(block_id='t')
        yield TextDeltaEvent(block_id='t', delta='should not happen')
        yield TextEndEvent(block_id='t')
        yield FinishMessageEvent(finish_reason='stop')

    async def generate(self, messages, model, tools, generation_config=None):
        return {}


async def echo(x: str = '', ctx: dict = None) -> dict:
    return {'echo': x}


@pytest.mark.asyncio
async def test_stop_on_first_tool_does_not_make_another_model_call():
    """It emitted `finish` and then kept going.

    The branch returned without setting `loop_state['control']`, so `_step_loop`
    read it as None, fell through to `continue`, and issued another LLM call
    after the terminal event had already been sent.
    """
    provider = CountingProvider()
    agent = Agent(
        id='stop',
        model={'provider': 'scripted', 'model': 'm'},
        tools=[ToolSpec.from_function(echo, name='echo')],
        policies={'max_steps': 24, 'tool_use_behavior': ToolUseBehavior.STOP_AFTER_TOOL},
    )
    agent._custom_provider = provider

    events = [e async for e in agent.run_stream({'message': 'hi'})]

    assert provider.calls == 1, f'made {provider.calls} model calls; the loop continued past finish'
    assert events[-1]['type'] == 'finish'
    assert events.count({'type': 'finish'}) == 1
    assert not any(e['type'] == 'text-delta' for e in events)
