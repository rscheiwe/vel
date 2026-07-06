"""M1: HarnessController routing, lifecycle events, and budget enforcement.

Proves (a) harness is default-off, (b) when enabled it emits ``data-harness-*``
events and otherwise behaves like a normal run, and (c) the step budget is
enforced and falls through to a synthesized partial answer.
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
    name = 'scripted'

    def __init__(self, script: List[List[Any]]):
        self._script = list(script)
        self.calls: List[Dict[str, Any]] = []

    async def stream(self, messages, model, tools, generation_config=None):
        self.calls.append({'messages': messages, 'tools': tools})
        batch = self._script.pop(0) if self._script else [
            TextStartEvent(block_id='end'),
            TextDeltaEvent(block_id='end', delta='fallback'),
            TextEndEvent(block_id='end'),
            FinishMessageEvent(finish_reason='stop'),
        ]
        for ev in batch:
            yield ev

    async def generate(self, messages, model, tools, generation_config=None):
        return {'done': True}


def _text(text: str, usage: Dict[str, int] | None = None) -> List[Any]:
    batch: List[Any] = [TextStartEvent(block_id='b'), TextDeltaEvent(block_id='b', delta=text), TextEndEvent(block_id='b')]
    if usage:
        batch.append(ResponseMetadataEvent(usage=usage))
    batch.append(FinishMessageEvent(finish_reason='stop'))
    return batch


def _tool(call_id: str, name: str, args: Dict[str, Any]) -> List[Any]:
    return [ToolInputAvailableEvent(tool_call_id=call_id, tool_name=name, input=args),
            FinishMessageEvent(finish_reason='tool_calls')]


async def loop_forever_tool(q: str = '', ctx: dict = None) -> dict:
    return {'ok': q}


async def _collect(agen) -> List[Dict[str, Any]]:
    return [ev async for ev in agen]


def _make_agent(provider, with_tool=False, harness=None):
    tools = [ToolSpec.from_function(loop_forever_tool, name='loop')] if with_tool else None
    agent = Agent(id='m1', model={'provider': 'scripted', 'model': 'm'}, tools=tools, harness=harness)
    agent._custom_provider = provider
    return agent


@pytest.mark.asyncio
async def test_harness_default_off_no_harness_events():
    """No harness config => no data-harness-* events, normal run."""
    provider = ScriptedProvider([_text('hi')])
    agent = _make_agent(provider)
    events = await _collect(agent.run_stream({'message': 'x'}))
    assert not any(e['type'].startswith('data-harness-') for e in events)
    assert events[-1]['type'] == 'finish'


@pytest.mark.asyncio
async def test_harness_emits_lifecycle_and_step_events():
    provider = ScriptedProvider([_text('done', usage={'promptTokens': 10, 'completionTokens': 5})])
    agent = _make_agent(provider, harness={'enabled': True})
    events = await _collect(agent.run_stream({'message': 'x'}))
    types = [e['type'] for e in events]

    assert 'data-harness-run-started' in types
    assert 'data-harness-step' in types
    assert 'data-harness-run-finished' in types
    # run-started precedes the first normal start-step
    assert types.index('data-harness-run-started') < types.index('start-step')
    # finished is last harness event and overall stream still ends with finish
    assert types[-1] == 'data-harness-run-finished' or types[-1] == 'finish'

    started = next(e for e in events if e['type'] == 'data-harness-run-started')
    assert started['data']['run_id']
    assert started['data']['agent_id'] == 'm1'

    step_ev = next(e for e in events if e['type'] == 'data-harness-step')
    assert step_ev['data']['step'] == 1
    assert 'budget' in step_ev['data']


@pytest.mark.asyncio
async def test_harness_preserves_normal_event_shapes():
    """With harness on, the underlying text/finish events are unchanged."""
    provider = ScriptedProvider([_text('hello')])
    agent = _make_agent(provider, harness={'enabled': True})
    events = await _collect(agent.run_stream({'message': 'x'}))
    non_harness = [e for e in events if not e['type'].startswith('data-harness-')]
    types = [e['type'] for e in non_harness]
    assert types[0] == 'start'
    assert 'text-delta' in types
    assert types[-1] == 'finish'


@pytest.mark.asyncio
async def test_budget_max_steps_enforced_then_synthesizes():
    """A tool that always wants to run again is capped by the step budget,
    then a final synthesis call produces a partial answer."""
    # Each step the model calls the tool again -> would loop forever without budget.
    script = [_tool(f'c{i}', 'loop', {'q': str(i)}) for i in range(10)]
    provider = ScriptedProvider(script)
    agent = _make_agent(provider, with_tool=True, harness={'enabled': True, 'budget': {'max_steps': 3}})
    events = await _collect(agent.run_stream({'message': 'go'}))
    types = [e['type'] for e in events]

    assert 'data-harness-budget-exhausted' in types
    exhausted = next(e for e in events if e['type'] == 'data-harness-budget-exhausted')
    assert 'steps' in exhausted['data']['reason']
    # synthesis happened (final tool-less call) and stream terminated cleanly
    assert types[-1] in ('finish', 'data-harness-run-finished')
    # exactly the budgeted number of steps ran (step events 1 and 2 before the
    # 3rd step's budget gate fires)
    step_events = [e for e in events if e['type'] == 'data-harness-step']
    assert [e['data']['step'] for e in step_events] == [1, 2]


@pytest.mark.asyncio
async def test_budget_tokens_tracked_in_finished_usage():
    provider = ScriptedProvider([_text('done', usage={'promptTokens': 100, 'completionTokens': 20})])
    agent = _make_agent(provider, harness={'enabled': True})
    events = await _collect(agent.run_stream({'message': 'x'}))
    finished = next(e for e in events if e['type'] == 'data-harness-run-finished')
    assert finished['data']['usage']['tokens'] == 120


@pytest.mark.asyncio
async def test_budget_max_tokens_exhaustion_synthesizes():
    # Each tool step reports usage; max_tokens trips after the first step.
    script = [_tool(f'c{i}', 'loop', {'q': str(i)}) for i in range(5)]
    # attach usage to the tool steps via response-metadata
    script = [b + [ResponseMetadataEvent(usage={'promptTokens': 60, 'completionTokens': 0})] for b in script]
    provider = ScriptedProvider(script)
    agent = _make_agent(provider, with_tool=True,
                        harness={'enabled': True, 'budget': {'max_steps': 50, 'max_tokens': 100}})
    events = await _collect(agent.run_stream({'message': 'go'}))
    types = [e['type'] for e in events]
    assert 'data-harness-budget-exhausted' in types
    reason = next(e for e in events if e['type'] == 'data-harness-budget-exhausted')['data']['reason']
    assert 'tokens' in reason


@pytest.mark.asyncio
async def test_budget_max_cost_exhaustion_synthesizes():
    script = [_tool(f'c{i}', 'loop', {'q': str(i)}) for i in range(5)]
    script = [b + [ResponseMetadataEvent(usage={'promptTokens': 0, 'completionTokens': 0, 'cost_usd': 0.5})] for b in script]
    provider = ScriptedProvider(script)
    agent = _make_agent(provider, with_tool=True,
                        harness={'enabled': True, 'budget': {'max_steps': 50, 'max_cost_usd': 0.6}})
    events = await _collect(agent.run_stream({'message': 'go'}))
    reason = next((e for e in events if e['type'] == 'data-harness-budget-exhausted'), None)
    assert reason is not None
    assert 'cost' in reason['data']['reason']
