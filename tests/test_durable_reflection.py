"""Durable reflection: a gated tool called during a thinking phase suspends the
whole reflection run and resumes on the human decision."""
from __future__ import annotations

from typing import Any, List

import pytest

from vel import Agent, ToolSpec
from vel.events import (
    TextStartEvent, TextDeltaEvent, TextEndEvent,
    ToolInputAvailableEvent, FinishMessageEvent,
)
from vel.providers import BaseProvider
from vel.thinking import ThinkingConfig
from vel.harness import ApprovalDecision

LOOKUPS: List[str] = []


class GatedProvider(BaseProvider):
    """Calls the gated `lookup` tool until a tool result is present, then reasons."""
    name = 'fake'

    async def stream(self, messages, model, tools, generation_config=None):
        has_tool_result = any(isinstance(m, dict) and m.get('role') == 'tool' for m in messages)
        if not has_tool_result:
            yield ToolInputAvailableEvent(tool_call_id='call-1', tool_name='lookup', input={'q': 'x'})
            yield FinishMessageEvent(finish_reason='tool_calls')
        else:
            yield TextStartEvent(block_id='b')
            yield TextDeltaEvent(block_id='b', delta='Reasoned answer. Confidence: 95%')
            yield TextEndEvent(block_id='b')
            yield FinishMessageEvent(finish_reason='stop')

    async def generate(self, *a, **k):
        return {}


async def lookup(q: str = '', ctx: dict = None) -> dict:
    LOOKUPS.append(q)
    return {'result': 'data'}


@pytest.fixture(autouse=True)
def _reset():
    LOOKUPS.clear()
    yield
    LOOKUPS.clear()


def _agent(tmp_path):
    agent = Agent(
        id='dr',
        model={'provider': 'fake', 'model': 'm'},
        tools=[ToolSpec.from_function(lookup, name='lookup')],
        thinking=ThinkingConfig(mode='reflection', effort='low', thinking_tools=True,
                                confidence_threshold=0.9),
        harness={
            'enabled': True,
            'db_path': str(tmp_path / 'vel.db'),
            'approval': {'enabled': True, 'mode': 'durable', 'require_for_tools': ['lookup']},
        },
    )
    agent.providers.register(GatedProvider())
    return agent


async def _collect(agen):
    return [e async for e in agen]


@pytest.mark.asyncio
async def test_reflection_suspends_on_gated_tool_and_resumes(tmp_path):
    agent = _agent(tmp_path)

    # Run 1: analyze phase calls the gated `lookup` -> suspends before it runs.
    suspend_events = await _collect(agent.run_stream({'message': 'q'}))
    types = [e['type'] for e in suspend_events]
    assert 'data-harness-approval-required' in types
    assert 'data-harness-suspended' in types
    assert LOOKUPS == []  # gated tool did NOT run yet

    appr = next(e for e in suspend_events if e['type'] == 'data-harness-approval-required')
    run_id, approval_id = appr['data']['run_id'], appr['data']['approval_id']
    assert appr['data']['tool_name'] == 'lookup'

    # Resume with approval: the phase re-runs, the tool executes, reflection finishes.
    resume_events = await _collect(agent.resume(run_id, [ApprovalDecision(approval_id, 'approve')]))
    rtypes = [e['type'] for e in resume_events]
    assert rtypes[0] == 'data-harness-resumed'
    assert LOOKUPS == ['x']  # approved tool ran exactly once
    assert 'tool-output-available' in rtypes
    assert 'data-thinking-complete' in rtypes
    assert any(e['type'] == 'text-delta' for e in resume_events)  # produced the answer
    assert 'finish' in rtypes


@pytest.mark.asyncio
async def test_reflection_reject_denies_tool_and_still_finishes(tmp_path):
    agent = _agent(tmp_path)
    suspend_events = await _collect(agent.run_stream({'message': 'q'}))
    appr = next(e for e in suspend_events if e['type'] == 'data-harness-approval-required')

    resume_events = await _collect(
        agent.resume(appr['data']['run_id'], [ApprovalDecision(appr['data']['approval_id'], 'reject')])
    )
    # Denied tool yields an error result (not the real lookup); run still completes.
    out = next((e for e in resume_events if e['type'] == 'tool-output-available'), None)
    assert out is not None and 'error' in out['output']
    assert LOOKUPS == []  # rejected -> never ran
    assert 'data-thinking-complete' in [e['type'] for e in resume_events]
