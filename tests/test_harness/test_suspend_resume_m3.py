"""M3: durable suspend/resume + HITL approval (the core deliverable).

Covers: checkpoint round-trip, suspension on an approval-required tool, resume
with approve (executes) and reject (denies), and resume in a *fresh* Agent
instance loading the checkpoint from disk (simulated process restart).
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
from vel.harness import ApprovalDecision
from vel.harness.checkpoint import CheckpointStore, RunCheckpoint
from vel.harness.approvals import ApprovalRequest


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
        return {'done': True}


def _text(t: str) -> List[Any]:
    return [TextStartEvent(block_id='b'), TextDeltaEvent(block_id='b', delta=t),
            TextEndEvent(block_id='b'), FinishMessageEvent(finish_reason='stop')]


def _tool(cid: str, name: str, args: Dict[str, Any]) -> List[Any]:
    return [ToolInputAvailableEvent(tool_call_id=cid, tool_name=name, input=args),
            FinishMessageEvent(finish_reason='tool_calls')]


async def danger(target: str = '', ctx: dict = None) -> dict:
    return {'deleted': target}


async def _collect(agen) -> List[Dict[str, Any]]:
    return [ev async for ev in agen]


def _harness(tmp_path):
    return {
        'enabled': True,
        'db_path': str(tmp_path / 'vel.db'),
        'approval': {'enabled': True, 'mode': 'durable', 'require_for_tools': ['danger']},
    }


def _make_agent(provider, tmp_path):
    agent = Agent(
        id='m3',
        model={'provider': 'scripted', 'model': 'm'},
        tools=[ToolSpec.from_function(danger, name='danger')],
        harness=_harness(tmp_path),
    )
    agent._custom_provider = provider
    return agent


# --------------------------------------------------------------------- checkpoint
def test_checkpoint_store_roundtrip(tmp_path):
    store = CheckpointStore(str(tmp_path / 'vel.db'))
    ckpt = RunCheckpoint(
        run_id='r1', agent_id='a', session_id='s', status='suspended', step=2,
        messages=[{'role': 'user', 'content': 'hi'}],
        pending_approvals=[ApprovalRequest('ap1', 'r1', 'tc1', 'danger', {'target': 'x'})],
        pending_tool_calls=[{'tool_call_id': 'tc1', 'tool_name': 'danger', 'input': {'target': 'x'}}],
        budget_state={'steps': 2}, config_hash='abc',
    )
    store.save(ckpt)
    loaded = store.load('r1')
    assert loaded is not None
    assert loaded.status == 'suspended'
    assert loaded.step == 2
    assert loaded.messages == [{'role': 'user', 'content': 'hi'}]
    assert loaded.pending_tool_calls[0]['tool_name'] == 'danger'
    assert loaded.pending_approvals[0].approval_id == 'ap1'
    assert [c.run_id for c in store.list_suspended('s')] == ['r1']
    store.set_status('r1', 'completed')
    assert store.load('r1').status == 'completed'
    assert store.list_suspended() == []


# --------------------------------------------------------------------- suspension
@pytest.mark.asyncio
async def test_run_suspends_on_approval_required_tool(tmp_path):
    provider = ScriptedProvider([_tool('tc1', 'danger', {'target': 'prod-db'}), _text('done')])
    agent = _make_agent(provider, tmp_path)
    events = await _collect(agent.run_stream({'message': 'delete prod-db'}))
    types = [e['type'] for e in events]

    assert 'data-harness-approval-required' in types
    assert 'data-harness-suspended' in types
    # the tool did NOT execute (no output) — it is gated
    assert not any(e['type'] == 'tool-output-available' for e in events)

    appr = next(e for e in events if e['type'] == 'data-harness-approval-required')
    assert appr['data']['tool_name'] == 'danger'
    assert appr['data']['tool_call_id'] == 'tc1'

    # checkpoint persisted with suspended status
    store = CheckpointStore(str(tmp_path / 'vel.db'))
    suspended = store.list_suspended()
    assert len(suspended) == 1
    assert suspended[0].pending_tool_calls[0]['tool_name'] == 'danger'


# --------------------------------------------------------------------- resume approve
@pytest.mark.asyncio
async def test_resume_approve_executes_tool_and_finishes(tmp_path):
    provider = ScriptedProvider([_tool('tc1', 'danger', {'target': 'prod-db'}), _text('all done')])
    agent = _make_agent(provider, tmp_path)
    suspend_events = await _collect(agent.run_stream({'message': 'delete'}))
    appr = next(e for e in suspend_events if e['type'] == 'data-harness-approval-required')
    run_id = appr['data']['run_id']
    approval_id = appr['data']['approval_id']

    resume_events = await _collect(
        agent.resume(run_id, [ApprovalDecision(approval_id, 'approve')])
    )
    types = [e['type'] for e in resume_events]

    assert types[0] == 'data-harness-resumed'
    # the approved tool actually executed this time
    out = next(e for e in resume_events if e['type'] == 'tool-output-available')
    assert out['output'] == {'deleted': 'prod-db'}
    # continued to a final answer + finished
    assert any(e['type'] == 'text-delta' for e in resume_events)
    assert 'data-harness-run-finished' in types

    # checkpoint marked completed
    store = CheckpointStore(str(tmp_path / 'vel.db'))
    assert store.load(run_id).status == 'completed'


# --------------------------------------------------------------------- resume reject
@pytest.mark.asyncio
async def test_resume_reject_denies_tool_and_continues(tmp_path):
    provider = ScriptedProvider([_tool('tc1', 'danger', {'target': 'prod-db'}), _text('understood')])
    agent = _make_agent(provider, tmp_path)
    suspend_events = await _collect(agent.run_stream({'message': 'delete'}))
    appr = next(e for e in suspend_events if e['type'] == 'data-harness-approval-required')

    resume_events = await _collect(
        agent.resume(appr['data']['run_id'], [ApprovalDecision(appr['data']['approval_id'], 'reject')])
    )
    # denied tool yields an error result, not the real output
    out = next(e for e in resume_events if e['type'] == 'tool-output-available')
    assert 'error' in out['output']
    assert 'denied' in out['output']['error']
    # run still continues to a final answer
    assert any(e['type'] == 'text-delta' for e in resume_events)


# ------------------------------------------------------ resume after process restart
@pytest.mark.asyncio
async def test_resume_in_fresh_agent_after_restart(tmp_path):
    # Run 1: suspend (this agent/process "dies" afterwards).
    provider1 = ScriptedProvider([_tool('tc1', 'danger', {'target': 'prod-db'})])
    agent1 = _make_agent(provider1, tmp_path)
    suspend_events = await _collect(agent1.run_stream({'message': 'delete'}))
    appr = next(e for e in suspend_events if e['type'] == 'data-harness-approval-required')
    run_id = appr['data']['run_id']

    # Run 2: brand-new Agent instance, same config/db — only the checkpoint on
    # disk carries state across the "restart".
    provider2 = ScriptedProvider([_text('done after restart')])
    agent2 = _make_agent(provider2, tmp_path)
    resume_events = await _collect(
        agent2.resume(run_id, [ApprovalDecision(appr['data']['approval_id'], 'approve')])
    )
    out = next(e for e in resume_events if e['type'] == 'tool-output-available')
    assert out['output'] == {'deleted': 'prod-db'}
    assert any(e['type'] == 'text-delta' for e in resume_events)


@pytest.mark.asyncio
async def test_resume_nonexistent_run_raises(tmp_path):
    provider = ScriptedProvider([_text('x')])
    agent = _make_agent(provider, tmp_path)
    with pytest.raises(ValueError):
        await _collect(agent.resume('does-not-exist', []))


# -------------------------------------------------- §6.3.3 per-step checkpoint
@pytest.mark.asyncio
async def test_running_checkpoint_persisted_each_step_then_completed(tmp_path):
    provider = ScriptedProvider([_text('done')])
    agent = _make_agent(provider, tmp_path)
    run_id = None
    async for ev in agent.run_stream({'message': 'hi'}):
        if ev['type'] == 'data-harness-run-started':
            run_id = ev['data']['run_id']
    store = CheckpointStore(str(tmp_path / 'vel.db'))
    ckpt = store.load(run_id)
    assert ckpt is not None              # a checkpoint exists for a normal run
    assert ckpt.status == 'completed'    # and is marked terminal at the end
    assert ckpt.step >= 1


# -------------------------------------------------- §12 Q5 config_hash guard
@pytest.mark.asyncio
async def test_resume_refused_on_config_change_unless_forced(tmp_path):
    provider = ScriptedProvider([_tool('tc1', 'danger', {'target': 'x'}), _text('ok')])
    agent = _make_agent(provider, tmp_path)
    suspend_events = await _collect(agent.run_stream({'message': 'delete'}))
    appr = next(e for e in suspend_events if e['type'] == 'data-harness-approval-required')
    run_id, approval_id = appr['data']['run_id'], appr['data']['approval_id']

    # Tamper with the persisted config_hash to simulate a changed agent/config.
    store = CheckpointStore(str(tmp_path / 'vel.db'))
    ck = store.load(run_id)
    ck.config_hash = 'STALEHASH'
    store.save(ck)

    provider2 = ScriptedProvider([_text('ok')])
    agent2 = _make_agent(provider2, tmp_path)
    with pytest.raises(ValueError):
        await _collect(agent2.resume(run_id, [ApprovalDecision(approval_id, 'approve')]))

    # force=True overrides the guard and resumes.
    provider3 = ScriptedProvider([_text('forced ok')])
    agent3 = _make_agent(provider3, tmp_path)
    events = await _collect(
        agent3.resume(run_id, [ApprovalDecision(approval_id, 'approve')], force=True)
    )
    assert any(e['type'] == 'tool-output-available' for e in events)
