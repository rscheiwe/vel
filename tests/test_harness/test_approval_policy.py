"""Feature 1: approval-as-predicate + approve-once-per-session memory.

Covers ApprovalGate.evaluate precedence, the input-aware policy predicate
(auto-approve / auto-deny / prompt), and session memory across runs.
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
from vel.harness import ApprovalConfig, ApprovalContext, ApprovalDecision, ApprovalGate
from vel.harness.approvals import normalize_approval_status


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


async def delete_file(path: str = '', ctx: dict = None) -> dict:
    return {'deleted': path}


async def _collect(agen) -> List[Dict[str, Any]]:
    return [ev async for ev in agen]


def _make_agent(provider, tmp_path, approval: Dict[str, Any]):
    agent = Agent(
        id='pol',
        model={'provider': 'scripted', 'model': 'm'},
        tools=[ToolSpec.from_function(delete_file, name='delete_file')],
        harness={
            'enabled': True,
            'db_path': str(tmp_path / 'vel.db'),
            'approval': {'enabled': True, 'mode': 'durable', **approval},
        },
    )
    agent._custom_provider = provider
    return agent


# ------------------------------------------------------------- unit: normalize
def test_normalize_status():
    assert normalize_approval_status(True) == 'user-approval'
    assert normalize_approval_status(False) == 'not-applicable'
    assert normalize_approval_status(None) == 'not-applicable'
    for s in ('approved', 'denied', 'user-approval', 'not-applicable'):
        assert normalize_approval_status(s) == s
    with pytest.raises(ValueError):
        normalize_approval_status('nope')


# ------------------------------------------------------------- unit: evaluate
def _ctx(tool_name='delete_file', tool_input=None, approved=()):
    return ApprovalContext(
        tool_name=tool_name,
        tool_input=tool_input or {},
        tool_call_id='tc',
        run_id='r',
        session_id='s',
        step=1,
        approved_tools=frozenset(approved),
        requires_confirmation=False,
    )


def test_evaluate_precedence_memory_then_policy_then_static():
    # session memory wins
    gate = ApprovalGate(ApprovalConfig(policy=lambda c: 'user-approval'))
    assert gate.evaluate(_ctx(approved={'delete_file'})) == 'not-applicable'

    # policy decisive when memory empty
    gate = ApprovalGate(ApprovalConfig(policy=lambda c: 'denied'))
    assert gate.evaluate(_ctx()) == 'denied'

    # policy abstains (None) -> fall through to static require_for_tools
    gate = ApprovalGate(ApprovalConfig(policy=lambda c: None, require_for_tools=['delete_file']))
    assert gate.evaluate(_ctx()) == 'user-approval'

    # no policy, not gated -> not-applicable
    gate = ApprovalGate(ApprovalConfig())
    assert gate.evaluate(_ctx()) == 'not-applicable'


def test_evaluate_is_input_aware():
    def policy(ctx: ApprovalContext):
        if ctx.tool_input.get('path', '').startswith('/prod'):
            return 'denied'
        return 'approved'
    gate = ApprovalGate(ApprovalConfig(policy=policy))
    assert gate.evaluate(_ctx(tool_input={'path': '/prod/db'})) == 'denied'
    assert gate.evaluate(_ctx(tool_input={'path': '/tmp/x'})) == 'approved'


# ------------------------------------------------------- integration: auto-approve
@pytest.mark.asyncio
async def test_policy_auto_approve_runs_without_suspend(tmp_path):
    provider = ScriptedProvider([_tool('tc1', 'delete_file', {'path': '/tmp/x'}), _text('ok')])
    agent = _make_agent(provider, tmp_path, {'policy': lambda c: 'approved'})
    events = await _collect(agent.run_stream({'message': 'go'}))
    types = [e['type'] for e in events]
    assert 'data-harness-suspended' not in types
    out = next(e for e in events if e['type'] == 'tool-output-available')
    assert out['output'] == {'deleted': '/tmp/x'}


# --------------------------------------------------------- integration: auto-deny
@pytest.mark.asyncio
async def test_policy_auto_deny_blocks_without_suspend(tmp_path):
    provider = ScriptedProvider([_tool('tc1', 'delete_file', {'path': '/prod/db'}), _text('understood')])
    agent = _make_agent(provider, tmp_path, {'policy': lambda c: 'denied'})
    events = await _collect(agent.run_stream({'message': 'go'}))
    types = [e['type'] for e in events]
    assert 'data-harness-suspended' not in types
    out = next(e for e in events if e['type'] == 'tool-output-available')
    assert 'error' in out['output'] and 'denied' in out['output']['error']


# ------------------------------------------------------- integration: prompt path
@pytest.mark.asyncio
async def test_policy_user_approval_suspends(tmp_path):
    provider = ScriptedProvider([_tool('tc1', 'delete_file', {'path': '/prod/db'}), _text('ok')])
    agent = _make_agent(provider, tmp_path, {'policy': lambda c: 'user-approval'})
    events = await _collect(agent.run_stream({'message': 'go'}))
    assert 'data-harness-suspended' in [e['type'] for e in events]


# --------------------------------------------------- integration: session memory
@pytest.mark.asyncio
async def test_remember_approvals_auto_approves_second_call(tmp_path):
    # Run 1: gated tool suspends, human approves (records session memory).
    p1 = ScriptedProvider([_tool('tc1', 'delete_file', {'path': '/a'}), _text('done')])
    agent = _make_agent(p1, tmp_path, {'require_for_tools': ['delete_file']})
    s1 = await _collect(agent.run_stream({'message': 'go'}, session_id='sess'))
    appr = next(e for e in s1 if e['type'] == 'data-harness-approval-required')
    await _collect(agent.resume(appr['data']['run_id'], [ApprovalDecision(appr['data']['approval_id'], 'approve')]))

    # Run 2: same session + tool -> auto-approved from memory, no suspend.
    p2 = ScriptedProvider([_tool('tc2', 'delete_file', {'path': '/b'}), _text('done')])
    agent._custom_provider = p2
    s2 = await _collect(agent.run_stream({'message': 'again'}, session_id='sess'))
    types = [e['type'] for e in s2]
    assert 'data-harness-suspended' not in types
    out = next(e for e in s2 if e['type'] == 'tool-output-available')
    assert out['output'] == {'deleted': '/b'}


@pytest.mark.asyncio
async def test_remember_disabled_reprompts(tmp_path):
    p1 = ScriptedProvider([_tool('tc1', 'delete_file', {'path': '/a'}), _text('done')])
    agent = _make_agent(p1, tmp_path, {'require_for_tools': ['delete_file'], 'remember_approvals': False})
    s1 = await _collect(agent.run_stream({'message': 'go'}, session_id='sess'))
    appr = next(e for e in s1 if e['type'] == 'data-harness-approval-required')
    await _collect(agent.resume(appr['data']['run_id'], [ApprovalDecision(appr['data']['approval_id'], 'approve')]))

    p2 = ScriptedProvider([_tool('tc2', 'delete_file', {'path': '/b'}), _text('done')])
    agent._custom_provider = p2
    s2 = await _collect(agent.run_stream({'message': 'again'}, session_id='sess'))
    assert 'data-harness-suspended' in [e['type'] for e in s2]
