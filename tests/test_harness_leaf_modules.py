import pytest

from vel.harness.approvals import ApprovalDecision, ApprovalGate, SQLiteApprovalStore
from vel.harness.budget import BudgetExhausted, HarnessBudget
from vel.harness.config import ApprovalConfig, HarnessBudgetConfig, HarnessConfig, SkillRef
from vel.harness.events import (
    HarnessApprovalRequiredEvent,
    HarnessBudgetExhaustedEvent,
    HarnessCompactionEvent,
    HarnessResumedEvent,
    HarnessRunFinishedEvent,
    HarnessRunStartedEvent,
    HarnessSandboxEvent,
    HarnessStepEvent,
    HarnessSuspendedEvent,
)
from vel.harness.skills import Skill, SkillRegistry, resolve_skills


def test_harness_config_defaults_match_spec():
    config = HarnessConfig()

    assert config.enabled is False
    assert config.durable is True
    assert config.compaction.enabled is True
    assert config.approval.mode == 'durable'
    assert config.sandbox.provider == 'none'
    assert config.budget.max_steps == 100
    assert config.store_backend is None
    assert config.db_path == '.vel/vel.db'


def test_harness_config_coerces_nested_dicts():
    config = HarnessConfig(
        **{
            'enabled': True,
            'compaction': {'trigger_token_ratio': 0.5},
            'approval': {'require_for_tools': ['deploy']},
            'sandbox': {'enabled': True, 'tools': ['bash']},
            'budget': {'max_steps': 3},
            'skills': [{'name': 'writer'}],
        }
    )

    assert config.enabled is True
    assert config.compaction.trigger_token_ratio == 0.5
    assert config.approval.require_for_tools == ['deploy']
    assert config.sandbox.tools == ['bash']
    assert config.budget.max_steps == 3
    assert config.skills[0].name == 'writer'


def test_harness_events_emit_data_shapes_and_transience():
    assert HarnessRunStartedEvent(run_id='r1', agent_id='a1', durable=True).to_dict() == {
        'type': 'data-harness-run-started',
        'data': {'run_id': 'r1', 'agent_id': 'a1', 'durable': True},
        'transient': True,
    }
    assert HarnessStepEvent(step=2, budget={'steps': 2, 'tokens': 10, 'cost': 0.0}).to_dict() == {
        'type': 'data-harness-step',
        'data': {'step': 2, 'budget': {'steps': 2, 'tokens': 10, 'cost': 0.0}},
        'transient': True,
    }
    assert HarnessCompactionEvent(
        before_tokens=100,
        after_tokens=40,
        strategy='summarize',
        removed=3,
    ).to_dict()['transient'] is False
    assert HarnessBudgetExhaustedEvent(reason='steps exhausted').to_dict() == {
        'type': 'data-harness-budget-exhausted',
        'data': {'reason': 'steps exhausted'},
        'transient': True,
    }


def test_all_harness_event_shapes():
    events = [
        HarnessRunStartedEvent(run_id='r', agent_id='a', durable=True).to_dict(),
        HarnessStepEvent(step=1, budget={'steps': 1, 'tokens': 0, 'cost': 0.0}).to_dict(),
        HarnessCompactionEvent(
            before_tokens=100,
            after_tokens=50,
            strategy='reduce',
            removed=3,
        ).to_dict(),
        HarnessApprovalRequiredEvent(
            approval_id='ap',
            run_id='r',
            tool_call_id='tc',
            tool_name='tool',
            reason='why',
        ).to_dict(),
        HarnessSuspendedEvent(run_id='r').to_dict(),
        HarnessResumedEvent(run_id='r').to_dict(),
        HarnessBudgetExhaustedEvent(reason='steps').to_dict(),
        HarnessSandboxEvent(event='created', sandbox_ref='sbx').to_dict(),
        HarnessRunFinishedEvent(run_id='r', status='completed', usage={'tokens': 1}).to_dict(),
    ]

    assert [event['type'] for event in events] == [
        'data-harness-run-started',
        'data-harness-step',
        'data-harness-compaction',
        'data-harness-approval-required',
        'data-harness-suspended',
        'data-harness-resumed',
        'data-harness-budget-exhausted',
        'data-harness-sandbox',
        'data-harness-run-finished',
    ]
    assert [event.get('transient', False) for event in events] == [
        True, True, False, False, False, True, True, True, False
    ]


def test_harness_budget_tracks_usage_and_raises():
    budget = HarnessBudget.from_config(HarnessBudgetConfig(max_steps=2, max_tokens=10))

    budget.bump_step()
    budget.bump({'usage': {'promptTokens': 3, 'completionTokens': 4}})

    assert budget.to_event_budget() == {'steps': 1, 'tokens': 7, 'cost': 0.0}
    budget.check()

    budget.bump_tokens(prompt=3)
    with pytest.raises(BudgetExhausted) as exc:
        budget.check()
    assert exc.value.reason == 'tokens exhausted (10/10)'


def test_harness_budget_restores_checkpoint_state():
    budget = HarnessBudget.from_config(
        HarnessBudgetConfig(max_steps=5, max_cost_usd=1.0),
        restore={
            'started_at': 123.0,
            'steps': 2,
            'prompt_tokens': 11,
            'completion_tokens': 13,
            'cost_usd': 0.25,
        },
    )

    assert budget.steps == 2
    assert budget.tokens == 24
    assert budget.cost_usd == 0.25
    assert budget.to_dict()['max_steps'] == 5


@pytest.mark.asyncio
async def test_approval_gate_requires_opens_and_records():
    class Tool:
        requires_confirmation = True

    gate = ApprovalGate(ApprovalConfig(require_for_tools=['deploy']))

    assert gate.requires_approval(object(), 'deploy') is True
    assert gate.requires_approval(Tool(), 'write') is True
    assert gate.requires_approval(object(), 'read') is False

    request = gate.build_request(
        run_id='r1',
        tool_call_id='tc1',
        tool_name='deploy',
        args={'service': 'api'},
        approval_id='a1',
    )
    opened = await gate.open([request])
    assert opened == [request]
    assert await gate.get_pending('r1') == [request]

    await gate.record(ApprovalDecision(approval_id='a1', decision='approve'))
    assert await gate.get_pending('r1') == []
    assert (await gate.get_decision('tc1')).decision == 'approve'


@pytest.mark.asyncio
async def test_sqlite_approval_store_persists_pending_and_decisions(tmp_path):
    store = SQLiteApprovalStore(str(tmp_path / 'vel.db'))
    gate = ApprovalGate(store=store)
    request = gate.build_request(
        run_id='r1',
        tool_call_id='tc1',
        tool_name='deploy',
        args={'service': 'api'},
        reason='needs deploy approval',
        approval_id='a1',
    )

    await gate.open([request])
    pending = await gate.get_pending('r1')
    assert pending == [request]
    assert await gate.get_decision('tc1') is None

    await gate.record(
        ApprovalDecision(
            approval_id='a1',
            decision='reject',
            note='not now',
            decided_by='tester',
            decided_at=123.0,
        )
    )

    assert await gate.get_pending('r1') == []
    decision = await gate.get_decision('tc1')
    assert decision.approval_id == 'a1'
    assert decision.decision == 'reject'
    assert decision.note == 'not now'
    assert decision.decided_by == 'tester'
    assert decision.decided_at == 123.0
    store.close()


@pytest.mark.asyncio
async def test_approval_timeout_and_idempotent_decisions(tmp_path):
    store = SQLiteApprovalStore(str(tmp_path / 'vel.db'))
    gate = ApprovalGate(ApprovalConfig(timeout_seconds=0, on_timeout='deny'), store=store)
    request = gate.build_request(
        run_id='r1',
        tool_call_id='tc-timeout',
        tool_name='deploy',
        args={},
        approval_id='ap-timeout',
    )

    await gate.open([request])
    assert await gate.get_pending('r1') == []
    assert (await gate.get_decision('tc-timeout')).decision == 'reject'

    await gate.record(ApprovalDecision('ap-timeout', 'approve', decided_at=1.0))
    await gate.record(ApprovalDecision('ap-timeout', 'reject', decided_at=2.0))
    decision = await gate.get_decision('tc-timeout')
    assert decision.decision == 'reject'
    assert decision.decided_at == 2.0
    store.close()


def test_skill_registry_resolves_inline_and_registered_skills():
    registry = SkillRegistry()
    registered = Skill(name='writer', instructions='Write clearly.')
    inline = Skill(name='reviewer', instructions='Review carefully.')
    registry.register(registered)

    assert resolve_skills(
        [SkillRef(name='writer'), SkillRef(name='reviewer', skill=inline)],
        registry,
    ) == [registered, inline]
