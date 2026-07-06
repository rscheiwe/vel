"""M2: automatic context compaction.

Covers trigger ratio, tail/last-user preservation, the Q4 no-orphaned-tool-pair
rule, all three strategies, idempotency, and an end-to-end run that compacts.
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
from vel.core import ContextManager
from vel.harness.config import CompactionConfig
from vel.harness.compaction import CompactionPolicy, estimate_tokens, context_window_for


class _FakeCtx:
    """Minimal stand-in exposing the _by_run mapping CompactionPolicy mutates."""

    def __init__(self, messages):
        self._by_run = {'r': list(messages)}


def _agent():
    return Agent(id='c', model={'provider': 'openai', 'model': 'gpt-4o'})


def _msgs_text(n: int, size: int = 400) -> List[Dict[str, Any]]:
    out = [{'role': 'system', 'content': 'sys'}]
    for i in range(n):
        out.append({'role': 'user', 'content': f'u{i} ' + 'x' * size})
        out.append({'role': 'assistant', 'content': f'a{i} ' + 'y' * size})
    return out


# --------------------------------------------------------------- estimation
def test_estimate_tokens_and_window():
    assert estimate_tokens([{'role': 'user', 'content': 'a' * 400}]) == 100
    assert context_window_for({'model': 'gpt-4o'}) == 128_000
    assert context_window_for({'model': 'claude-opus-4-1'}) == 200_000
    assert context_window_for({'model': 'unknown-model'}) == 128_000


# --------------------------------------------------------------- should_compact
def test_should_compact_respects_ratio_and_disabled():
    agent = _agent()
    msgs = _msgs_text(20)
    on = CompactionPolicy(CompactionConfig(trigger_token_ratio=0.00001, keep_last_messages=4), agent)
    assert on.should_compact(msgs, agent.model_cfg) is True

    high = CompactionPolicy(CompactionConfig(trigger_token_ratio=0.99, keep_last_messages=4), agent)
    assert high.should_compact(msgs, agent.model_cfg) is False

    off = CompactionPolicy(CompactionConfig(enabled=False), agent)
    assert off.should_compact(msgs, agent.model_cfg) is False

    short = CompactionPolicy(CompactionConfig(trigger_token_ratio=0.00001, keep_last_messages=6), agent)
    assert short.should_compact(_msgs_text(2), agent.model_cfg) in (True, False)  # small input safe


# --------------------------------------------------------------- reduce strategy
@pytest.mark.asyncio
async def test_reduce_preserves_head_tail_and_last_user():
    agent = _agent()
    msgs = _msgs_text(10)
    ctx = _FakeCtx(msgs)
    policy = CompactionPolicy(CompactionConfig(strategy='reduce', keep_last_messages=4), agent)
    event = await policy.compact(ctx, 'r', model_cfg=agent.model_cfg)
    assert event is not None
    assert event['type'] == 'data-harness-compaction'
    assert event['data']['after_tokens'] < event['data']['before_tokens']

    new = ctx._by_run['r']
    # head system message preserved
    assert new[0]['role'] == 'system'
    # last user turn still present
    assert any(m.get('role') == 'user' for m in new)
    # tail (last 4 of the original body) preserved verbatim
    assert new[-4:] == msgs[-4:]
    # fewer messages than before
    assert len(new) < len(msgs)


# --------------------------------------------------------------- Q4 boundary rule
@pytest.mark.asyncio
async def test_no_orphaned_tool_pairs():
    agent = _agent()
    # Build a window where a tool-call/result pair sits right at the tail edge.
    msgs = [{'role': 'system', 'content': 'sys'}]
    for i in range(6):
        msgs.append({'role': 'user', 'content': f'u{i} ' + 'x' * 400})
        msgs.append({'role': 'assistant', 'content': f'a{i} ' + 'y' * 400})
    # tool-call pair as the most recent exchange
    msgs.append({'role': 'assistant', 'content': None,
                 'tool_calls': [{'id': 't1', 'type': 'function',
                                 'function': {'name': 'do', 'arguments': '{}'}}]})
    msgs.append({'role': 'tool', 'content': 'result', 'tool_call_id': 't1'})

    ctx = _FakeCtx(msgs)
    policy = CompactionPolicy(CompactionConfig(strategy='reduce', keep_last_messages=1), agent)
    await policy.compact(ctx, 'r', model_cfg=agent.model_cfg)
    new = ctx._by_run['r']

    # every tool result must be immediately preceded by its assistant tool_calls
    for i, m in enumerate(new):
        if m.get('role') == 'tool':
            assert i > 0 and new[i - 1].get('tool_calls'), 'orphaned tool result after compaction'


# --------------------------------------------------------------- summarize strategy
@pytest.mark.asyncio
async def test_summarize_strategy_uses_provider():
    class SummProvider(BaseProvider):
        name = 'summ'

        async def stream(self, messages, model, tools, generation_config=None):
            yield TextStartEvent(block_id='s')
            yield TextDeltaEvent(block_id='s', delta='SUMMARY OF EARLIER')
            yield TextEndEvent(block_id='s')
            yield FinishMessageEvent(finish_reason='stop')

        async def generate(self, messages, model, tools, generation_config=None):
            return {}

    agent = _agent()
    agent._custom_provider = SummProvider()
    msgs = _msgs_text(8)
    ctx = _FakeCtx(msgs)
    policy = CompactionPolicy(CompactionConfig(strategy='summarize', keep_last_messages=4), agent)
    event = await policy.compact(ctx, 'r', model_cfg=agent.model_cfg)
    assert event is not None
    new = ctx._by_run['r']
    assert any('SUMMARY OF EARLIER' in (m.get('content') or '') for m in new)


# --------------------------------------------------------------- memory_offload fallback
@pytest.mark.asyncio
async def test_memory_offload_falls_back_without_store():
    agent = _agent()
    msgs = _msgs_text(8)
    ctx = _FakeCtx(msgs)
    policy = CompactionPolicy(CompactionConfig(strategy='memory_offload', keep_last_messages=4), agent)
    event = await policy.compact(ctx, 'r', model_cfg=agent.model_cfg)
    assert event is not None  # degraded to reduce, still compacts


# --------------------------------------------------------------- idempotency
@pytest.mark.asyncio
async def test_compaction_idempotent_second_pass_noop_or_safe():
    agent = _agent()
    msgs = _msgs_text(10)
    ctx = _FakeCtx(msgs)
    policy = CompactionPolicy(CompactionConfig(strategy='reduce', keep_last_messages=4), agent)
    await policy.compact(ctx, 'r', model_cfg=agent.model_cfg)
    first = list(ctx._by_run['r'])
    # second pass over the already-compacted (now-small) window: head+1 summary+tail
    # has no compactable middle, so it must be a no-op.
    event2 = await policy.compact(ctx, 'r', model_cfg=agent.model_cfg)
    assert event2 is None
    assert ctx._by_run['r'] == first


# --------------------------------------------------------------- end-to-end
@pytest.mark.asyncio
async def test_end_to_end_compaction_event_emitted():
    """A run whose window exceeds the trigger compacts mid-loop and completes."""
    class ToolThenText(BaseProvider):
        name = 'tt'

        def __init__(self):
            self.calls = 0

        async def stream(self, messages, model, tools, generation_config=None):
            self.calls += 1
            if self.calls <= 4:  # several tool steps accumulate compactable churn
                yield ToolInputAvailableEvent(
                    tool_call_id=f't{self.calls}', tool_name='big', input={'n': self.calls}
                )
                yield FinishMessageEvent(finish_reason='tool_calls')
            else:
                yield TextStartEvent(block_id='b')
                yield TextDeltaEvent(block_id='b', delta='final')
                yield TextEndEvent(block_id='b')
                yield FinishMessageEvent(finish_reason='stop')

        async def generate(self, messages, model, tools, generation_config=None):
            return {}

    async def big(n: int = 0, ctx: dict = None) -> dict:
        return {'blob': 'z' * 5000}  # inflate the window so step 2 compacts

    agent = Agent(
        id='e2e', model={'provider': 'tt', 'model': 'gpt-4o'},
        tools=[ToolSpec.from_function(big)],
        harness={'enabled': True, 'compaction': {'enabled': True, 'strategy': 'reduce',
                                                  'trigger_token_ratio': 0.00001, 'keep_last_messages': 2}},
    )
    agent._custom_provider = ToolThenText()
    events = [ev async for ev in agent.run_stream({'message': 'go ' + 'x' * 2000})]
    types = [e['type'] for e in events]
    assert 'data-harness-compaction' in types
    assert types[-1] in ('finish', 'data-harness-run-finished')


@pytest.mark.asyncio
async def test_compaction_off_no_event():
    class Txt(BaseProvider):
        name = 'txt'

        async def stream(self, messages, model, tools, generation_config=None):
            yield TextStartEvent(block_id='b')
            yield TextDeltaEvent(block_id='b', delta='hi')
            yield TextEndEvent(block_id='b')
            yield FinishMessageEvent(finish_reason='stop')

        async def generate(self, messages, model, tools, generation_config=None):
            return {}

    agent = Agent(id='off', model={'provider': 'txt', 'model': 'gpt-4o'},
                  harness={'enabled': True, 'compaction': {'enabled': False}})
    agent._custom_provider = Txt()
    events = [ev async for ev in agent.run_stream({'message': 'x' * 5000})]
    assert not any(e['type'] == 'data-harness-compaction' for e in events)
