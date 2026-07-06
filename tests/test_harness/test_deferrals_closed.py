"""Tests for the previously-deferred items now closed:

#3 auto session->sandbox reuse (reconnect across runs in a session)
#4 memory_offload writing to FactStore + optional ReasoningBank
#5 opt-in durable sub-agents via as_tool(durable=True), backwards compatible
"""
from __future__ import annotations

import hashlib
from typing import Any, Dict, List

import numpy as np
import pytest

from vel import Agent, ToolSpec
from vel.core import ContextManager
from vel.core.context import MemoryConfig
from vel.events import (
    TextStartEvent,
    TextDeltaEvent,
    TextEndEvent,
    ToolInputAvailableEvent,
    FinishMessageEvent,
)
from vel.providers import BaseProvider
from vel.harness.config import CompactionConfig
from vel.harness.compaction import CompactionPolicy


class ScriptedProvider(BaseProvider):
    name = 'scripted'

    def __init__(self, script: List[List[Any]]):
        self._script = list(script)

    async def stream(self, messages, model, tools, generation_config=None):
        batch = self._script.pop(0) if self._script else [
            TextStartEvent(block_id='e'), TextDeltaEvent(block_id='e', delta='done'),
            TextEndEvent(block_id='e'), FinishMessageEvent(finish_reason='stop')]
        for ev in batch:
            yield ev

    async def generate(self, messages, model, tools, generation_config=None):
        return {}


def _tool(cid, name, args):
    return [ToolInputAvailableEvent(tool_call_id=cid, tool_name=name, input=args),
            FinishMessageEvent(finish_reason='tool_calls')]


def _text(t):
    return [TextStartEvent(block_id='b'), TextDeltaEvent(block_id='b', delta=t),
            TextEndEvent(block_id='b'), FinishMessageEvent(finish_reason='stop')]


async def _collect(agen):
    return [ev async for ev in agen]


def _mock_encode(texts):
    return np.array(
        [[b / 255.0 for b in hashlib.sha256(t.encode()).digest()[:16]] for t in texts],
        dtype=np.float32,
    )


def _msgs(n, size=400):
    out = [{'role': 'system', 'content': 'sys'}]
    for i in range(n):
        out.append({'role': 'user', 'content': f'u{i} ' + 'x' * size})
        out.append({'role': 'assistant', 'content': f'a{i} ' + 'y' * size})
    return out


# ----------------------------------------------------- #3 sandbox session reuse
@pytest.mark.asyncio
async def test_per_session_sandbox_reused_across_runs(tmp_path):
    ws = tmp_path / 'ws'
    ws.mkdir()
    harness = {
        'enabled': True,
        'db_path': str(tmp_path / 'vel.db'),
        'approval': {'enabled': True, 'mode': 'durable', 'require_for_confirmation_flag': False},
        'sandbox': {
            'enabled': True, 'provider': 'local_subprocess', 'lifecycle': 'per_session',
            'tools': ['read', 'write'],
            'provider_options': {'unsafe_local': True, 'root': str(ws)},
        },
    }
    agent = Agent(id='reuse', model={'provider': 'scripted', 'model': 'm'}, harness=harness)
    # run 1 writes plan.md, run 2 reads it — same session.
    agent._custom_provider = ScriptedProvider([
        _tool('w', 'sandbox_write', {'path': 'plan.md', 'content': 'shared plan'}), _text('wrote'),
        _tool('r', 'sandbox_read', {'path': 'plan.md'}), _text('read'),
    ])

    ev1 = await _collect(agent.run_stream({'message': 'plan'}, session_id='s1'))
    ref1 = next(e for e in ev1 if e['type'] == 'data-harness-sandbox' and e['data']['event'] == 'created')['data']['sandbox_ref']

    ev2 = await _collect(agent.run_stream({'message': 'continue'}, session_id='s1'))
    sb2 = [e for e in ev2 if e['type'] == 'data-harness-sandbox']
    # run 2 RECONNECTED to the same sandbox rather than creating a new one
    assert any(e['data']['event'] == 'connected' for e in sb2)
    assert sb2[0]['data']['sandbox_ref'] == ref1
    # and it can read what run 1 wrote
    read_out = next(e for e in ev2 if e['type'] == 'tool-output-available' and 'content' in e['output'])
    assert read_out['output']['content'] == 'shared plan'


# ----------------------------------------------------- #4 memory_offload + RB
@pytest.mark.asyncio
async def test_memory_offload_writes_factstore_and_reasoningbank(tmp_path):
    ctx = ContextManager()
    ctx.set_memory_config(MemoryConfig(mode='all', db_path=str(tmp_path / 'mem.db'),
                                       embeddings_fn=_mock_encode))
    ctx._by_run['r'] = _msgs(8)
    agent = Agent(id='m', model={'provider': 'openai', 'model': 'gpt-4o'})
    policy = CompactionPolicy(CompactionConfig(strategy='memory_offload', keep_last_messages=4), agent)

    event = await policy.compact(ctx, 'r', model_cfg=agent.model_cfg)
    assert event is not None and event['data']['strategy'] == 'memory_offload'

    # raw turns landed in the FactStore under the run namespace
    facts = ctx.fact_list('compaction:r')
    assert len(facts) > 0
    assert any('u0' in str(f['value']) or 'a0' in str(f['value']) for f in facts)

    # a distilled strategy landed in ReasoningBank
    rb_store = ctx._adapters.get('rb_store')
    assert rb_store is not None
    strategies = rb_store.retrieve({'source': 'compaction', 'run': 'r'}, k=5, min_conf=0.0)
    assert len(strategies) >= 1


@pytest.mark.asyncio
async def test_memory_offload_noop_when_memory_disabled(tmp_path):
    ctx = ContextManager()  # no memory configured
    ctx._by_run['r'] = _msgs(8)
    agent = Agent(id='m2', model={'provider': 'openai', 'model': 'gpt-4o'})
    policy = CompactionPolicy(CompactionConfig(strategy='memory_offload', keep_last_messages=4), agent)
    event = await policy.compact(ctx, 'r', model_cfg=agent.model_cfg)
    # still compacts the window (degrades to reduce) without raising
    assert event is not None
    assert len(ctx._by_run['r']) < len(_msgs(8))


# ----------------------------------------------------- #5 durable sub-agents
@pytest.mark.asyncio
async def test_as_tool_durable_runs_through_harness(tmp_path):
    sub = Agent(
        id='sub', model={'provider': 'scripted', 'model': 'm'},
        harness={'enabled': True, 'db_path': str(tmp_path / 'sub.db'),
                 'budget': {'max_steps': 10}},
    )
    sub._custom_provider = ScriptedProvider([_text('sub answer')])
    tool = sub.as_tool(name='sub_agent', durable=True)
    # invoke the tool handler directly
    out = await tool._handler({'message': 'do research'}, {})
    assert out == {'response': 'sub answer'}


@pytest.mark.asyncio
async def test_as_tool_default_is_non_durable_backwards_compatible():
    import inspect
    # default must be False (additive, backwards compatible)
    assert inspect.signature(Agent.as_tool).parameters['durable'].default is False

    # and the default path must route through run() (the legacy non-durable
    # path), NOT run_stream — verify by spying on both.
    sub = Agent(id='sub2', model={'provider': 'scripted', 'model': 'm'})
    calls = {'run': 0, 'run_stream': 0}

    async def fake_run(inp, **kw):
        calls['run'] += 1
        return 'plain answer'

    async def fake_run_stream(inp, **kw):
        calls['run_stream'] += 1
        if False:
            yield {}

    sub.run = fake_run
    sub.run_stream = fake_run_stream
    tool = sub.as_tool(name='sub2_agent')  # durable defaults False
    out = await tool._handler({'message': 'hi'}, {})
    assert out == {'response': 'plain answer'}
    assert calls == {'run': 1, 'run_stream': 0}
