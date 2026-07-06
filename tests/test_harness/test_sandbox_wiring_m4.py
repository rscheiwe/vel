"""M4: sandbox lifecycle wired into the run (controller §6.7).

Proves that with ``sandbox.enabled`` the harness creates a session, injects the
sandbox tools, and the agent can maintain a workspace file (the "plan tracking"
pattern) across steps — then the sandbox is torn down (per_run).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import pytest

from vel import Agent
from vel.events import (
    TextStartEvent,
    TextDeltaEvent,
    TextEndEvent,
    ToolInputAvailableEvent,
    FinishMessageEvent,
)
from vel.providers import BaseProvider


class ScriptedProvider(BaseProvider):
    name = 'scripted'

    def __init__(self, script: List[List[Any]]):
        self._script = list(script)
        self.last_tools = None

    async def stream(self, messages, model, tools, generation_config=None):
        self.last_tools = tools
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


def _harness(tmp_path):
    ws = tmp_path / 'ws'
    ws.mkdir()
    return {
        'enabled': True,
        'db_path': str(tmp_path / 'vel.db'),
        # routine workspace file ops are NOT gated; only explicit tools would be
        'approval': {'enabled': True, 'mode': 'durable', 'require_for_confirmation_flag': False},
        'sandbox': {
            'enabled': True,
            'provider': 'local_subprocess',
            'lifecycle': 'per_run',
            'tools': ['read', 'write', 'list'],
            'provider_options': {'unsafe_local': True, 'root': str(ws)},
        },
    }


@pytest.mark.asyncio
async def test_sandbox_created_tools_injected_and_plan_tracked(tmp_path):
    # Step 1: write a plan file; Step 2: read it back; Step 3: answer.
    provider = ScriptedProvider([
        _tool('c1', 'sandbox_write', {'path': 'plan.md', 'content': '1. research\n2. report'}),
        _tool('c2', 'sandbox_read', {'path': 'plan.md'}),
        _text('plan tracked'),
    ])
    agent = Agent(id='research', model={'provider': 'scripted', 'model': 'gpt-4o'},
                  harness=_harness(tmp_path))
    agent._custom_provider = provider

    events = await _collect(agent.run_stream({'message': 'plan a research task'}))
    types = [e['type'] for e in events]

    # sandbox lifecycle events bracket the run
    sb = [e for e in events if e['type'] == 'data-harness-sandbox']
    assert any(e['data']['event'] == 'created' for e in sb)
    assert any(e['data']['event'] == 'closed' for e in sb)

    # the sandbox tools were injected and visible to the model
    assert 'sandbox_write' in agent._injected_tools
    assert 'sandbox_read' in agent._injected_tools

    # the write executed (not gated) and the read returned the plan content
    outputs = [e for e in events if e['type'] == 'tool-output-available']
    read_out = next(o for o in outputs if 'content' in o['output'])
    assert '1. research' in read_out['output']['content']

    # plan.md really exists on disk in the workspace (relative paths resolve
    # under <root>/workspace, the default sandbox workdir)
    assert (tmp_path / 'ws' / 'workspace' / 'plan.md').read_text().startswith('1. research')
    assert types[-1] in ('finish', 'data-harness-run-finished')


@pytest.mark.asyncio
async def test_sandbox_disabled_is_noop(tmp_path):
    provider = ScriptedProvider([_text('hi')])
    agent = Agent(id='nosb', model={'provider': 'scripted', 'model': 'gpt-4o'},
                  harness={'enabled': True, 'db_path': str(tmp_path / 'vel.db')})
    agent._custom_provider = provider
    events = await _collect(agent.run_stream({'message': 'x'}))
    assert not any(e['type'] == 'data-harness-sandbox' for e in events)
    assert agent._injected_tools == {}
