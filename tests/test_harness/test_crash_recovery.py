"""Feature 2: crash recovery — per-tool checkpoints + replay-skip.

Covers per-tool checkpoint persistence (opt-in), the default-off backwards
behavior, and recovery skipping already-completed tools.
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
from vel.harness.checkpoint import CheckpointStore, RunCheckpoint

CALLS: List[str] = []


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


def _two_tools() -> List[Any]:
    return [
        ToolInputAvailableEvent(tool_call_id='tc1', tool_name='counter', input={'target': 'a'}),
        ToolInputAvailableEvent(tool_call_id='tc2', tool_name='counter', input={'target': 'b'}),
        FinishMessageEvent(finish_reason='tool_calls'),
    ]


async def counter(target: str = '', ctx: dict = None) -> dict:
    CALLS.append(target)
    return {'counted': target}


async def _collect(agen) -> List[Dict[str, Any]]:
    return [ev async for ev in agen]


def _make_agent(provider, tmp_path, *, checkpoint_each_tool: bool):
    agent = Agent(
        id='rec',
        model={'provider': 'scripted', 'model': 'm'},
        tools=[ToolSpec.from_function(counter, name='counter')],
        harness={
            'enabled': True,
            'db_path': str(tmp_path / 'vel.db'),
            'checkpoint_each_tool': checkpoint_each_tool,
            'approval': {'enabled': False},
        },
    )
    agent._custom_provider = provider
    return agent


@pytest.fixture(autouse=True)
def _reset_calls():
    CALLS.clear()
    yield
    CALLS.clear()


def _spy_saves(monkeypatch) -> List[List[str]]:
    """Capture completed_tool_calls of every checkpoint written."""
    captured: List[List[str]] = []
    original = CheckpointStore.save

    def spy(self, ckpt: RunCheckpoint):
        captured.append(list(ckpt.completed_tool_calls))
        return original(self, ckpt)

    monkeypatch.setattr(CheckpointStore, 'save', spy)
    return captured


@pytest.mark.asyncio
async def test_per_tool_checkpoint_when_enabled(tmp_path, monkeypatch):
    captured = _spy_saves(monkeypatch)
    provider = ScriptedProvider([_two_tools(), _text('done')])
    agent = _make_agent(provider, tmp_path, checkpoint_each_tool=True)
    await _collect(agent.run_stream({'message': 'go'}))
    # at least one checkpoint captured mid-step progress (tc1 completed).
    assert any('tc1' in c for c in captured)
    assert CALLS == ['a', 'b']


@pytest.mark.asyncio
async def test_no_per_tool_checkpoint_by_default(tmp_path, monkeypatch):
    captured = _spy_saves(monkeypatch)
    provider = ScriptedProvider([_two_tools(), _text('done')])
    agent = _make_agent(provider, tmp_path, checkpoint_each_tool=False)
    await _collect(agent.run_stream({'message': 'go'}))
    # default: only step-boundary checkpoints, never any completed_tool_calls.
    assert all(c == [] for c in captured)
    assert CALLS == ['a', 'b']


@pytest.mark.asyncio
async def test_recover_skips_completed_tools(tmp_path):
    # Simulate a crash mid-step: tc1 completed + committed, tc2 not yet run.
    db = str(tmp_path / 'vel.db')
    store = CheckpointStore(db)
    ckpt = RunCheckpoint(
        run_id='r1', agent_id='rec', session_id='s', status='running', step=1,
        messages=[
            {'role': 'user', 'content': 'go'},
            {'role': 'assistant', 'content': None, 'tool_calls': [
                {'id': 'tc1', 'type': 'function', 'function': {'name': 'counter', 'arguments': '{"target": "a"}'}},
                {'id': 'tc2', 'type': 'function', 'function': {'name': 'counter', 'arguments': '{"target": "b"}'}},
            ]},
            {'role': 'tool', 'tool_call_id': 'tc1', 'content': '{"counted": "a"}'},
        ],
        pending_tool_calls=[
            {'tool_call_id': 'tc1', 'tool_name': 'counter', 'input': {'target': 'a'}},
            {'tool_call_id': 'tc2', 'tool_name': 'counter', 'input': {'target': 'b'}},
        ],
        completed_tool_calls=['tc1'],
        budget_state={'steps': 1},
        config_hash='',  # empty -> resume guard skipped
    )
    store.save(ckpt)

    provider = ScriptedProvider([_text('done')])
    agent = _make_agent(provider, tmp_path, checkpoint_each_tool=True)
    events = await _collect(agent.recover('r1'))

    # tc1 (completed) NOT re-run; only tc2 executed on recovery.
    assert CALLS == ['b']
    rec = next(e for e in events if e['type'] == 'data-harness-recovered')
    assert rec['data']['skipped_tools'] == 1
    out = [e for e in events if e['type'] == 'tool-output-available']
    assert out and out[-1]['output'] == {'counted': 'b'}
    assert 'data-harness-run-finished' in [e['type'] for e in events]
    assert store.load('r1').status == 'completed'


@pytest.mark.asyncio
async def test_recover_requires_running_checkpoint(tmp_path):
    provider = ScriptedProvider([_text('x')])
    agent = _make_agent(provider, tmp_path, checkpoint_each_tool=True)
    with pytest.raises(ValueError):
        await _collect(agent.recover('missing'))
