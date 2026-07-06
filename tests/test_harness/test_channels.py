"""Feature 3b: channel adapters (CLI + Slack) over RunManager."""
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
from vel.harness import ApprovalDecision, CLIChannel, RunManager, format_event
from vel.harness.channels.slack import _approval_blocks


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


def _agent(provider, tmp_path, *, gated=False):
    approval = {'enabled': True, 'mode': 'durable', 'require_for_tools': ['delete_file']} if gated \
        else {'enabled': False}
    agent = Agent(
        id='ch',
        model={'provider': 'scripted', 'model': 'm'},
        tools=[ToolSpec.from_function(delete_file, name='delete_file')],
        harness={'enabled': True, 'db_path': str(tmp_path / 'vel.db'), 'approval': approval},
    )
    agent._custom_provider = provider
    return agent


class CollectingCLI(CLIChannel):
    def __init__(self, *a, auto='approve', **kw):
        super().__init__(*a, **kw)
        self.sent: List[str] = []
        self._auto = auto

    async def send(self, text: str) -> None:
        self.sent.append(text)

    async def request_approval(self, pending):
        return [
            ApprovalDecision(approval_id=(e.get('data', {}) or {}).get('approval_id'), decision=self._auto)
            for e in pending
        ]


# --------------------------------------------------------------- format_event
def test_format_event_renders_known_types():
    assert format_event({'type': 'tool-input-available', 'toolName': 'x'}) == '→ calling `x`'
    assert 'approval required' in format_event(
        {'type': 'data-harness-approval-required', 'data': {'tool_name': 'x'}})
    assert format_event({'type': 'text-delta', 'delta': 'hi'}) is None
    assert format_event({'type': 'totally-unknown'}) is None


# --------------------------------------------------------------- CLI channel
@pytest.mark.asyncio
async def test_cli_channel_streams_text(tmp_path):
    rm = RunManager(db_path=str(tmp_path / 'events.db'))
    agent = _agent(ScriptedProvider([_text('hello world')]), tmp_path)
    ch = CollectingCLI(agent, run_manager=rm, harness=agent.harness_config)
    await ch.handle('hi')
    assert 'hello world' in ''.join(ch.sent)


@pytest.mark.asyncio
async def test_cli_channel_approval_round_trip(tmp_path):
    rm = RunManager(db_path=str(tmp_path / 'events.db'))
    agent = _agent(
        ScriptedProvider([_tool('tc1', 'delete_file', {'path': '/x'}), _text('all done')]),
        tmp_path, gated=True,
    )
    ch = CollectingCLI(agent, run_manager=rm, harness=agent.harness_config, auto='approve')
    run_id = await ch.handle('delete /x')

    joined = '\n'.join(ch.sent)
    assert 'approval required' in joined      # the gate was surfaced
    assert 'all done' in joined               # and the run continued after approval
    assert await rm.get_status(run_id) == 'completed'


@pytest.mark.asyncio
async def test_cli_channel_reject(tmp_path):
    rm = RunManager(db_path=str(tmp_path / 'events.db'))
    agent = _agent(
        ScriptedProvider([_tool('tc1', 'delete_file', {'path': '/x'}), _text('ok understood')]),
        tmp_path, gated=True,
    )
    ch = CollectingCLI(agent, run_manager=rm, harness=agent.harness_config, auto='reject')
    await ch.handle('delete /x')
    assert 'ok understood' in '\n'.join(ch.sent)  # continued after rejection


# --------------------------------------------------------------- Slack blocks
def test_slack_approval_blocks_encode_ids():
    import json
    blocks = _approval_blocks({'run_id': 'r1', 'approval_id': 'a1', 'tool_name': 'delete_file'})
    actions = [b for b in blocks if b['type'] == 'actions'][0]['elements']
    values = [json.loads(el['value']) for el in actions]
    assert {'run_id': 'r1', 'approval_id': 'a1', 'decision': 'approve'} in values
    assert {'run_id': 'r1', 'approval_id': 'a1', 'decision': 'reject'} in values


@pytest.mark.asyncio
async def test_slack_submit_interaction_resumes(tmp_path):
    pytest.importorskip('slack_sdk')
    import json
    from vel.harness import SlackChannel

    agent = _agent(ScriptedProvider([_text('x')]), tmp_path)
    ch = SlackChannel(agent, token='xoxb-test', harness=agent.harness_config)

    resumed: List[Any] = []

    async def fake_resume(run_id, decisions):
        resumed.append((run_id, decisions))

    async def fake_pump(run_id, cursor=0):
        return None

    ch.rm.resume = fake_resume       # type: ignore[assignment]
    ch._pump = fake_pump             # type: ignore[assignment]

    payload = {
        'channel': {'id': 'C1'},
        'actions': [{'value': json.dumps({'run_id': 'r9', 'approval_id': 'a9', 'decision': 'approve'})}],
    }
    await ch.submit_interaction(payload)
    assert resumed and resumed[0][0] == 'r9'
    assert resumed[0][1][0].approval_id == 'a9'
    assert resumed[0][1][0].decision == 'approve'


def test_slack_requires_sdk_or_constructs(tmp_path):
    agent = _agent(ScriptedProvider([_text('x')]), tmp_path)
    from vel.harness import SlackChannel
    try:
        import slack_sdk  # noqa: F401
    except ImportError:
        with pytest.raises(ImportError):
            SlackChannel(agent, token='xoxb-test')
    else:
        assert SlackChannel(agent, token='xoxb-test') is not None
