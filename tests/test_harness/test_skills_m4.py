"""M4: skill resolution wired into the controller (run-start integration).

A configured skill injects its instructions (into the system context) and its
tools (into the available tool set) for the run.
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
from vel.harness import Skill, SkillRef
from vel.harness.controller import HarnessController
from vel.harness.config import HarnessConfig


class ScriptedProvider(BaseProvider):
    name = 'scripted'

    def __init__(self, script: List[List[Any]]):
        self._script = list(script)
        self.last_tools = None
        self.last_messages = None

    async def stream(self, messages, model, tools, generation_config=None):
        self.last_tools = tools
        self.last_messages = messages
        batch = self._script.pop(0) if self._script else [
            TextStartEvent(block_id='e'), TextDeltaEvent(block_id='e', delta='ok'),
            TextEndEvent(block_id='e'), FinishMessageEvent(finish_reason='stop')]
        for ev in batch:
            yield ev

    async def generate(self, messages, model, tools, generation_config=None):
        return {}


async def skill_tool(x: str = '', ctx: dict = None) -> dict:
    return {'skilled': x}


async def _collect(agen):
    return [ev async for ev in agen]


@pytest.mark.asyncio
async def test_skill_injects_instructions_and_tools():
    provider = ScriptedProvider([[
        TextStartEvent(block_id='b'), TextDeltaEvent(block_id='b', delta='done'),
        TextEndEvent(block_id='b'), FinishMessageEvent(finish_reason='stop')]])
    skill = Skill(
        name='researcher',
        instructions='You are an expert researcher. Cite sources.',
        tools=[ToolSpec.from_function(skill_tool, name='skill_tool')],
    )
    agent = Agent(
        id='sk', model={'provider': 'scripted', 'model': 'm'},
        harness=HarnessConfig(enabled=True, skills=[SkillRef(name='researcher', skill=skill)]),
    )
    agent._custom_provider = provider

    await _collect(agent.run_stream({'message': 'hi'}))

    # the skill's tool is now visible to the model
    assert 'skill_tool' in agent._injected_tools
    schema_names = list(provider.last_tools.keys()) if isinstance(provider.last_tools, dict) else [
        t.get('function', {}).get('name') or t.get('name') for t in (provider.last_tools or [])
    ]
    assert any('skill_tool' == n for n in schema_names)
    # the skill's instructions were prepended into the run context
    joined = ' '.join(str(m.get('content', '')) for m in (provider.last_messages or []))
    assert 'expert researcher' in joined


@pytest.mark.asyncio
async def test_skill_tool_is_executable():
    provider = ScriptedProvider([
        [ToolInputAvailableEvent(tool_call_id='c1', tool_name='skill_tool', input={'x': 'q'}),
         FinishMessageEvent(finish_reason='tool_calls')],
        [TextStartEvent(block_id='b'), TextDeltaEvent(block_id='b', delta='done'),
         TextEndEvent(block_id='b'), FinishMessageEvent(finish_reason='stop')],
    ])
    skill = Skill(name='s', instructions='use it', tools=[ToolSpec.from_function(skill_tool, name='skill_tool')])
    agent = Agent(id='sk2', model={'provider': 'scripted', 'model': 'm'},
                  harness=HarnessConfig(enabled=True, skills=[SkillRef(name='s', skill=skill)]))
    agent._custom_provider = provider
    events = await _collect(agent.run_stream({'message': 'go'}))
    out = next(e for e in events if e['type'] == 'tool-output-available')
    assert out['output'] == {'skilled': 'q'}


@pytest.mark.asyncio
async def test_no_skills_no_injection():
    provider = ScriptedProvider([[
        TextStartEvent(block_id='b'), TextDeltaEvent(block_id='b', delta='done'),
        TextEndEvent(block_id='b'), FinishMessageEvent(finish_reason='stop')]])
    agent = Agent(id='ns', model={'provider': 'scripted', 'model': 'm'},
                  harness=HarnessConfig(enabled=True))
    agent._custom_provider = provider
    await _collect(agent.run_stream({'message': 'hi'}))
    assert agent._injected_tools == {}
