from __future__ import annotations
import asyncio, uuid
from typing import Any, AsyncGenerator, Dict, List
from .reducer import State, reduce
from .providers import ProviderRegistry
from .tools import ToolRegistry, validate_io
from .context import ContextManager
from .storage import RunStore

class Agent:
    def __init__(self, id: str, model: Dict[str, Any], prompt_env: str='prod',
                 tools: List[str]|None=None, policies: Dict[str, Any]|None=None):
        self.id = id
        self.model_cfg = model
        self.prompt_env = prompt_env
        self.tools = tools or []
        self.policies = policies or {'max_steps': 24, 'retry': {'attempts': 2}}
        self.providers = ProviderRegistry.default()
        self.toolreg = ToolRegistry.default()
        self.ctxmgr = ContextManager()
        self.store = RunStore.default()

    async def _call_llm_plan(self, run_id: str) -> Dict[str, Any]:
        messages = self.ctxmgr.messages_for_llm(run_id)
        provider = self.providers.get(self.model_cfg['provider'])
        step = await provider.plan(messages, model=self.model_cfg['model'], tools=self.toolreg.schemas())
        return step

    async def _call_tool(self, step: Dict[str, Any]) -> Dict[str, Any]:
        tname = step['tool']
        args = step.get('args', {})
        tool = self.toolreg.get(tname)
        validate_io(tool.input_schema, args)
        result = await tool.run(args, ctx={})
        validate_io(tool.output_schema, result)
        return result

    async def run_stream(self, input: Dict[str, Any]) -> AsyncGenerator[Dict[str, Any], None]:
        run_id = str(uuid.uuid4())
        state = State(run_id=run_id)
        await self.store.append_event(run_id, {'kind':'start', 'agent_id': self.id, 'input':input})
        event: Dict[str, Any] = {'kind':'start'}
        steps = 0
        while True:
            state, effects = reduce(state, event)
            for eff in effects:
                if eff.kind == 'emit':
                    yield eff.payload
                elif eff.kind == 'call_llm':
                    step = await self._call_llm_plan(run_id)
                    event = {'kind':'llm_step', 'step': step}
                    await self.store.append_event(run_id, event)
                    break
                elif eff.kind == 'call_tool':
                    result = await self._call_tool(eff.payload)
                    event = {'kind':'tool_result', 'result': result}
                    await self.store.append_event(run_id, event)
                    break
                elif eff.kind == 'halt':
                    final = eff.payload.get('final','')
                    await self.store.append_event(run_id, {'kind':'final','answer':final})
                    yield {'kind':'final','answer':final}
                    return
            steps += 1
            if steps > self.policies.get('max_steps', 24):
                yield {'kind':'error', 'message':'max steps exceeded'}
                return

async def run_stream(agent: 'Agent', input: Dict[str, Any]):
    async for e in agent.run_stream(input):
        yield e
