from __future__ import annotations
import asyncio
import warnings
from typing import Any, AsyncGenerator, Dict, List, Optional, Literal
from .core import State, reduce, ContextManager
from .providers import ProviderRegistry
from .tools import ToolRegistry, validate_io
from .storage import RunStore
from .events import (
    StreamEvent, ToolCallEvent, ToolResultEvent,
    ErrorEvent, FinishMessageEvent
)
from .prompts import PromptContextManager

class Agent:
    def __init__(self, id: str, model: Dict[str, Any], prompt_env: str='prod',
                 tools: List[str]|None=None, policies: Dict[str, Any]|None=None,
                 context_manager: Optional[ContextManager]=None,
                 session_persistence: Optional[Literal['transient', 'persistent']]=None,
                 prompt_id: Optional[str]=None,
                 prompt_vars: Optional[Dict[str, Any]]=None,
                 # Deprecated (backwards compatibility)
                 session_storage: Optional[Literal['memory', 'database']]=None):
        """
        Initialize an Agent.

        Vel has three distinct memory systems:
        1. **Message History** - Conversation turns (managed by ContextManager)
        2. **Fact Store** - Long-term structured facts (via MemoryConfig)
        3. **Session Persistence** - Where message history is saved (this parameter)

        Args:
            id: Agent identifier
            model: Model config with 'provider' and 'model' keys
            prompt_env: Environment for prompts (default: 'prod')
            tools: List of tool names to enable
            policies: Execution policies (max_steps, retry, etc.)

            context_manager: Custom context manager instance. Pass:
                - None or ContextManager() for default (full message history)
                - StatelessContextManager() for no message history
                - ContextManager(max_history=10) for limited message history
                - Your own custom ContextManager subclass

            session_persistence: Where message history is saved:
                - 'transient': In-memory only (default, fast, not persistent)
                - 'persistent': Database-backed (survives restarts, requires PostgreSQL)
                - None: defaults to 'transient'

            prompt_id: Optional prompt template ID (e.g., 'chat-agent:v1')
            prompt_vars: Optional variables for prompt template rendering

            session_storage: [DEPRECATED] Use session_persistence instead
                - 'memory' → use 'transient'
                - 'database' → use 'persistent'
        """
        self.id = id
        self.model_cfg = model
        self.prompt_env = prompt_env
        self.tools = tools or []
        self.policies = policies or {'max_steps': 24, 'retry': {'attempts': 2}}

        # Handle backwards compatibility for session_storage
        if session_storage is not None:
            warnings.warn(
                f"Agent parameter 'session_storage' is deprecated and will be removed in v2.0. "
                f"Use 'session_persistence' instead. "
                f"('memory' → 'transient', 'database' → 'persistent')",
                DeprecationWarning,
                stacklevel=2
            )
            # Map old values to new
            mapping = {'memory': 'transient', 'database': 'persistent'}
            session_persistence = mapping.get(session_storage, 'transient')

        # Set session persistence (default to 'transient')
        self.session_persistence = session_persistence or 'transient'
        self.providers = ProviderRegistry.default()
        self.toolreg = ToolRegistry.default()

        # Context manager setup with prompt support
        if context_manager is not None:
            # User provided custom context manager - use as-is
            self.ctxmgr = context_manager
        elif prompt_id:
            # Prompt template provided - use PromptContextManager
            self.ctxmgr = PromptContextManager(
                prompt_id=prompt_id,
                prompt_vars=prompt_vars,
                prompt_env=prompt_env
            )
        else:
            # Default context manager (backwards compatible)
            self.ctxmgr = ContextManager()

        self.store = RunStore.default()

    async def _load_session(self, session_id: str):
        """Load session from database if using persistent storage"""
        if self.session_persistence == 'persistent':
            context = await self.store.load_session(session_id)
            if context:
                self.ctxmgr.set_session_context(session_id, context)

    async def _save_session(self, session_id: str):
        """Save session to database if using persistent storage"""
        if self.session_persistence == 'persistent':
            context = self.ctxmgr.get_session_context(session_id)
            await self.store.save_session(session_id, context)

    async def _call_llm_generate(self, run_id: str, session_id: Optional[str] = None) -> Dict[str, Any]:
        """Non-streaming LLM call"""
        messages = self.ctxmgr.messages_for_llm(run_id, session_id)
        provider = self.providers.get(self.model_cfg['provider'])
        step = await provider.generate(messages, model=self.model_cfg['model'], tools=self.toolreg.schemas())
        return step

    async def _call_tool(self, tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a tool"""
        tool = self.toolreg.get(tool_name)
        validate_io(tool.input_schema, args)
        result = await tool.run(args, ctx={})
        validate_io(tool.output_schema, result)
        return result

    async def run(self, input: Dict[str, Any], session_id: Optional[str] = None) -> str:
        """
        Non-streaming run - returns final answer only.

        Args:
            input: Input dict with 'message' field
            session_id: Optional session ID for multi-turn conversations.
                       If provided, context persists across multiple run() calls.
        """
        # Load session from DB if using database storage
        if session_id:
            await self._load_session(session_id)

        run_id = await self.store.create_run(self.id)
        self.ctxmgr.set_input(run_id, input, session_id)
        state = State(run_id=run_id)
        await self.store.append_event(run_id, {'kind':'start', 'agent_id': self.id, 'input':input})
        event: Dict[str, Any] = {'kind':'start'}
        steps = 0
        final_answer = ''

        try:
            while True:
                state, effects = reduce(state, event)
                for eff in effects:
                    if eff.kind == 'call_llm':
                        step = await self._call_llm_generate(run_id, session_id)
                        event = {'kind':'llm_step', 'step': step}
                        await self.store.append_event(run_id, event)
                        break
                    elif eff.kind == 'call_tool':
                        result = await self._call_tool(eff.payload['tool'], eff.payload.get('args', {}))
                        # Add tool result to context
                        self.ctxmgr.append_tool_result(run_id, eff.payload['tool'], result, session_id)
                        event = {'kind':'tool_result', 'result': result}
                        await self.store.append_event(run_id, event)
                        break
                    elif eff.kind == 'halt':
                        final_answer = eff.payload.get('final','')
                        # Add assistant response to context
                        self.ctxmgr.append_assistant_message(run_id, final_answer, session_id)
                        # Save session to DB if using database storage
                        if session_id:
                            await self._save_session(session_id)
                        await self.store.append_event(run_id, {'kind':'final','answer':final_answer})
                        await self.store.update_status(run_id, 'completed')
                        return final_answer
                steps += 1
                if steps > self.policies.get('max_steps', 24):
                    msg = 'max steps exceeded'
                    await self.store.append_event(run_id, {'kind':'error','message': msg})
                    await self.store.update_status(run_id, 'failed')
                    raise RuntimeError(msg)
        except asyncio.CancelledError:
            await self.store.update_status(run_id, 'canceled')
            raise
        except Exception as e:
            await self.store.append_event(run_id, {'kind':'error','message': str(e)})
            await self.store.update_status(run_id, 'failed')
            raise

    async def run_stream(self, input: Dict[str, Any], session_id: Optional[str] = None) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Streaming run - yields stream protocol events as they occur.

        Args:
            input: Input dict with 'message' field
            session_id: Optional session ID for multi-turn conversations.
                       If provided, context persists across multiple run_stream() calls.
        """
        # Load session from DB if using database storage
        if session_id:
            await self._load_session(session_id)

        run_id = await self.store.create_run(self.id)
        self.ctxmgr.set_input(run_id, input, session_id)
        await self.store.append_event(run_id, {'kind':'start', 'agent_id': self.id, 'input':input})

        steps = 0
        max_steps = self.policies.get('max_steps', 24)

        try:
            while steps < max_steps:
                steps += 1

                # Get messages and stream LLM response
                messages = self.ctxmgr.messages_for_llm(run_id, session_id)
                provider = self.providers.get(self.model_cfg['provider'])

                # Track what happened during streaming
                full_text = []
                tool_calls = []  # list of {tool_call_id, tool_name, input}
                finish_reason = 'stop'

                # Stream from provider and forward events
                async for event in provider.stream(messages, model=self.model_cfg['model'], tools=self.toolreg.schemas()):
                    # Forward stream protocol events
                    yield event.to_dict()

                    # Track text content
                    if event.type == 'text-delta':
                        full_text.append(event.delta)

                    # Track tool calls
                    elif event.type == 'tool-call':
                        tool_calls.append({
                            'tool_call_id': event.tool_call_id,
                            'tool_name': event.tool_name,
                            'input': event.input
                        })

                    # Track completion
                    elif event.type == 'finish-message':
                        finish_reason = event.finish_reason
                        break

                    # Handle errors
                    elif event.type == 'error':
                        await self.store.append_event(run_id, {'kind':'error', 'message': event.error})
                        await self.store.update_status(run_id, 'failed')
                        return

                # If we got text and no tool calls, we're done
                if full_text and not tool_calls:
                    answer = ''.join(full_text)
                    self.ctxmgr.append_assistant_message(run_id, answer, session_id)
                    # Save session to DB if using database storage
                    if session_id:
                        await self._save_session(session_id)
                    await self.store.append_event(run_id, {'kind':'final', 'answer': answer})
                    await self.store.update_status(run_id, 'completed')
                    return

                # If we got tool calls, execute them and continue
                if tool_calls:
                    for tc in tool_calls:
                        try:
                            # Execute tool
                            result = await self._call_tool(tc['tool_name'], tc['input'])

                            # Emit tool result event
                            output_event = ToolResultEvent(
                                tool_call_id=tc['tool_call_id'],
                                result=result
                            )
                            yield output_event.to_dict()

                            # Add to context for next iteration
                            self.ctxmgr.append_tool_result(run_id, tc['tool_name'], result, session_id)

                            await self.store.append_event(run_id, {
                                'kind': 'tool_result',
                                'tool': tc['tool_name'],
                                'result': result
                            })
                        except Exception as e:
                            error_event = ErrorEvent(error=f"Tool execution failed: {str(e)}")
                            yield error_event.to_dict()
                            await self.store.append_event(run_id, {'kind':'error', 'message': str(e)})
                            await self.store.update_status(run_id, 'failed')
                            return

                    # Continue loop to get next LLM response
                    continue

                # If we got here with no text and no tool calls, something's wrong
                await self.store.append_event(run_id, {'kind':'error', 'message': 'No response from LLM'})
                await self.store.update_status(run_id, 'failed')
                error_event = ErrorEvent(error='No response from LLM')
                yield error_event.to_dict()
                return

            # Max steps exceeded
            msg = f'max steps ({max_steps}) exceeded'
            await self.store.append_event(run_id, {'kind':'error','message': msg})
            await self.store.update_status(run_id, 'failed')
            error_event = ErrorEvent(error=msg)
            yield error_event.to_dict()

        except asyncio.CancelledError:
            await self.store.update_status(run_id, 'canceled')
            raise
        except Exception as e:
            await self.store.append_event(run_id, {'kind':'error','message': str(e)})
            await self.store.update_status(run_id, 'failed')
            error_event = ErrorEvent(error=str(e))
            yield error_event.to_dict()
            raise

async def run_stream(agent: 'Agent', input: Dict[str, Any]):
    """Helper function for streaming"""
    async for e in agent.run_stream(input):
        yield e
