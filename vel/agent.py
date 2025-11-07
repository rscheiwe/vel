from __future__ import annotations
import asyncio
import warnings
import logging
import uuid
from typing import Any, AsyncGenerator, Dict, List, Optional, Literal

# Configure logger for error surfacing
logger = logging.getLogger('vel.agent')
from .core import State, reduce, ContextManager
from .providers import ProviderRegistry
from .tools import ToolRegistry, validate_io
from .events import (
    StreamEvent, StartEvent, FinishEvent, ToolInputAvailableEvent, ToolOutputAvailableEvent,
    ErrorEvent, FinishMessageEvent, StepStartEvent, StepFinishEvent
)
from .prompts import PromptContextManager

class Agent:
    def __init__(self, id: str, model: Dict[str, Any], prompt_env: str='prod',
                 tools: List[str]|None=None, policies: Dict[str, Any]|None=None,
                 context_manager: Optional[ContextManager]=None,
                 session_persistence: Optional[Literal['transient', 'persistent']]=None,
                 prompt_id: Optional[str]=None,
                 prompt_vars: Optional[Dict[str, Any]]=None,
                 generation_config: Optional[Dict[str, Any]]=None,
                 rlm: Optional[Dict[str, Any]]=None,
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
            model: Model config with 'provider' and 'model' keys. Optionally include 'api_key'
                   to override environment variable for this specific agent instance.
                   Examples:
                   - {'provider': 'openai', 'model': 'gpt-4o'}  # Uses OPENAI_API_KEY env var
                   - {'provider': 'openai', 'model': 'gpt-4o', 'api_key': 'sk-...'}  # Uses provided key
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

            generation_config: Model generation parameters (temperature, max_tokens, etc.)
                Common parameters:
                - temperature: float (0-2) - Sampling temperature
                - max_tokens: int - Maximum output tokens
                - top_p: float (0-1) - Nucleus sampling
                - top_k: int - Top-K sampling (Gemini, Anthropic)
                - presence_penalty: float (-2 to 2) - Penalize new tokens (OpenAI)
                - frequency_penalty: float (-2 to 2) - Penalize repeated tokens (OpenAI)
                - stop: List[str] - Stop sequences
                - seed: int - Reproducibility seed (OpenAI, Anthropic)

            rlm: RLM (Recursive Language Model) configuration for handling long contexts
                Dictionary that will be converted to RlmConfig. Set 'enabled': True to activate.
                See RlmConfig for full options.

            session_storage: [DEPRECATED] Use session_persistence instead
                - 'memory' → use 'transient'
                - 'database' → use 'persistent'
        """
        self.id = id
        self.model_cfg = model
        self.prompt_env = prompt_env
        self.tools = tools or []
        self.policies = policies or {'max_steps': 24, 'retry': {'attempts': 2}}
        self.generation_config = generation_config or {}

        # RLM configuration
        self.rlm_config = None
        if rlm:
            from .rlm import RlmConfig
            if isinstance(rlm, dict):
                self.rlm_config = RlmConfig(**rlm)
            else:
                self.rlm_config = rlm

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

        # Provider setup: If model config has api_key, create provider instance directly
        # Otherwise, use shared registry (backward compatible)
        self.providers = ProviderRegistry.default()
        self._custom_provider = None

        # Check if model config has api_key - if so, create custom provider instance
        if 'api_key' in self.model_cfg:
            provider_name = self.model_cfg['provider']
            api_key = self.model_cfg['api_key']

            # Import provider classes
            from .providers import OpenAIProvider, OpenAIResponsesProvider, GeminiProvider, AnthropicProvider

            # Create provider instance with API key
            if provider_name == 'openai':
                self._custom_provider = OpenAIProvider(api_key=api_key)
            elif provider_name == 'openai-responses':
                self._custom_provider = OpenAIResponsesProvider(api_key=api_key)
            elif provider_name == 'google':
                self._custom_provider = GeminiProvider(api_key=api_key)
            elif provider_name == 'anthropic':
                self._custom_provider = AnthropicProvider(api_key=api_key)
            else:
                raise ValueError(f"Unknown provider: {provider_name}. Cannot create instance with custom API key.")

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

    def _get_provider(self):
        """Get provider instance (custom or from registry)"""
        if self._custom_provider:
            return self._custom_provider
        return self.providers.get(self.model_cfg['provider'])

    async def _call_llm_generate(self, run_id: str, session_id: Optional[str] = None, generation_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Non-streaming LLM call"""
        messages = self.ctxmgr.messages_for_llm(run_id, session_id)
        provider = self._get_provider()
        # Merge agent-level and call-level generation configs
        config = {**self.generation_config, **(generation_config or {})}
        step = await provider.generate(messages, model=self.model_cfg['model'], tools=self.toolreg.schemas(), generation_config=config)
        return step

    async def _call_tool(self, tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a tool"""
        tool = self.toolreg.get(tool_name)
        validate_io(tool.input_schema, args)
        result = await tool.run(args, ctx={})
        validate_io(tool.output_schema, result)
        return result

    async def run(
        self,
        input: Dict[str, Any],
        session_id: Optional[str] = None,
        generation_config: Optional[Dict[str, Any]] = None,
        context_refs: Optional[Any] = None,
        rlm: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Non-streaming run - returns final answer only.

        Args:
            input: Input dict with either:
                   - 'message': str - Single message (Vel manages history via session_id)
                   - 'messages': List[Dict] - Full conversation history (stateless, client-managed)

                   Examples:
                   - {'message': 'Hello'} - Session-based (use with session_id)
                   - {'messages': [{'role': 'user', 'content': 'Hello'}]} - Stateless

            session_id: Optional session ID for multi-turn conversations.
                       Only used with 'message' input (ignored when 'messages' provided).
                       If provided, context persists across multiple run() calls.

            generation_config: Optional per-run generation config that overrides agent-level config.
            context_refs: Optional context references for RLM (large documents, files, URLs)
            rlm: Optional per-run RLM config that overrides agent-level config
        """
        # Check if RLM is enabled (per-run override or agent-level config)
        rlm_config = None
        if rlm and rlm.get('enabled'):
            from .rlm import RlmConfig
            rlm_config = RlmConfig(**rlm) if isinstance(rlm, dict) else rlm
        elif self.rlm_config and self.rlm_config.enabled:
            rlm_config = self.rlm_config

        # If RLM is enabled and we have context, route to RLM controller
        if rlm_config and context_refs:
            from .rlm import RlmController

            controller = RlmController(config=rlm_config, agent=self)
            result = await controller.run(
                user_query=input.get('message', str(input)),
                context_refs=context_refs,
                session_id=session_id
            )
            return result['answer']

        run_id = str(uuid.uuid4())
        self.ctxmgr.set_input(run_id, input, session_id)
        state = State(run_id=run_id)
        event: Dict[str, Any] = {'kind':'start'}
        steps = 0
        final_answer = ''

        try:
            while True:
                state, effects = reduce(state, event)
                for eff in effects:
                    if eff.kind == 'call_llm':
                        step = await self._call_llm_generate(run_id, session_id, generation_config)
                        event = {'kind':'llm_step', 'step': step}
                        break
                    elif eff.kind == 'call_tool':
                        result = await self._call_tool(eff.payload['tool'], eff.payload.get('args', {}))
                        # Add tool result to context
                        self.ctxmgr.append_tool_result(run_id, eff.payload['tool'], result, session_id)
                        event = {'kind':'tool_result', 'result': result}
                        break
                    elif eff.kind == 'halt':
                        final_answer = eff.payload.get('final','')
                        # Add assistant response to context
                        self.ctxmgr.append_assistant_message(run_id, final_answer, session_id)
                        return final_answer
                steps += 1
                if steps > self.policies.get('max_steps', 24):
                    msg = 'max steps exceeded'
                    raise RuntimeError(msg)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            # Log detailed error information
            error_type = type(e).__name__
            logger.error(f"Agent run failed: {error_type}: {str(e)}", exc_info=True)
            raise

    async def run_stream(
        self,
        input: Dict[str, Any],
        session_id: Optional[str] = None,
        generation_config: Optional[Dict[str, Any]] = None,
        context_refs: Optional[Any] = None,
        rlm: Optional[Dict[str, Any]] = None
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Streaming run - yields stream protocol events as they occur.

        Args:
            input: Input dict with either:
                   - 'message': str - Single message (Vel manages history via session_id)
                   - 'messages': List[Dict] - Full conversation history (stateless, client-managed)

                   Examples:
                   - {'message': 'Hello'} - Session-based (use with session_id)
                   - {'messages': [{'role': 'user', 'content': 'Hello'}]} - Stateless

            session_id: Optional session ID for multi-turn conversations.
                       Only used with 'message' input (ignored when 'messages' provided).
                       If provided, context persists across multiple run_stream() calls.

            generation_config: Optional per-run generation config that overrides agent-level config.
            context_refs: Optional context references for RLM (large documents, files, URLs)
            rlm: Optional per-run RLM config that overrides agent-level config
        """
        # Check if RLM is enabled (per-run override or agent-level config)
        rlm_config = None
        if rlm and rlm.get('enabled'):
            from .rlm import RlmConfig
            rlm_config = RlmConfig(**rlm) if isinstance(rlm, dict) else rlm
        elif self.rlm_config and self.rlm_config.enabled:
            rlm_config = self.rlm_config

        # If RLM is enabled and we have context, route to RLM controller
        if rlm_config and context_refs:
            from .rlm import RlmController

            controller = RlmController(config=rlm_config, agent=self)
            async for event in controller.run_stream(
                user_query=input.get('message', str(input)),
                context_refs=context_refs,
                session_id=session_id
            ):
                yield event
            return

        run_id = str(uuid.uuid4())
        self.ctxmgr.set_input(run_id, input, session_id)

        # Emit start event (V5 UI Stream Protocol)
        yield StartEvent().to_dict()

        steps = 0
        max_steps = self.policies.get('max_steps', 24)

        try:
            while steps < max_steps:
                steps += 1

                # Emit start-step event (V5 UI Stream Protocol for multi-step agents)
                yield StepStartEvent().to_dict()

                # Get messages and stream LLM response
                messages = self.ctxmgr.messages_for_llm(run_id, session_id)
                provider = self._get_provider()

                # Merge agent-level and per-run generation configs
                config = {**self.generation_config, **(generation_config or {})}

                # Track what happened during streaming
                full_text = []
                tool_calls = []  # list of {tool_call_id, tool_name, input}
                finish_reason = 'stop'
                usage = None
                response_metadata = None

                # Stream from provider and forward events
                async for event in provider.stream(messages, model=self.model_cfg['model'], tools=self.toolreg.schemas(), generation_config=config):
                    # Track metadata for finish events (don't forward finish-message)
                    if event.type == 'finish-message':
                        finish_reason = event.finish_reason
                        continue  # Don't forward, consume internally

                    # Track response metadata (usage, model info)
                    # AI SDK v5 parity: Consume internally, don't forward
                    elif event.type == 'response-metadata':
                        if not response_metadata:
                            response_metadata = {}
                        # Update metadata (can come in multiple events)
                        if hasattr(event, 'id') and event.id:
                            response_metadata['id'] = event.id
                        if hasattr(event, 'model_id') and event.model_id:
                            response_metadata['modelId'] = event.model_id
                        if hasattr(event, 'timestamp') and event.timestamp:
                            response_metadata['timestamp'] = event.timestamp
                        if hasattr(event, 'usage') and event.usage:
                            usage = event.usage
                        continue  # Don't forward, consume internally

                    # Forward all other stream protocol events
                    yield event.to_dict()

                    # Track text content
                    if event.type == 'text-delta':
                        full_text.append(event.delta)

                    # Track tool calls (V5 UI Stream Protocol)
                    elif event.type == 'tool-input-available':
                        tool_calls.append({
                            'tool_call_id': event.tool_call_id,
                            'tool_name': event.tool_name,
                            'input': event.input
                        })

                    # Handle errors
                    elif event.type == 'error':
                        # Log detailed error information automatically
                        error_context = {
                            'error': event.error,
                            'provider': getattr(event, 'provider', 'unknown'),
                            'error_type': getattr(event, 'error_type', None),
                            'error_code': getattr(event, 'error_code', None),
                            'status_code': getattr(event, 'status_code', None)
                        }
                        logger.error(f"Agent error: {error_context}")

                        # Yield the full error event (includes all context)
                        yield event.to_dict()
                        yield {'type': 'finish'}
                        return

                # If we got text and no tool calls, we're done
                if full_text and not tool_calls:
                    answer = ''.join(full_text)
                    self.ctxmgr.append_assistant_message(run_id, answer, session_id)

                    # Emit finish-step event (AI SDK v5 spec: simple event, no fields)
                    yield {'type': 'finish-step'}

                    # Emit finish event (AI SDK v5 spec: simple event, no fields)
                    yield {'type': 'finish'}
                    return

                # If we got tool calls, execute them and continue
                if tool_calls:
                    for tc in tool_calls:
                        try:
                            # Execute tool
                            result = await self._call_tool(tc['tool_name'], tc['input'])

                            # Emit tool output event (V5 UI Stream Protocol)
                            output_event = ToolOutputAvailableEvent(
                                tool_call_id=tc['tool_call_id'],
                                output=result
                            )
                            yield output_event.to_dict()

                            # Add to context for next iteration
                            self.ctxmgr.append_tool_result(run_id, tc['tool_name'], result, session_id)

                        except Exception as e:
                            error_event = ErrorEvent(error=f"Tool execution failed: {str(e)}")
                            yield error_event.to_dict()
                            yield {'type': 'finish'}
                            return

                    # Emit finish-step event (AI SDK v5 spec: simple event, no fields)
                    yield {'type': 'finish-step'}

                    # Continue loop to get next LLM response
                    continue

                # If we got here with no text and no tool calls, something's wrong
                error_event = ErrorEvent(error='No response from LLM')
                yield error_event.to_dict()
                yield {'type': 'finish'}
                return

            # Max steps exceeded
            msg = f'max steps ({max_steps}) exceeded'
            error_event = ErrorEvent(error=msg)
            yield error_event.to_dict()
            yield {'type': 'finish'}

        except asyncio.CancelledError:
            raise
        except Exception as e:
            error_event = ErrorEvent(error=str(e))
            yield error_event.to_dict()
            raise

async def run_stream(agent: 'Agent', input: Dict[str, Any]):
    """Helper function for streaming"""
    async for e in agent.run_stream(input):
        yield e
