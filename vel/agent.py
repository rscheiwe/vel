from __future__ import annotations
import asyncio
from contextlib import aclosing
import json
import re
import time
import warnings
import logging
import uuid
from typing import Any, Callable, AsyncGenerator, Dict, List, Optional, Literal, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from .integrations.langfuse import ObservabilityConfig
    from .integrations.base import ObservabilityHandler, SpanContext

# Configure logger for error surfacing
logger = logging.getLogger('vel.agent')
from .core import State, reduce, ContextManager
from .core.tool_behavior import (
    ToolUseBehavior, ToolUseDecision, ToolEvent, ToolUseDirective, HandoffConfig
)
from .core.guardrails import GuardrailEngine, GuardrailError
from .core.structured_output import (
    StructuredOutputPolicy, StructuredOutputValidationError,
    parse_structured_output, get_retry_prompt, get_json_mode_system_prompt,
    to_strict_response_format,
)
from .core.hooks import (
    HookRegistry, RunStartHookEvent, RunFinallyHookEvent,
    StepStartHookEvent, StepEndHookEvent, ToolCallHookEvent, ToolResultHookEvent,
    LLMRequestHookEvent, LLMResponseHookEvent, FinishHookEvent, ErrorHookEvent
)
from .providers import ProviderRegistry
from .tools import ToolRegistry, ToolSpec, validate_io
from .events import (
    StreamEvent, StartEvent, FinishEvent, ToolInputAvailableEvent, ToolOutputAvailableEvent,
    ToolOutputErrorEvent, AbortEvent,
    TextEndEvent, ReasoningEndEvent,
    ErrorEvent, FinishMessageEvent, StepStartEvent, StepFinishEvent,
    ObjectElementEvent, ObjectPartialEvent, ObjectCompleteEvent,
    EventMetadata, add_metadata
)
from .core.json_stream_parser import (
    IncrementalJsonParser, detect_output_mode, get_element_type, OutputMode,
    StreamedElement, PartialObject
)
from .prompts import PromptContextManager, PromptTemplate

class _OpenStreamState:
    """Tracks what a run has left open, so a cancelled run can still be closed.

    A cancelled stream must remain well-formed. Cancelling mid-answer leaves a
    text block open; mid-thought leaves a reasoning block open; mid-tool leaves
    a tool call that a client will render as a spinner forever. Clients balance
    strictly by id, so every one of those has to be closed before the terminal
    event or the UI is stuck on a run that has already stopped.

    Observing the emitted events is the only reliable place to know this — the
    inner layers each know about their own blocks, but nothing else sees all of
    them in one place.
    """

    def __init__(self) -> None:
        self.text: List[str] = []
        self.reasoning: List[str] = []
        self.tools: List[str] = []
        self.open_steps = 0

    def observe(self, event: Dict[str, Any]) -> None:
        if not isinstance(event, dict):
            return
        etype = event.get('type')
        if etype == 'text-start':
            self.text.append(event.get('id'))
        elif etype == 'text-end':
            self._drop(self.text, event.get('id'))
        elif etype == 'reasoning-start':
            self.reasoning.append(event.get('id'))
        elif etype == 'reasoning-end':
            self._drop(self.reasoning, event.get('id'))
        elif etype in ('tool-input-available', 'tool-input-start'):
            tid = event.get('toolCallId')
            if tid is not None and tid not in self.tools:
                self.tools.append(tid)
        elif etype in ('tool-output-available', 'tool-output-error'):
            self._drop(self.tools, event.get('toolCallId'))
        elif etype == 'start-step':
            self.open_steps += 1
        elif etype == 'finish-step':
            self.open_steps = max(0, self.open_steps - 1)

    @staticmethod
    def _drop(bucket: List[str], value: Any) -> None:
        if value in bucket:
            bucket.remove(value)

    def closing_events(self, reason: Optional[str] = None) -> List[Dict[str, Any]]:
        """The terminal sequence for a cancelled run, in the required order.

        Blocks first, then any in-flight tool, then the step, then `abort`, then
        `finish`. A cancelled tool is reported as `tool-output-error` rather than
        left dangling — the client needs a terminal event keyed to the same id,
        and inventing a successful output would be a lie.
        """
        events: List[Dict[str, Any]] = []
        for block_id in list(self.text):
            events.append(TextEndEvent(block_id=block_id).to_dict())
        for block_id in list(self.reasoning):
            events.append(ReasoningEndEvent(block_id=block_id).to_dict())
        for tool_call_id in list(self.tools):
            events.append(ToolOutputErrorEvent(
                tool_call_id=tool_call_id,
                error_text=reason or 'Run cancelled before this tool returned',
            ).to_dict())
        for _ in range(self.open_steps):
            events.append({'type': 'finish-step'})
        events.append(AbortEvent(reason=reason).to_dict())
        events.append({'type': 'finish'})

        self.text.clear()
        self.reasoning.clear()
        self.tools.clear()
        self.open_steps = 0
        return events


async def _empty_async_iter():
    """Yields nothing. Lets the prefetched branch reuse the same `async for`."""
    return
    yield  # pragma: no cover - unreachable, marks this an async generator


class Agent:
    def __init__(self, id: str, model: Dict[str, Any], prompt_env: str='prod',
                 tools: List[Union[str, 'ToolSpec']]|None=None, policies: Dict[str, Any]|None=None,
                 context_manager: Optional[ContextManager]=None,
                 session_persistence: Optional[Literal['transient', 'persistent']]=None,
                 prompt_id: Optional[str]=None,
                 prompt_vars: Optional[Dict[str, Any]]=None,
                 prompt: Optional['PromptTemplate']=None,
                 generation_config: Optional[Dict[str, Any]]=None,
                 rlm: Optional[Dict[str, Any]]=None,
                 thinking: Optional[Any]=None,  # ThinkingConfig for extended thinking
                 harness: Optional[Any]=None,  # HarnessConfig or dict for Harness Mode (opt-in)
                 tool_context: Optional[Dict[str, Any]]=None,
                 # Guardrails
                 input_guardrails: Optional[List]=None,
                 output_guardrails: Optional[List]=None,
                 tool_guardrails: Optional[Dict[str, List]]=None,
                 # Structured output
                 output_type: Optional[type]=None,
                 structured_output_policy: Optional[StructuredOutputPolicy]=None,
                 # Lifecycle hooks
                 hooks: Optional[Dict[str, Any]]=None,
                 # Dynamic instructions
                 instruction: Optional[Any]=None,  # str or Callable[[dict], str]
                 # Direct system prompt (takes priority over prompt/prompt_id)
                 system_prompt: Optional[Any]=None,  # str or Callable[[dict], str]
                 # Scratchpad for working memory
                 scratchpad: Optional[Any]=None,  # bool, dict, or ScratchpadConfig
                 # Observability
                 observability: Optional[Union['ObservabilityConfig', Dict[str, Any]]]=None,
                 # Tool approval callback (for CLI/TUI approval flows)
                 tool_approval_callback: Optional[Any]=None,  # Callable[[str, Dict], Awaitable[bool]]
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
            tools: List of tools to enable. Can be:
                - str: Tool name (looked up in global registry)
                - ToolSpec: Tool instance (used directly, no registration required)
                Example: tools=['websearch', ToolSpec.from_function(my_func)]
            policies: Execution policies dictionary. Options:
                - max_steps: int (default: 24) - Maximum execution steps
                - retry: dict - Retry configuration
                - stop_on_first_tool: bool (default: False) - Halt after any tool execution
                - tool_behavior: dict - Per-tool configuration
                    Example: {'tool-a': {'stop_on_first_use': True}}
                    When a tool has 'stop_on_first_use': True, execution halts after that
                    specific tool runs, returning raw tool output instead of LLM response.
                - tool_use_behavior: ToolUseBehavior enum - Control flow after tool execution
                    - RUN_LLM_AGAIN (default): Continue to next LLM call
                    - STOP_AFTER_TOOL: Stop after any tool executes
                    - STOP_AT_TOOLS: Stop when tools in stop_at_tools list execute
                    - CUSTOM_HANDLER: Use custom_tool_handler callback
                - stop_at_tools: List[str] - Tool names that halt execution (with STOP_AT_TOOLS)
                - custom_tool_handler: Callable[[ToolEvent], ToolUseDecision|ToolUseDirective]
                - reset_tool_choice: bool (default: False) - Add prompt to prevent tool loops

            context_manager: Custom context manager instance. Pass:
                - None or ContextManager() for default (full message history)
                - StatelessContextManager() for no message history
                - ContextManager(max_history=10) for limited message history
                - Your own custom ContextManager subclass

            session_persistence: Where message history is saved:
                - 'transient': In-memory only (default, fast, not persistent)
                - 'persistent': Database-backed (survives restarts, requires PostgreSQL)
                - None: defaults to 'transient'

            prompt_id: Optional prompt template ID to look up in registry (legacy approach)
            prompt_vars: Optional variables for prompt template rendering
            prompt: Optional PromptTemplate instance to use directly (preferred approach).
                    When provided, no registry lookup is needed. Example:
                    - prompt=PromptTemplate(id='my-agent', system='You are {{role}}')
                    Note: If both `prompt` and `prompt_id` are provided, `prompt` takes precedence.

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

            thinking: Extended Thinking configuration (ThinkingConfig or dict).
                Enables multi-pass reasoning (Analyze -> Critique -> Refine -> Conclude).
                Example: ThinkingConfig(mode='reflection', max_refinements=3)
                See vel.thinking.ThinkingConfig for options.

            tool_context: Optional context dict to pass to all tool handlers via ctx parameter.
                Useful for passing shared resources like storage backends, database connections, etc.
                Example: {'storage': MessageBasedStorage(messages)}

            input_guardrails: List of async guardrail functions to validate user input.
                Signature: async def guardrail(content, ctx) -> GuardrailResult | bool
                Example: [validate_no_pii, require_min_length]

            output_guardrails: List of async guardrail functions to validate LLM output.
                Signature: async def guardrail(content, ctx) -> GuardrailResult | bool
                Example: [must_be_json, no_harmful_content]

            tool_guardrails: Dict mapping tool names to their guardrail functions.
                Example: {'get_weather': [validate_location]}

            output_type: Pydantic model class for structured output validation.
                When set, agent will force JSON mode and validate/retry output.
                Example: output_type=WeatherResponse

            structured_output_policy: Policy for handling validation failures.
                Default: StructuredOutputPolicy(max_retries=1, on_failure="raise")
                Example: StructuredOutputPolicy(max_retries=2, on_failure="return_raw")

            hooks: Dict of lifecycle hook handlers for observability and tracing.
                Supported hooks: on_step_start, on_step_end, on_tool_call, on_tool_result,
                on_llm_request, on_llm_response, on_finish, on_error
                Example: {'on_tool_call': my_tool_logger, 'on_error': my_error_handler}

            instruction: Dynamic system instruction, can be string or callable.
                If callable, evaluated per-run with context dict: (ctx) -> str
                Example: lambda ctx: f"User tier: {ctx.get('user_tier', 'free')}"

            system_prompt: Direct system prompt string or callable. Takes priority over
                prompt/prompt_id templates. Useful for dynamic prompt building:
                - str: Used directly as system prompt
                - Callable[[dict], str]: Called with context dict from run()
                Example: lambda ctx: f"You are {ctx.get('role')}. Skills: {ctx.get('skill')}"
                Priority: system_prompt > prompt template > default

            scratchpad: Scratchpad configuration for ephemeral working memory.
                Enables agents to maintain context during multi-step tool execution.
                Can be:
                - True: Use default ScratchpadConfig
                - dict: Configuration options (max_entries, summary_max_chars, etc.)
                - ScratchpadConfig: Explicit config object
                Example: scratchpad=True or scratchpad=ScratchpadConfig(max_entries=50)

            observability: Observability configuration for tracing and monitoring.
                Can be:
                - None: Observability disabled (default)
                - ObservabilityConfig: Full configuration object
                - dict: Configuration dict (converted to ObservabilityConfig)

                Example:
                    observability=ObservabilityConfig(
                        provider='langfuse',
                        user_id='user-123',
                        tags=['production']
                    )

            session_storage: [DEPRECATED] Use session_persistence instead
                - 'memory' → use 'transient'
                - 'database' → use 'persistent'
        """
        self.id = id
        self.model_cfg = model
        self.prompt_env = prompt_env
        self.policies = policies or {'max_steps': 24, 'retry': {'attempts': 2}}
        self.generation_config = generation_config or {}
        self.tool_context = tool_context or {}

        # Normalize tools: support both strings (global registry) and ToolSpec instances
        self._instance_tools: Dict[str, ToolSpec] = {}  # Instance-level tools
        self._tool_names: List[str] = []  # All tool names (for schema filtering)
        self._injected_tools: Dict[str, ToolSpec] = {}  # Dynamically injected tools (per-run)

        for tool in (tools or []):
            if isinstance(tool, str):
                # String: reference to global registry (DEPRECATED)
                warnings.warn(
                    f"Passing tool names as strings is deprecated and will be removed in Vel v2.0. "
                    f"Pass ToolSpec instances directly instead:\n"
                    f"  tool = ToolSpec.from_function(your_function)\n"
                    f"  agent = Agent(tools=[tool])\n"
                    f"See examples/dynamic_tools.py for migration examples.",
                    DeprecationWarning,
                    stacklevel=2
                )
                self._tool_names.append(tool)
            elif isinstance(tool, ToolSpec):
                # ToolSpec instance: store directly (no registration needed!)
                self._instance_tools[tool.name] = tool
                self._tool_names.append(tool.name)
            else:
                raise TypeError(
                    f"Invalid tool type: {type(tool).__name__}. "
                    f"Expected str or ToolSpec. "
                    f"Use ToolSpec.from_function(fn) to wrap functions."
                )

        # Guardrails engine
        self.guardrails = GuardrailEngine(
            input_guardrails=input_guardrails,
            output_guardrails=output_guardrails,
            tool_guardrails=tool_guardrails
        )

        # Structured output
        self.output_type = output_type
        self.structured_output_policy = structured_output_policy or StructuredOutputPolicy()

        # When an output_type is set on an OpenAI model, use the provider's native
        # Structured Outputs (response_format json_schema, strict) so the API
        # guarantees schema-conforming JSON via constrained decoding. This removes
        # the prompt-and-parse-and-retry failure modes (empty/non-JSON output,
        # shape mismatches) at the source. Skipped for non-object root types and
        # if a response_format was already provided explicitly.
        if output_type is not None and self.model_cfg.get('provider') == 'openai' \
                and 'response_format' not in self.generation_config:
            # Prefer strict Structured Outputs (schema-guaranteed). When the schema
            # can't be made strict (free-form dict fields, array root), fall back to
            # JSON mode, which still guarantees syntactically valid JSON.
            response_format = to_strict_response_format(output_type) or {'type': 'json_object'}
            self.generation_config = {**self.generation_config, 'response_format': response_format}

        # Lifecycle hooks
        self.hooks = HookRegistry(hooks)

        # Dynamic instructions
        self.instruction = instruction

        # Direct system prompt (takes priority over prompt templates)
        self._system_prompt = system_prompt

        # RLM configuration
        self.rlm_config = None
        if rlm:
            from .rlm import RlmConfig
            if isinstance(rlm, dict):
                self.rlm_config = RlmConfig(**rlm)
            else:
                self.rlm_config = rlm

        # Extended Thinking configuration
        self.thinking_config = None
        if thinking:
            from .thinking import ThinkingConfig
            if isinstance(thinking, dict):
                self.thinking_config = ThinkingConfig(**thinking)
            else:
                self.thinking_config = thinking

        # Harness Mode configuration (opt-in; default-off bolt-on)
        self.harness_config = None
        if harness:
            from .harness import HarnessConfig
            if isinstance(harness, dict):
                self.harness_config = HarnessConfig(**harness)
            else:
                self.harness_config = harness

        # Scratchpad configuration
        self._scratchpad_config = None
        self._scratchpad_summary: Optional[str] = None
        if scratchpad:
            from .tools.scratchpad import ScratchpadConfig
            if isinstance(scratchpad, bool):
                self._scratchpad_config = ScratchpadConfig()
            elif isinstance(scratchpad, dict):
                self._scratchpad_config = ScratchpadConfig(**scratchpad)
            else:
                self._scratchpad_config = scratchpad

        # Observability configuration
        self._observability_config: Optional['ObservabilityConfig'] = None
        self._observer: Optional['ObservabilityHandler'] = None
        if observability:
            from .integrations.langfuse import ObservabilityConfig, build_handler
            if isinstance(observability, dict):
                self._observability_config = ObservabilityConfig(**observability)
            else:
                self._observability_config = observability
            # Build handler if enabled
            if self._observability_config.enabled and self._observability_config.provider != 'none':
                self._observer = build_handler(self._observability_config, self.id)

        # Tool approval callback for CLI/TUI approval flows
        # Signature: async def callback(tool_name: str, tool_args: Dict) -> bool
        # Returns True to approve, False to deny
        self._tool_approval_callback = tool_approval_callback

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

        # Check if model config has api_key or base_url - if so, create custom
        # provider instance. `base_url` is honored for the OpenAI-shaped
        # providers only, since it is what lets one process reach an
        # OpenAI-compatible gateway without hijacking OPENAI_API_BASE for every
        # other agent in the same process.
        if 'api_key' in self.model_cfg or 'base_url' in self.model_cfg:
            provider_name = self.model_cfg['provider']
            api_key = self.model_cfg.get('api_key')
            base_url = self.model_cfg.get('base_url')

            # Import provider classes
            from .providers import OpenAIProvider, OpenAIResponsesProvider, GeminiProvider, AnthropicProvider

            # Create provider instance with API key
            if provider_name == 'openai':
                self._custom_provider = OpenAIProvider(api_key=api_key, base_url=base_url)
            elif provider_name == 'openai-responses':
                self._custom_provider = OpenAIResponsesProvider(api_key=api_key, base_url=base_url)
            elif provider_name == 'google':
                self._custom_provider = GeminiProvider(api_key=api_key)
            elif provider_name == 'anthropic':
                self._custom_provider = AnthropicProvider(api_key=api_key)
            else:
                raise ValueError(f"Unknown provider: {provider_name}. Cannot create instance with custom API key.")

            if base_url and provider_name not in ('openai', 'openai-responses'):
                raise ValueError(
                    f"base_url is only supported for OpenAI-compatible providers, not '{provider_name}'."
                )

        self.toolreg = ToolRegistry.default()

        # Context manager setup with prompt support
        if context_manager is not None:
            # User provided custom context manager - use as-is
            self.ctxmgr = context_manager
        elif prompt is not None:
            # Dynamic prompt template provided - use PromptContextManager (preferred)
            self.ctxmgr = PromptContextManager(
                prompt=prompt,
                prompt_vars=prompt_vars,
                prompt_env=prompt_env
            )
        elif prompt_id:
            # Prompt template ID provided - look up in registry (legacy)
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

    def _get_system_prompt(self, run_id: str, context: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """
        Get the rendered system prompt for a run.

        Priority order:
        1. Direct system_prompt (string or callable) - highest priority
        2. PromptContextManager template - if system_prompt not set
        3. None - provider default

        Args:
            run_id: The run identifier
            context: Optional runtime context dict for callable system_prompt

        Returns:
            Rendered system prompt string, or None if no prompt configured
        """
        context = context or {}

        # Priority 1: Direct system_prompt
        if self._system_prompt is not None:
            if callable(self._system_prompt):
                return self._system_prompt(context)
            return self._system_prompt

        # Priority 2: Template via PromptContextManager
        if hasattr(self.ctxmgr, 'get_rendered_system_prompt'):
            return self.ctxmgr.get_rendered_system_prompt()

        # Priority 3: None (provider default)
        return None

    def get_system_prompt(self, context: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """
        Get the resolved system prompt for external inspection.

        Public method for consumers (like Valis/Mesh) to retrieve the
        system prompt that would be used for a run.

        Args:
            context: Optional runtime context dict for callable system_prompt

        Returns:
            Resolved system prompt string, or None if no prompt configured
        """
        return self._get_system_prompt(run_id='', context=context)

    def _get_tool(self, name: str) -> ToolSpec:
        """
        Get tool by name from injected tools, instance tools, or global registry.

        IMPORTANT: Only tools explicitly passed to this agent (via tools array)
        or dynamically injected during the run are accessible. Global registry
        is only used to resolve string references from the tools array.

        Args:
            name: Tool name

        Returns:
            ToolSpec instance

        Raises:
            KeyError: If tool not found or not authorized for this agent
        """
        # Check injected tools first (highest precedence - dynamically added during run)
        if name in self._injected_tools:
            return self._injected_tools[name]

        # Check instance tools (ToolSpec instances passed directly)
        if name in self._instance_tools:
            return self._instance_tools[name]

        # Check if tool was passed by name (string) in tools array
        # Only allow global registry lookup if the tool was explicitly listed
        if name in self._tool_names:
            return self.toolreg.get(name)

        # Tool not authorized for this agent
        raise KeyError(
            f"Tool '{name}' not found. This agent only has access to: {self._tool_names}. "
            f"If you need this tool, add it to the tools array when creating the agent."
        )

    def _get_tool_schemas(self) -> Dict[str, Any]:
        """
        Get schemas for all tools (injected + instance + global registry).

        Returns:
            Dict mapping tool names to their schemas
        """
        schemas = {}

        # Collect all tool names including injected tools
        all_tool_names = list(self._tool_names) + [
            name for name in self._injected_tools if name not in self._tool_names
        ]

        for tool_name in all_tool_names:
            if tool_name in self._injected_tools:
                # Injected tool (highest precedence)
                tool = self._injected_tools[tool_name]
                schemas[tool_name] = {
                    'input': tool.input_schema,
                    'output': tool.output_schema,
                    'description': tool.description
                }
            elif tool_name in self._instance_tools:
                # Instance tool
                tool = self._instance_tools[tool_name]
                schemas[tool_name] = {
                    'input': tool.input_schema,
                    'output': tool.output_schema,
                    'description': tool.description
                }
            else:
                # Global registry tool
                schemas.update(
                    self.toolreg.schemas(self.tool_context, filter_tools=[tool_name])
                )

        return schemas

    def as_tool(
        self,
        name: Optional[str] = None,
        description: Optional[str] = None,
        input_schema: Optional[Dict[str, Any]] = None,
        output_schema: Optional[Dict[str, Any]] = None,
        pass_context: bool = True,
        durable: bool = False
    ) -> 'ToolSpec':
        """
        Expose this agent as a tool that can be used by other agents.

        This enables hierarchical agent composition where an orchestrator agent
        can delegate tasks to specialized sub-agents.

        Args:
            name: Tool name (defaults to sanitized agent ID). Characters like
                ':', '-', '.' are replaced with '_' for LLM compatibility.
            description: Tool description shown to the orchestrator LLM.
            durable: If True AND this sub-agent has Harness Mode enabled, run it
                through the harness (budget, compaction, sandbox, per-step
                checkpointing) instead of the plain run() path. Approval is still
                forced inline because a tool call must return a value (a sub-agent
                cannot suspend the parent mid-step). Default False keeps the
                original non-durable behavior — fully backwards compatible.
            input_schema: Custom JSON schema for tool input. Defaults to
                {'message': string} if not provided.
            output_schema: Custom JSON schema for tool output. Defaults to
                empty (flexible output) if not provided.
            pass_context: If True (default), the parent agent's tool_context
                is merged into this agent's tool_context during execution.

        Returns:
            ToolSpec that wraps this agent

        Example:
            ```python
            researcher = Agent(id='researcher:v1', ...)
            orchestrator = Agent(
                id='orchestrator',
                tools=[
                    researcher.as_tool(
                        name='research_expert',
                        description='Delegate research tasks to the research agent.'
                    )
                ]
            )
            ```
        """
        from .tools import ToolSpec

        # Sanitize tool name: replace :, -, . with underscores
        raw_name = name or self.id
        tool_name = re.sub(r'[:\-.]', '_', raw_name)
        tool_desc = description or f"Run the {self.id} agent"

        # Default input schema: simple message-based interface
        default_input_schema = {
            'type': 'object',
            'properties': {
                'message': {
                    'type': 'string',
                    'description': f'Message to send to the {tool_name} agent'
                }
            },
            'required': ['message']
        }
        final_input_schema = input_schema or default_input_schema

        # Default output schema: empty = flexible (matches ToolSpec pattern)
        final_output_schema = output_schema or {}

        # Capture pass_context in closure
        _pass_context = pass_context
        _durable = durable
        _custom_input_schema = input_schema is not None

        # Create handler that calls this agent
        async def agent_tool_handler(input: Dict[str, Any], ctx: Dict[str, Any]) -> Dict[str, Any]:
            # Nesting: pull the parent trace context (if any) so this sub-agent
            # run attaches to the parent trace instead of forking a new one.
            _tc = ctx.get('_vel_trace_context')
            # Context passthrough: merge parent's ctx with sub-agent's tool_context
            original_context = self.tool_context
            if _pass_context:
                merged_context = {**self.tool_context, **ctx}
            else:
                merged_context = self.tool_context
            # Never leak the trace-context carrier into the sub-agent's own ctx.
            merged_context.pop('_vel_trace_context', None)

            self.tool_context = merged_context

            try:
                # Extract message based on schema type
                if _custom_input_schema:
                    # Custom schema: pass entire input as JSON message
                    if 'message' in input and isinstance(input['message'], str):
                        message = input['message']
                    else:
                        message = json.dumps(input)
                else:
                    # Default schema: extract 'message' or 'query' key
                    message = input.get('message', input.get('query', str(input)))

                # A tool call must return a value, so a sub-agent can never
                # suspend the parent mid-step: force approval.mode='inline' for
                # the duration (spec §6.9 / §12 Q7). This applies to both the
                # non-durable and durable paths below.
                _saved_approval_mode = None
                _hcfg = getattr(self, 'harness_config', None)
                _harness_on = _hcfg is not None and getattr(_hcfg, 'enabled', False)
                if _harness_on:
                    _saved_approval_mode = _hcfg.approval.mode
                    _hcfg.approval.mode = 'inline'

                # Run the agent. durable=True + Harness enabled -> run through the
                # harness loop (budget/compaction/sandbox/checkpointing) and
                # collect the streamed text as the tool result; otherwise use the
                # plain non-durable run() path (default, backwards compatible).
                try:
                    _nest_kwargs = {} if _tc is None else {'trace_context': _tc}
                    if _durable and _harness_on:
                        _parts: list = []
                        async for _ev in self.run_stream({'message': message}, **_nest_kwargs):
                            if _ev.get('type') == 'text-delta':
                                _parts.append(_ev.get('delta', ''))
                        result = ''.join(_parts)
                    else:
                        result = await self.run({'message': message}, **_nest_kwargs)
                finally:
                    if _saved_approval_mode is not None:
                        _hcfg.approval.mode = _saved_approval_mode

                # Return result as-is (no success wrapper for successful responses)
                if isinstance(result, str):
                    return {'response': result}
                elif isinstance(result, dict):
                    return result
                else:
                    # Pydantic model or other object
                    if hasattr(result, 'model_dump'):
                        return result.model_dump()
                    elif hasattr(result, 'dict'):
                        return result.dict()
                    else:
                        return {'response': str(result)}

            except Exception as e:
                # Only errors get wrapped with success: False
                return {
                    'success': False,
                    'error': str(e),
                    'error_type': type(e).__name__
                }
            finally:
                # Restore original context
                self.tool_context = original_context

        return ToolSpec(
            name=tool_name,
            input_schema=final_input_schema,
            output_schema=final_output_schema,
            handler=agent_tool_handler,
            description=tool_desc
        )

    async def _call_llm_generate(self, run_id: str, session_id: Optional[str] = None, generation_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Non-streaming LLM call"""
        messages = self.ctxmgr.messages_for_llm(run_id, session_id)
        provider = self._get_provider()
        # Merge agent-level and call-level generation configs
        config = {**self.generation_config, **(generation_config or {})}
        # Get schemas from instance tools + global registry
        tool_schemas = self._get_tool_schemas()
        step = await provider.generate(messages, model=self.model_cfg['model'], tools=tool_schemas, generation_config=config)
        return step

    async def _call_tool(self, tool_name: str, args: Dict[str, Any], trace_context: Optional['TraceContext'] = None) -> Dict[str, Any]:
        """Execute a tool (instance or global registry)"""
        # Get tool from instance registry or global registry
        tool = self._get_tool(tool_name)

        # Always validate input
        validate_io(tool.input_schema, args)

        # Execute tool. When observability is active, pass the parent trace
        # context so a sub-agent-as-tool nests under the parent trace.
        ctx = self.tool_context
        if trace_context is not None:
            ctx = {**self.tool_context, '_vel_trace_context': trace_context}
        result = await tool.run(args, ctx=ctx)

        # Only validate output if schema is non-empty (flexible by default)
        if tool.output_schema:
            validate_io(tool.output_schema, result)

        return result

    def should_stop_after_tool(self, tool_name: str) -> bool:
        """
        Check if execution should halt after this specific tool executes.

        Args:
            tool_name: Name of the tool that was executed

        Returns:
            True if execution should stop and return raw tool output,
            False if execution should continue normally
        """
        # Check new enum-based tool_use_behavior first
        behavior = self.policies.get('tool_use_behavior')
        if behavior:
            if behavior == ToolUseBehavior.STOP_AFTER_TOOL:
                return True
            elif behavior == ToolUseBehavior.STOP_AT_TOOLS:
                stop_at = self.policies.get('stop_at_tools', [])
                return tool_name in stop_at
            elif behavior == ToolUseBehavior.CUSTOM_HANDLER:
                return False  # Custom handler decides in process_tool_result
            elif behavior == ToolUseBehavior.RUN_LLM_AGAIN:
                return False

        # Check per-tool behavior (backwards compatible)
        tool_behaviors = self.policies.get('tool_behavior', {})
        if tool_name in tool_behaviors:
            return tool_behaviors[tool_name].get('stop_on_first_use', False)

        # Fall back to global setting (defaults to False)
        return self.policies.get('stop_on_first_tool', False)

    def _process_tool_result(
        self,
        tool_name: str,
        args: Dict[str, Any],
        result: Any,
        run_id: str,
        step: int,
        session_id: Optional[str] = None
    ) -> ToolUseDirective:
        """
        Process tool result through custom handler if configured.

        Returns ToolUseDirective with decision and any modifications.
        """
        behavior = self.policies.get('tool_use_behavior')

        # If custom handler is configured, call it
        if behavior == ToolUseBehavior.CUSTOM_HANDLER:
            handler = self.policies.get('custom_tool_handler')
            if handler:
                # Build ToolEvent
                messages = self.ctxmgr.messages_for_llm(run_id, session_id)
                event = ToolEvent(
                    tool_name=tool_name,
                    args=args,
                    output=result,
                    step=step,
                    messages=messages,
                    run_id=run_id,
                    session_id=session_id
                )

                # Call handler
                handler_result = handler(event)

                # Normalize to ToolUseDirective
                if isinstance(handler_result, ToolUseDecision):
                    return ToolUseDirective(decision=handler_result)
                elif isinstance(handler_result, ToolUseDirective):
                    return handler_result
                else:
                    # Assume it's a string decision
                    return ToolUseDirective(decision=ToolUseDecision(handler_result))

        # Default: continue
        return ToolUseDirective(decision=ToolUseDecision.CONTINUE)

    def _process_inject_tools(self, result: Any) -> List[str]:
        """
        Process tool output for inject_tools directive.

        If a tool returns an 'inject_tools' key in its output, those tools
        will be dynamically added to this agent's available tools for the
        remainder of the current run.

        Args:
            result: Tool output (dict or any)

        Returns:
            List of newly injected tool names (for logging/debugging)

        Example tool output:
            {
                "inject_tools": [
                    {
                        "name": "email_send",
                        "description": "Send an email",
                        "input_schema": {...},
                        "handler": "email_send"  # References globally registered handler
                    }
                ],
                "message": "Found 1 matching tool: email_send"
            }
        """
        if not isinstance(result, dict):
            return []

        inject_tools = result.get('inject_tools', [])
        if not inject_tools:
            return []

        injected_names = []

        for tool_def in inject_tools:
            name = tool_def.get('name')
            if not name:
                logger.warning("inject_tools entry missing 'name', skipping")
                continue

            # Skip if already available
            if name in self._injected_tools or name in self._instance_tools:
                logger.debug(f"Tool '{name}' already available, skipping injection")
                continue

            # Get handler - either from global registry or inline
            handler = None
            handler_ref = tool_def.get('handler')
            registered_tool = None  # Track for copying _unpack_args

            if handler_ref:
                # Handler reference - look up in global registry
                try:
                    registered_tool = self.toolreg.get(handler_ref)
                    handler = registered_tool._handler  # Note: _handler is the private attribute
                except KeyError:
                    logger.warning(f"Handler '{handler_ref}' not found in registry for tool '{name}'")
                    continue
            elif callable(tool_def.get('handler_fn')):
                # Inline handler function (less common)
                handler = tool_def['handler_fn']
            else:
                logger.warning(f"No handler specified for inject_tools entry '{name}'")
                continue

            # Create ToolSpec for injected tool
            # Preserve _unpack_args flag if the handler came from a registered tool
            unpack_args = getattr(registered_tool, '_unpack_args', False) if registered_tool else False
            injected_tool = ToolSpec(
                name=name,
                description=tool_def.get('description', ''),
                input_schema=tool_def.get('input_schema', {'type': 'object', 'properties': {}}),
                output_schema=tool_def.get('output_schema', {}),
                handler=handler,
                _unpack_args=unpack_args
            )

            self._injected_tools[name] = injected_tool
            injected_names.append(name)
            logger.info(f"Injected tool '{name}' for current run")

        return injected_names

    def _clear_injected_tools(self) -> None:
        """Clear all injected tools (called at start of new run)."""
        self._injected_tools.clear()

    def clear_scratchpad_context(self) -> None:
        """
        Clear stored scratchpad summary.

        Call when starting a new conversation or changing topics.
        The summary from the previous run will no longer be injected.
        """
        self._scratchpad_summary = None

    def _get_reset_tool_choice_message(self) -> Optional[Dict[str, Any]]:
        """
        Get system message to reset tool choice if enabled.

        Returns None if reset_tool_choice is not enabled.
        """
        if self.policies.get('reset_tool_choice', False):
            return {
                'role': 'system',
                'content': 'The previous tool did not resolve the request; reconsider tool selection.'
            }
        return None

    async def run(
        self,
        input: Dict[str, Any],
        session_id: Optional[str] = None,
        generation_config: Optional[Dict[str, Any]] = None,
        context_refs: Optional[Any] = None,
        rlm: Optional[Dict[str, Any]] = None,
        observability_context: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None,
        stateless: bool = False,
        trace_id: Optional[str] = None,
        trace_context: Optional['TraceContext'] = None,
    ) -> Union[str, Dict[str, Any]]:
        """
        Non-streaming run - returns final answer or raw tool output.

        trace_context: when provided (e.g. by a parent run via as_tool, or a
            coordinator that owns the trace), this run's spans/generations nest
            under the parent trace via the parent's handler instead of forking a
            new trace. trace_id optionally supplies the root trace id.

        Returns:
            - str: Final answer from LLM (default behavior)
            - Dict[str, Any]: Raw tool output if stop_on_first_tool policy is enabled

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
            observability_context: Optional per-run observability context overrides.
                Merges with agent-level ObservabilityConfig.
                Supported keys: user_id, session_id, tags, metadata, trace_name
            context: Optional runtime context dict for callable system_prompt.
                Passed to system_prompt(context) if system_prompt is callable.
                Separate from context_refs (which is for RLM document content).
            stateless: If True, skip session state mutation. Messages won't be
                saved to session history. Useful for Mesh/Valis integration where
                state is managed externally. Default: False.
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
        trace_id = trace_id or run_id
        run_start_time = time.time()
        self.ctxmgr.set_input(run_id, input, session_id, stateless=stateless)

        # Clear any injected tools from previous runs
        self._clear_injected_tools()

        # Setup observability trace. When nested (trace_context given), use the
        # PARENT's handler and open a child span under the parent trace; the child
        # inherits the parent's sampling decision and must NOT end/flush the trace.
        observer = trace_context.handler if trace_context else self._observer
        trace_ctx: Optional['SpanContext'] = None
        current_step_ctx: Optional['SpanContext'] = None
        if observer and (trace_context is not None or observer.should_sample()):
            from .integrations.base import SpanKind, GenerationData, ToolData
            # Merge observability_context with config. When nested under a parent
            # trace, this agent may have no observability config of its own — fall
            # back to sensible defaults so nesting still works.
            obs_config = self._observability_config
            if obs_config is not None and observability_context:
                obs_config = obs_config.with_context(**observability_context)
            _trace_name = (obs_config.trace_name if obs_config else None) or self.id
            _capture_input = obs_config.capture_input if obs_config else True
            _trace_input = input if _capture_input else {'message': '[redacted]'}

            if trace_context is not None:
                trace_ctx = observer.start_span(
                    trace_context.span,
                    _trace_name,
                    SpanKind.AGENT_RUN,
                    input=_trace_input,
                )
            else:
                trace_ctx = observer.start_trace(
                    trace_id=trace_id,
                    name=_trace_name,
                    input=_trace_input,
                    metadata=obs_config.metadata if obs_config else None,
                    tags=obs_config.tags if obs_config else None,
                    user_id=obs_config.user_id if obs_config else None,
                    session_id=(obs_config.session_id if obs_config else None) or session_id,
                )

            # Wire up observer to ContextManager for memory operation tracing
            self.ctxmgr.set_observer(observer, trace_ctx)

            # Emit run start hook
            await self.hooks.emit('on_run_start', RunStartHookEvent(
                run_id=run_id,
                session_id=session_id,
                input=input,
                agent_id=self.id
            ))

        # Setup scratchpad for this run
        scratchpad = None
        if self._scratchpad_config:
            from .tools.scratchpad import Scratchpad, get_scratchpad_tools
            scratchpad = Scratchpad(self._scratchpad_config)
            for tool in get_scratchpad_tools(scratchpad):
                self._injected_tools[tool.name] = tool
            # Inject previous run's summary into context
            if self._scratchpad_summary:
                self.ctxmgr._by_run[run_id].insert(0, {
                    'role': 'system',
                    'content': f"[Previous Run Context]\n{self._scratchpad_summary}"
                })

        # Add dynamic instruction if set
        if self.instruction:
            if callable(self.instruction):
                instruction_text = self.instruction(self.tool_context)
            else:
                instruction_text = self.instruction
            self.ctxmgr._by_run[run_id].insert(0, {'role': 'system', 'content': instruction_text})

        # Add structured output schema prompt if output_type is set
        if self.output_type:
            schema_prompt = get_json_mode_system_prompt(self.output_type)
            self.ctxmgr._by_run[run_id].insert(0, {'role': 'system', 'content': schema_prompt})

        # Run input guardrails
        if self.guardrails.has_input_guardrails:
            ctx = {'run_id': run_id, 'session_id': session_id}
            content = input.get('message', input)
            passed, modified, error = await self.guardrails.check_input(content, ctx)
            if not passed:
                raise GuardrailError('input', error, content)
            # If content was modified, update the input
            if modified != content and 'message' in input:
                input['message'] = modified
                self.ctxmgr.set_input(run_id, input, session_id, stateless=stateless)

        state = State(run_id=run_id)
        event: Dict[str, Any] = {'kind':'start'}
        steps = 0
        final_answer = ''
        structured_output_attempts = 0
        last_valid_output = None

        try:
            while True:
                # Structured-output validation retries restart the LLM call but
                # are bounded by StructuredOutputPolicy.max_retries, not the tool
                # max_steps budget. This flag lets us skip the step increment for
                # those retries (set in the structured-retry branch below).
                structured_retry = False
                state, effects = reduce(state, event)
                for eff in effects:
                    if eff.kind == 'call_llm':
                        # Get messages for LLM call (for observability)
                        messages_for_obs = self.ctxmgr.messages_for_llm(run_id, session_id)

                        # Start step span for observability
                        if trace_ctx and observer:
                            from .integrations.base import SpanKind, GenerationData
                            current_step_ctx = observer.start_span(
                                trace_ctx, f"step-{steps}", SpanKind.STEP,
                                input={'messages_count': len(messages_for_obs), 'step': steps}
                            )
                        llm_start_time = time.time()

                        step = await self._call_llm_generate(run_id, session_id, generation_config)

                        # Log LLM generation
                        if trace_ctx and observer:
                            llm_end_time = time.time()
                            llm_latency = (llm_end_time - llm_start_time) * 1000
                            gen_observation_id = observer.log_generation(
                                current_step_ctx or trace_ctx,
                                GenerationData(
                                    model=self.model_cfg.get('model', 'unknown'),
                                    provider=self.model_cfg.get('provider', 'unknown'),
                                    messages=messages_for_obs,
                                    response=step.get('answer'),
                                    tool_calls=[{'tool': step.get('tool'), 'args': step.get('args')}] if step.get('tool') else None,
                                    usage=step.get('usage'),
                                    generation_config=generation_config,
                                    latency_ms=llm_latency,
                                    start_time=llm_start_time,
                                    end_time=llm_end_time,
                                )
                            )
                            if self.hooks.has_hook('on_llm_response'):
                                await self.hooks.emit('on_llm_response', LLMResponseHookEvent(
                                    run_id=run_id, trace_id=trace_id, session_id=session_id, step=steps,
                                    response=step.get('answer'), tool_calls=[{'tool': step.get('tool'), 'args': step.get('args')}] if step.get('tool') else None,
                                    usage=step.get('usage'), observation_id=gen_observation_id,
                                    duration_ms=llm_latency,
                                ))

                        event = {'kind':'llm_step', 'step': step}
                        break
                    elif eff.kind == 'call_tool':
                        tool_name = eff.payload['tool']
                        tool_args = eff.payload.get('args', {})

                        # Run tool guardrails
                        if self.guardrails.has_tool_guardrails(tool_name):
                            ctx = {'run_id': run_id, 'session_id': session_id, 'tool_name': tool_name}
                            passed, modified_args, error = await self.guardrails.check_tool(tool_name, tool_args, ctx)
                            if not passed:
                                raise GuardrailError(f'tool:{tool_name}', error, tool_args)
                            tool_args = modified_args

                        # Track tool execution time
                        tool_start_time = time.time()
                        tool_error = None
                        _tool_tc = None
                        if trace_ctx and observer:
                            from .integrations.base import TraceContext
                            _tool_tc = TraceContext(handler=observer, span=current_step_ctx or trace_ctx)
                        try:
                            result = await self._call_tool(tool_name, tool_args, trace_context=_tool_tc)
                        except Exception as e:
                            tool_error = str(e)
                            raise
                        finally:
                            # Log tool execution
                            if trace_ctx and observer:
                                from .integrations.base import ToolData
                                tool_latency = (time.time() - tool_start_time) * 1000
                                tool_observation_id = observer.log_tool(
                                    current_step_ctx or trace_ctx,
                                    ToolData(
                                        tool_name=tool_name,
                                        input=tool_args,
                                        output=result if not tool_error else None,
                                        error=tool_error,
                                        latency_ms=tool_latency,
                                    )
                                )
                                if self.hooks.has_hook('on_tool_result'):
                                    await self.hooks.emit('on_tool_result', ToolResultHookEvent(
                                        run_id=run_id, trace_id=trace_id, session_id=session_id, step=steps,
                                        tool_name=tool_name, result=result if not tool_error else None,
                                        error=tool_error, observation_id=tool_observation_id,
                                        duration_ms=tool_latency,
                                    ))

                        # Process through custom handler if configured
                        directive = self._process_tool_result(
                            tool_name, tool_args, result, run_id, steps, session_id
                        )

                        # Handle directive decision
                        if directive.decision == ToolUseDecision.STOP:
                            return directive.final_output if directive.final_output is not None else result
                        elif directive.decision == ToolUseDecision.ERROR:
                            raise RuntimeError(f"Tool handler returned ERROR for {tool_name}")

                        # Check if we should stop after this tool (non-custom behavior)
                        if self.should_stop_after_tool(tool_name):
                            return result  # Return raw tool output

                        # Handle message modifications from directive
                        if directive.replace_messages is not None:
                            # Replace context messages (advanced use case)
                            self.ctxmgr._by_run[run_id] = directive.replace_messages
                        elif directive.add_messages:
                            # Add extra messages before next LLM call
                            for msg in directive.add_messages:
                                if msg['role'] == 'system':
                                    self.ctxmgr._by_run[run_id].insert(0, msg)
                                else:
                                    self.ctxmgr._by_run[run_id].append(msg)

                        # Handle handoff (Phase 4)
                        if directive.handoff_agent:
                            # TODO: Implement handoff in Phase 4
                            pass

                        # Normal behavior: add to context and continue
                        self.ctxmgr.append_tool_result(run_id, tool_name, result, session_id)

                        # Add reset tool choice message if enabled
                        reset_msg = self._get_reset_tool_choice_message()
                        if reset_msg:
                            self.ctxmgr._by_run[run_id].append(reset_msg)

                        event = {'kind':'tool_result', 'result': result}
                        break
                    elif eff.kind == 'halt':
                        final_answer = eff.payload.get('final','')

                        # End step span
                        if current_step_ctx and observer:
                            observer.end_span(current_step_ctx, output=final_answer)
                            current_step_ctx = None

                        # Run output guardrails
                        if self.guardrails.has_output_guardrails:
                            ctx = {'run_id': run_id, 'session_id': session_id}
                            passed, modified, error = await self.guardrails.check_output(final_answer, ctx)
                            if not passed:
                                raise GuardrailError('output', error, final_answer)
                            final_answer = modified

                        # Validate structured output if output_type is set
                        if self.output_type:
                            try:
                                parsed = parse_structured_output(final_answer, self.output_type)
                                last_valid_output = parsed
                                # Add assistant response to context
                                self.ctxmgr.append_assistant_message(run_id, final_answer, session_id)
                                return parsed
                            except Exception as e:
                                structured_output_attempts += 1
                                policy = self.structured_output_policy

                                if structured_output_attempts > policy.max_retries:
                                    # Handle failure based on policy
                                    if policy.on_failure == "raise":
                                        raise StructuredOutputValidationError(e, final_answer, self.output_type)
                                    elif policy.on_failure == "return_raw":
                                        self.ctxmgr.append_assistant_message(run_id, final_answer, session_id)
                                        return final_answer
                                    elif policy.on_failure == "return_last_valid":
                                        if last_valid_output is not None:
                                            return last_valid_output
                                        raise StructuredOutputValidationError(e, final_answer, self.output_type)

                                # Retry: add error message and continue
                                retry_prompt = get_retry_prompt(self.output_type, e)
                                self.ctxmgr._by_run[run_id].append({'role': 'system', 'content': retry_prompt})
                                event = {'kind': 'start'}  # Restart to call LLM again
                                structured_retry = True
                                break

                        # Add assistant response to context
                        self.ctxmgr.append_assistant_message(run_id, final_answer, session_id)
                        return final_answer

                # End step span if it's still open (tool loop continues)
                if current_step_ctx and observer:
                    observer.end_span(current_step_ctx, output={
                        'status': 'structured_retry' if structured_retry else 'continue',
                        'has_tool_call': False if structured_retry else True,
                    })
                    current_step_ctx = None

                # A structured-output validation retry is not a tool iteration:
                # it is bounded by StructuredOutputPolicy.max_retries and must not
                # consume the tool max_steps budget. Restart without charging a step.
                if structured_retry:
                    continue

                steps += 1
                if steps > self.policies.get('max_steps', 24):
                    # Max steps exceeded - make one final LLM call WITHOUT tools to synthesize a response
                    logger.warning(f'max steps ({self.policies.get("max_steps", 24)}) exceeded, making final synthesis call')

                    # Add a message to guide the final response
                    synthesis_msg = {
                        'role': 'user',
                        'content': 'You have reached the maximum number of steps. Please synthesize a response based on the information you have gathered so far. Do not call any more tools.'
                    }
                    self.ctxmgr.append(run_id, synthesis_msg, session_id)

                    # Make final LLM call without tools
                    messages = self.ctxmgr.messages_for_llm(run_id, session_id)
                    system_prompt = self._get_system_prompt(run_id, context=context)
                    if system_prompt:
                        messages = [{'role': 'system', 'content': system_prompt}] + messages

                    provider = self._get_provider()
                    response = await provider.generate(messages, self.model_cfg.get('model', 'gpt-4o'), tools=[])

                    final_answer = response.get('answer', 'Unable to complete the request within the allowed steps.')
                    self.ctxmgr.append_assistant_message(run_id, final_answer, session_id)
                    return final_answer

        except asyncio.CancelledError:
            raise
        except Exception as e:
            # Log detailed error information
            error_type = type(e).__name__
            logger.error(f"Agent run failed: {error_type}: {str(e)}", exc_info=True)

            # End observability trace with error. Nested runs close only their
            # own span and never end/flush the parent-owned trace.
            if trace_ctx and observer:
                if trace_context is not None:
                    observer.end_span(trace_ctx, error=str(e))
                else:
                    observer.end_trace(trace_ctx, error=str(e))
                    observer.flush()

            # Emit error hook
            await self.hooks.emit('on_error', ErrorHookEvent(
                run_id=run_id,
                session_id=session_id,
                error=e,
                error_message=str(e),
                step=steps
            ))
            raise
        finally:
            # Capture scratchpad summary for next run
            if scratchpad:
                self._scratchpad_summary = scratchpad.get_summary()

            # End observability trace if not already ended (success case).
            # Nested runs close only their own span; the parent owns end/flush.
            run_duration_ms = (time.time() - run_start_time) * 1000
            if trace_ctx and observer:
                if trace_context is not None:
                    observer.end_span(trace_ctx, output=final_answer)
                else:
                    observer.end_trace(trace_ctx, output=final_answer)
                    observer.flush()

            # Clear observer from ContextManager
            self.ctxmgr.clear_observer()

            # Emit run finally hook
            await self.hooks.emit('on_run_finally', RunFinallyHookEvent(
                run_id=run_id,
                session_id=session_id,
                output=final_answer,
                total_steps=steps,
                total_duration_ms=run_duration_ms
            ))

    async def run_stream(
        self,
        input: Dict[str, Any],
        session_id: Optional[str] = None,
        generation_config: Optional[Dict[str, Any]] = None,
        context_refs: Optional[Any] = None,
        rlm: Optional[Dict[str, Any]] = None,
        thinking: Optional[Any] = None,
        harness: Optional[Any] = None,
        observability_context: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None,
        stateless: bool = False,
        node_id: Optional[str] = None,
        external_run_id: Optional[str] = None,
        trace_id: Optional[str] = None,
        trace_context: Optional['TraceContext'] = None,
        cancel_token: Optional[asyncio.Event] = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Streaming run - yields stream protocol events as they occur.

        Cancellation: pass an ``asyncio.Event`` as ``cancel_token`` and set it to
        stop the run. It is checked after each emitted event, so cancellation
        takes effect at the next event boundary rather than mid-write, and the
        run closes every block it left open before emitting ``abort`` then
        ``finish`` — a cancelled stream stays well-formed.

        Cancelling is not an error. A caller who stops a run, or a client that
        goes away, gets ``abort``; ``error`` continues to mean the run failed.

        Note: full parent-trace nesting for the streaming path (threading the
        parent handler through the stream loop) is a follow-up; the non-streaming
        run() path is fully nested. trace_context is accepted here so callers /
        as_tool can pass it uniformly without error.

        Note: If stop_on_first_tool policy is enabled (globally or per-tool), execution
        halts after tool execution. The tool-output-available event is still emitted,
        followed by finish-step and finish events.

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
            thinking: Optional per-run ThinkingConfig for extended thinking (runtime override)
            observability_context: Optional per-run observability context overrides.
                Merges with agent-level ObservabilityConfig.
                Supported keys: user_id, session_id, tags, metadata, trace_name
            context: Optional runtime context dict for callable system_prompt.
                Passed to system_prompt(context) if system_prompt is callable.
                Separate from context_refs (which is for RLM document content).
            stateless: If True, skip session state mutation. Messages won't be
                saved to session history. Useful for Mesh/Valis integration where
                state is managed externally. Default: False.
            node_id: Optional node identifier for Mesh orchestration.
                When set, events will include metadata with this node_id.
            external_run_id: Optional correlation ID for Mesh/external tracing.
                When set, events will include metadata with this ID for
                correlation across systems.
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

        # Check if Extended Thinking is enabled (per-run override or agent-level config)
        thinking_config = None
        if thinking:
            from .thinking import ThinkingConfig
            if isinstance(thinking, dict):
                thinking_config = ThinkingConfig(**thinking)
            else:
                thinking_config = thinking
        elif self.thinking_config and self.thinking_config.mode == 'reflection':
            thinking_config = self.thinking_config

        # Resolve Harness Mode config early so Extended Thinking can COMPOSE with
        # it (thinking bounded by the harness budget + traced). When absent/
        # disabled, harness_cfg stays None and the base loop runs unchanged.
        harness_cfg = None
        if harness:
            from .harness import HarnessConfig
            harness_cfg = HarnessConfig(**harness) if isinstance(harness, dict) else harness
        elif self.harness_config and self.harness_config.enabled:
            harness_cfg = self.harness_config
        if harness_cfg is not None and not harness_cfg.enabled:
            harness_cfg = None

        # Extended Thinking: route to the reflection engine. When Harness Mode is
        # also configured, thinking composes with it — the refine loop is bounded
        # by the harness budget (wallclock/tokens). Durable suspend/resume of a
        # reflection run is future work.
        if thinking_config and thinking_config.mode == 'reflection':
            async for event in self._run_with_thinking(
                input,
                session_id,
                thinking_config,
                generation_config=generation_config,
                context_refs=context_refs,
                rlm=rlm,
                observability_context=observability_context,
                context=context,
                stateless=stateless,
                node_id=node_id,
                external_run_id=external_run_id,
                harness_cfg=harness_cfg,
            ):
                yield event
            return

        # v1 does not compose Harness with RLM. If RLM was enabled but not
        # triggered (e.g. no context_refs) and harness is on, harness wins.
        if harness_cfg is not None and rlm_config is not None:
            logger.warning(
                "Harness Mode is enabled alongside RLM; for v1 Harness takes "
                "precedence and RLM is not composed."
            )

        # When Harness Mode drives a detached run (via RunManager), the
        # caller-supplied external_run_id IS the durable run id — the checkpoint,
        # approval records, and event log must all share it so resume()/recover()
        # can find the run. Outside Harness Mode the id stays an internal uuid and
        # external_run_id remains a pure correlation tag (non-harness path
        # unchanged).
        if external_run_id and harness_cfg is not None:
            run_id = external_run_id
        else:
            run_id = str(uuid.uuid4())
        run_start_time = time.time()
        self.ctxmgr.set_input(run_id, input, session_id, stateless=stateless)

        # Clear any injected tools from previous runs
        self._clear_injected_tools()

        # Setup event metadata for Mesh/Valis orchestration
        _event_metadata: Optional[EventMetadata] = None
        _event_step = 0
        if node_id or external_run_id:
            _event_metadata = EventMetadata(
                node_id=node_id or self.id,
                run_id=external_run_id
            )

        def _wrap_event(event_dict: Dict[str, Any]) -> Dict[str, Any]:
            """Wrap event with metadata if orchestration is enabled."""
            nonlocal _event_step
            if _event_metadata is None:
                return event_dict
            event_type = event_dict.get('type', '')
            # Add metadata to streaming content events
            if event_type in ('text-delta', 'reasoning-delta', 'reasoning-signature',
                              'tool-input-available', 'tool-output-available',
                              'start-step', 'finish-step', 'finish-message'):
                _event_metadata.step = _event_step
                _event_metadata.timestamp = time.time()
                _event_step += 1
                return add_metadata(event_dict, _event_metadata)
            return event_dict

        # Setup observability trace. When nested (trace_context given), use the
        # PARENT's handler and open a child span under the parent trace; the child
        # inherits the parent's sampling decision and must NOT end/flush the trace.
        # (Extends the run()-path nesting fix to the streaming path — closes the
        # deferred stream-path gap in TRACE_AGENT_WORK.md §7.)
        observer = trace_context.handler if trace_context else self._observer
        trace_ctx: Optional['SpanContext'] = None
        current_step_ctx: Optional['SpanContext'] = None
        final_answer = ''
        if observer and (trace_context is not None or observer.should_sample()):
            from .integrations.base import SpanKind, GenerationData, ToolData
            # Merge observability_context with config. When nested under a parent
            # trace, this agent may have no observability config of its own — fall
            # back to sensible defaults so nesting still works.
            obs_config = self._observability_config
            if obs_config is not None and observability_context:
                obs_config = obs_config.with_context(**observability_context)
            _trace_name = (obs_config.trace_name if obs_config else None) or self.id
            _capture_input = obs_config.capture_input if obs_config else True
            _trace_input = input if _capture_input else {'message': '[redacted]'}

            if trace_context is not None:
                trace_ctx = observer.start_span(
                    trace_context.span,
                    _trace_name,
                    SpanKind.AGENT_RUN,
                    input=_trace_input,
                )
            else:
                trace_ctx = observer.start_trace(
                    trace_id=run_id,
                    name=_trace_name,
                    input=_trace_input,
                    metadata=obs_config.metadata if obs_config else None,
                    tags=obs_config.tags if obs_config else None,
                    user_id=obs_config.user_id if obs_config else None,
                    session_id=(obs_config.session_id if obs_config else None) or session_id,
                )

            # Wire up observer to ContextManager for memory operation tracing
            self.ctxmgr.set_observer(observer, trace_ctx)

            # Emit run start hook
            await self.hooks.emit('on_run_start', RunStartHookEvent(
                run_id=run_id,
                session_id=session_id,
                input=input,
                agent_id=self.id
            ))

        # Setup scratchpad for this run
        scratchpad = None
        if self._scratchpad_config:
            from .tools.scratchpad import Scratchpad, get_scratchpad_tools
            scratchpad = Scratchpad(self._scratchpad_config)
            for tool in get_scratchpad_tools(scratchpad):
                self._injected_tools[tool.name] = tool
            # Inject previous run's summary into context
            if self._scratchpad_summary:
                self.ctxmgr._by_run[run_id].insert(0, {
                    'role': 'system',
                    'content': f"[Previous Run Context]\n{self._scratchpad_summary}"
                })

        # Add dynamic instruction if set
        if self.instruction:
            if callable(self.instruction):
                instruction_text = self.instruction(self.tool_context)
            else:
                instruction_text = self.instruction
            self.ctxmgr._by_run[run_id].insert(0, {'role': 'system', 'content': instruction_text})

        # Add structured output schema prompt if output_type is set
        if self.output_type:
            schema_prompt = get_json_mode_system_prompt(self.output_type)
            self.ctxmgr._by_run[run_id].insert(0, {'role': 'system', 'content': schema_prompt})

        # Run input guardrails
        if self.guardrails.has_input_guardrails:
            ctx = {'run_id': run_id, 'session_id': session_id}
            content = input.get('message', input)
            passed, modified, error = await self.guardrails.check_input(content, ctx)
            if not passed:
                error_event = ErrorEvent(error=f"Input guardrail failed: {error}")
                yield error_event.to_dict()
                yield {'type': 'finish'}
                return
            # If content was modified, update the input
            if modified != content and 'message' in input:
                input['message'] = modified
                self.ctxmgr.set_input(run_id, input, session_id, stateless=stateless)

        # Emit start event (V5 UI Stream Protocol)
        yield StartEvent().to_dict()

        loop_state: Dict[str, Any] = {'steps': 0, 'final_answer': final_answer}

        # Harness Mode: build the controller + hooks (default-off). When
        # harness_cfg is None all hooks are None and _step_loop is byte-identical
        # to the legacy path (proven by tests/test_harness/test_step_loop_equivalence.py).
        harness_controller = None
        _hk_pre_step = _hk_prepass = _hk_resolver = _hk_budget = None
        _hk_on_tool = None
        _hk_max_steps = None
        if harness_cfg is not None:
            from .harness import HarnessController
            from .harness.exceptions import BudgetExhausted, SuspendRun
            harness_controller = HarnessController(agent=self, config=harness_cfg)
            harness_controller.bind_run(run_id=run_id, session_id=session_id, context=context)
            _hk_pre_step = harness_controller.pre_step_hook
            _hk_prepass = harness_controller.approval_prepass
            _hk_resolver = harness_controller.approval_resolver
            _hk_budget = harness_controller.budget_hook
            # Per-tool checkpoint hook only when opted in (crash recovery).
            _hk_on_tool = (
                harness_controller.on_tool_completed
                if harness_cfg.checkpoint_each_tool
                else None
            )
            _hk_max_steps = harness_controller.effective_max_steps

        try:
            if harness_controller is not None:
                yield harness_controller.run_started_event()
                # Create/inject the sandbox (no-op if disabled) before the loop
                # so sandbox tools are visible to the model on step 1 (§6.7).
                for _sb_event in await harness_controller.ensure_sandbox():
                    yield _sb_event
            # Every emitted event passes through here, which makes this the one
            # place that knows what the run has left open. `aclosing` shuts the
            # inner generator down deterministically on cancel rather than
            # leaving it to the garbage collector.
            _open = _OpenStreamState()
            _cancelled = False
            async with aclosing(self._step_loop(
                run_id=run_id,
                session_id=session_id,
                context=context,
                generation_config=generation_config,
                trace_ctx=trace_ctx,
                observer=observer,
                wrap_event=_wrap_event,
                loop_state=loop_state,
                max_steps=_hk_max_steps,
                pre_step_hook=_hk_pre_step,
                approval_prepass=_hk_prepass,
                approval_resolver=_hk_resolver,
                budget_hook=_hk_budget,
                on_tool_completed=_hk_on_tool,
            )) as _steps:
                async for event in _steps:
                    if harness_controller is not None:
                        harness_controller.observe_event(event)
                    _open.observe(event)
                    yield event

                    # Checked after the event rather than before, so a cancel
                    # never truncates a delta that was already produced.
                    if cancel_token is not None and cancel_token.is_set():
                        _cancelled = True
                        break

            if _cancelled:
                for _close in _open.closing_events('Run cancelled'):
                    if harness_controller is not None:
                        harness_controller.observe_event(_close)
                    yield _close
                if harness_controller is not None:
                    for _sb_event in await harness_controller.close_sandbox():
                        yield _sb_event
                    harness_controller.mark_cancelled(run_id)
                    yield harness_controller.run_finished_event('cancelled')
                return

            if harness_controller is not None:
                for _sb_event in await harness_controller.close_sandbox():
                    yield _sb_event
                harness_controller.mark_completed(run_id)
                yield harness_controller.run_finished_event('completed')
        except asyncio.CancelledError:
            raise
        except Exception as e:
            # Harness Mode durable suspension: persist checkpoint + emit approval
            # and suspended events, then return (run kept alive for resume; the
            # sandbox is intentionally NOT closed so resume can reconnect).
            if harness_controller is not None and isinstance(e, SuspendRun):
                for ev in harness_controller.on_suspend(e):
                    yield ev
                return
            # Harness Mode budget exhaustion: emit event + synthesize a partial
            # answer (reuse the shared max-steps synthesis behavior).
            if harness_controller is not None and isinstance(e, BudgetExhausted):
                yield harness_controller.budget_exhausted_event(e)
                async for event in self._synthesize_final(
                    run_id, session_id, context=context, wrap_event=_wrap_event,
                    reason=getattr(e, 'reason', str(e)),
                ):
                    yield event
                for _sb_event in await harness_controller.close_sandbox():
                    yield _sb_event
                harness_controller.mark_completed(run_id)
                yield harness_controller.run_finished_event('completed')
                return
            # Ensure error message is never empty
            error_msg = str(e) if str(e) else f"{type(e).__name__}: {repr(e)}"
            logger.error(f"Agent stream error: {error_msg}", exc_info=True)

            # End observability trace with error. Nested runs close only their own
            # span; the parent owns end/flush.
            if trace_ctx and observer:
                if trace_context is not None:
                    observer.end_span(trace_ctx, error=error_msg)
                else:
                    observer.end_trace(trace_ctx, error=error_msg)
                    observer.flush()

            # Emit error hook
            await self.hooks.emit('on_error', ErrorHookEvent(
                run_id=run_id,
                session_id=session_id,
                error=e,
                error_message=error_msg,
                step=loop_state['steps']
            ))

            error_event = ErrorEvent(error=error_msg)
            yield error_event.to_dict()
            raise
        finally:
            final_answer = loop_state['final_answer']
            # Capture scratchpad summary for next run
            if scratchpad:
                self._scratchpad_summary = scratchpad.get_summary()

            # End observability trace if not already ended (success case). Nested
            # runs close only their own span; the parent owns end/flush.
            run_duration_ms = (time.time() - run_start_time) * 1000
            if trace_ctx and observer:
                if trace_context is not None:
                    observer.end_span(trace_ctx, output=final_answer)
                else:
                    observer.end_trace(trace_ctx, output=final_answer)
                    observer.flush()

            # Clear observer from ContextManager
            self.ctxmgr.clear_observer()

            # Emit run finally hook
            await self.hooks.emit('on_run_finally', RunFinallyHookEvent(
                run_id=run_id,
                session_id=session_id,
                output=final_answer,
                total_steps=loop_state['steps'],
                total_duration_ms=run_duration_ms
            ))

    async def resume(
        self,
        run_id: str,
        decisions: List['ApprovalDecision'],
        *,
        harness: Optional[Any] = None,
        context: Optional[Dict[str, Any]] = None,
        generation_config: Optional[Dict[str, Any]] = None,
        force: bool = False,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Resume a suspended durable run with human approval decisions.

        Loads the persisted checkpoint, applies the decisions (approve/reject),
        re-executes the suspended step's tool calls, then continues the agent
        loop — emitting the same stream events as ``run_stream``.

        Args:
            run_id: The suspended run to resume.
            decisions: Approval decisions for the run's pending approvals.
            harness: Optional HarnessConfig/dict override (defaults to the
                agent-level config).
            context: Optional runtime context for callable system prompts.
            generation_config: Optional per-run generation config.
            force: Resume even if the config hash changed (logs nothing extra;
                use with care — model/tool changes can corrupt continuation).

        Yields:
            Stream protocol event dicts for the resumed run.

        Raises:
            ValueError: If Harness Mode is not configured, or the run is not in
                a suspended state, or the config changed and ``force`` is False.
        """
        from .harness import HarnessConfig, HarnessController

        harness_cfg = None
        if harness:
            harness_cfg = HarnessConfig(**harness) if isinstance(harness, dict) else harness
        elif self.harness_config:
            harness_cfg = self.harness_config
        if harness_cfg is None or not harness_cfg.enabled:
            raise ValueError("resume() requires Harness Mode to be enabled")

        # A suspended REFLECTION run resumes its phase state machine, not the
        # step loop. Peek at the checkpoint to dispatch.
        from .harness.checkpoint import CheckpointStore
        _ckpt = CheckpointStore(harness_cfg.db_path).load(run_id)
        if _ckpt is not None and _ckpt.reflection is not None:
            async for event in self._resume_reflection(
                run_id, decisions, _ckpt, harness_cfg, context=context
            ):
                yield event
            return

        controller = HarnessController(agent=self, config=harness_cfg)
        # Trace the resume leg (reuse the run_id trace so it continues the original
        # execution's trace). Without this, post-suspension work is untraced.
        observer, trace_ctx = self._begin_leg_trace(run_id)
        try:
            async for event in controller.resume(
                run_id,
                decisions,
                context=context,
                generation_config=generation_config,
                force=force,
                observer=observer,
                trace_ctx=trace_ctx,
            ):
                yield event
        finally:
            self._end_leg_trace(observer, trace_ctx)

    async def _resume_reflection(
        self,
        run_id: str,
        decisions: List['ApprovalDecision'],
        ckpt: Any,
        harness_cfg: Any,
        *,
        context: Optional[Dict[str, Any]] = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Resume a suspended reflection run: apply the decision(s), rehydrate the
        phase state, and continue the phase machine from the cursor."""
        from .thinking import ReflectionController
        from .harness import HarnessController
        from .harness.exceptions import SuspendRun
        from .harness.events import HarnessResumedEvent

        config = self.thinking_config
        if config is None:
            raise ValueError("resume of a reflection run requires the agent's thinking config")

        hc = HarnessController(agent=self, config=harness_cfg)
        hc.bind_run(run_id=run_id, session_id=ckpt.session_id, context=context)
        controller = ReflectionController(
            agent=self, config=config, budget=harness_cfg.budget, harness_controller=hc
        )
        controller._run_id = run_id

        # Record the human decisions, then rehydrate state/scratch/cursor and add
        # approved tool NAMES so the re-run of the suspended phase runs them.
        approval_by_id = {r.approval_id: r for r in ckpt.pending_approvals}
        for d in decisions:
            await hc._gate.record(d)
        cursor = controller.restore(ckpt)
        for d in decisions:
            req = approval_by_id.get(d.approval_id)
            if req is not None:
                if d.decision == 'approve':
                    controller._approved_tools.add(req.tool_name)
                else:
                    controller._denied_tools.add(req.tool_name)

        hc._checkpoints.set_status(run_id, 'running')
        yield HarnessResumedEvent(run_id=run_id).to_dict()
        yield StartEvent().to_dict()
        yield StepStartEvent().to_dict()

        scratch_id = ckpt.reflection['scratch_id']
        reasoning_parts: List[str] = []
        answer_parts: List[str] = []
        answer_step_started = False
        suspended = False
        observer, trace_ctx = self._begin_leg_trace(run_id)
        try:
            async for event in controller._drive_phases(
                cursor, scratch_id, trace_ctx=trace_ctx, observer=observer, session_id=None
            ):
                if event.get('type') == 'text-start' and not answer_step_started:
                    yield StepFinishEvent().to_dict()
                    yield StepStartEvent().to_dict()
                    answer_step_started = True
                yield event
                et = event.get('type')
                if et == 'reasoning-delta':
                    reasoning_parts.append(event.get('delta', ''))
                elif et == 'text-delta':
                    answer_parts.append(event.get('delta', ''))
        except SuspendRun as s:
            suspended = True
            for ev in hc.on_suspend(s):
                yield ev
        finally:
            self._end_leg_trace(observer, trace_ctx)
            self.ctxmgr._by_run.pop(scratch_id, None)

        if suspended:
            return

        # Reflection finished on resume: emit completion + persist to the real run.
        from .events import DataEvent
        yield DataEvent(
            type='data-thinking-complete',
            data={
                'steps': cursor.get('step', 0),
                'iterations': controller.state.iteration,
                'final_confidence': controller.state.confidence,
                'thinking_tokens': controller.state.total_tokens,
                'thinking_model': controller._thinking_model or self.model_cfg.get('model'),
            },
            transient=False,
        ).to_dict()
        self.ctxmgr.append_assistant_with_reasoning(
            run_id, ''.join(reasoning_parts), ''.join(answer_parts), {}, ckpt.session_id
        )
        hc._checkpoints.set_status(run_id, 'completed')
        yield StepFinishEvent().to_dict()
        yield FinishEvent().to_dict()

    async def recover(
        self,
        run_id: str,
        *,
        harness: Optional[Any] = None,
        context: Optional[Dict[str, Any]] = None,
        generation_config: Optional[Dict[str, Any]] = None,
        force: bool = False,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Recover a run that crashed while ``running`` (not suspended).

        Loads the last ``running`` checkpoint, rehydrates the message window, and
        — if the crash happened mid-step — re-executes only the tool calls that
        had not completed (those already committed are skipped via
        ``completed_tool_calls``), then continues the loop. Requires the run to
        have been executed with ``harness.checkpoint_each_tool=True`` for
        mid-step granularity; otherwise recovery restarts from the last step
        boundary. Emits the same stream events as ``run_stream`` plus
        ``data-harness-recovered``.

        Args:
            run_id: The run to recover.
            harness: Optional HarnessConfig/dict override (defaults to the
                agent-level config).
            context: Optional runtime context for callable system prompts.
            generation_config: Optional per-run generation config.
            force: Recover even if the config hash changed.

        Yields:
            Stream protocol event dicts for the recovered run.

        Raises:
            ValueError: If Harness Mode is not enabled, or there is no
                recoverable ``running`` checkpoint for ``run_id``.
        """
        from .harness import HarnessConfig, HarnessController

        harness_cfg = None
        if harness:
            harness_cfg = HarnessConfig(**harness) if isinstance(harness, dict) else harness
        elif self.harness_config:
            harness_cfg = self.harness_config
        if harness_cfg is None or not harness_cfg.enabled:
            raise ValueError("recover() requires Harness Mode to be enabled")

        controller = HarnessController(agent=self, config=harness_cfg)
        observer, trace_ctx = self._begin_leg_trace(run_id)
        try:
            async for event in controller.recover(
                run_id,
                context=context,
                generation_config=generation_config,
                force=force,
                observer=observer,
                trace_ctx=trace_ctx,
            ):
                yield event
        finally:
            self._end_leg_trace(observer, trace_ctx)

    def _begin_leg_trace(self, run_id: str):
        """Open an observability trace for a resume/recover leg (reusing the
        run_id trace so it continues the original execution's trace). Returns
        ``(observer, trace_ctx)``; ``trace_ctx`` is None when not sampling."""
        observer = self._observer
        if not (observer and observer.should_sample()):
            return observer, None
        obs_config = self._observability_config
        name = (obs_config.trace_name if obs_config else None) or self.id
        # `input` is a REQUIRED parameter of ObservabilityHandler.start_trace, so
        # omitting it raised TypeError and killed the run — but only on a
        # resume/recover leg, which is why it surfaced as an intermittent
        # failure rather than a broken build. A leg has no new caller input (it
        # continues an existing run whose input was recorded when the trace was
        # first opened at `_run_stream_impl`), so an empty mapping is the honest
        # value here; passing None would be reported to the backend as an input
        # of null rather than "nothing new was supplied".
        trace_ctx = observer.start_trace(
            trace_id=run_id,
            name=name,
            input={},
            metadata=obs_config.metadata if obs_config else None,
            tags=obs_config.tags if obs_config else None,
            user_id=obs_config.user_id if obs_config else None,
            session_id=obs_config.session_id if obs_config else None,
        )
        self.ctxmgr.set_observer(observer, trace_ctx)
        return observer, trace_ctx

    def _end_leg_trace(self, observer, trace_ctx) -> None:
        if trace_ctx and observer:
            observer.end_trace(trace_ctx, output=None)
            observer.flush()
            self.ctxmgr.clear_observer()

    async def _step_loop(
        self,
        run_id: str,
        session_id: Optional[str],
        *,
        context: Optional[Dict[str, Any]] = None,
        generation_config: Optional[Dict[str, Any]] = None,
        trace_ctx: Optional['SpanContext'] = None,
        observer: Optional[Any] = None,
        wrap_event: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
        loop_state: Optional[Dict[str, Any]] = None,
        max_steps: Optional[int] = None,
        pre_step_hook: Optional[Callable[..., Any]] = None,
        approval_prepass: Optional[Callable[..., Any]] = None,
        approval_resolver: Optional[Callable[[str], bool]] = None,
        budget_hook: Optional[Callable[..., Any]] = None,
        on_tool_completed: Optional[Callable[..., Any]] = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Core agentic step loop, extracted from run_stream (M0).

        With all hooks None this is byte-for-byte identical to the legacy loop.
        Harness Mode passes hooks to interpose budget/compaction/approval logic.

        Args:
            run_id: Active run identifier.
            session_id: Optional session id for multi-turn context.
            context: Runtime context dict for a callable system_prompt.
            generation_config: Per-run generation config override.
            trace_ctx: Observability trace span context, if sampling.
            wrap_event: Event-metadata wrapper (Mesh/Valis); identity if None.
            loop_state: Mutable dict the caller reads after the loop for
                ``steps``/``final_answer`` (used by run_stream's except/finally).
            pre_step_hook: Async generator hook run before each step (compaction).
            approval_hook: Async predicate consulted at the tool-approval seam.
            budget_hook: Async hook run before each step; may raise to stop.

        Yields:
            Stream protocol event dicts, identical in shape to run_stream.
        """
        if wrap_event is None:
            wrap_event = lambda event_dict: event_dict
        if loop_state is None:
            loop_state = {'steps': 0, 'final_answer': ''}
        # When not threaded a nesting-aware observer (e.g. harness resume/recover
        # legs), fall back to the agent's own observer — behavior unchanged.
        if observer is None:
            observer = self._observer
        current_step_ctx: Optional['SpanContext'] = None

        steps = 0
        if max_steps is None:
            max_steps = self.policies.get('max_steps', 24)
        structured_output_attempts = 0

        while steps < max_steps:
            steps += 1
            loop_state['steps'] = steps
            if budget_hook is not None:
                await budget_hook(steps)
            if pre_step_hook is not None:
                async for _hook_event in pre_step_hook(run_id, session_id, steps):
                    yield _hook_event

            # Get messages and stream LLM response
            messages = self.ctxmgr.messages_for_llm(run_id, session_id)

            # Prepend system prompt if set (for prompt caching)
            system_prompt = self._get_system_prompt(run_id, context=context)
            if system_prompt:
                messages = [{'role': 'system', 'content': system_prompt}] + messages

            # Start step span for observability
            if trace_ctx and observer:
                from .integrations.base import SpanKind, GenerationData, ToolData
                current_step_ctx = observer.start_span(
                    trace_ctx, f"step-{steps}", SpanKind.STEP,
                    input={'messages_count': len(messages), 'step': steps}
                )

            # Emit start-step event (V5 UI Stream Protocol for multi-step agents)
            yield wrap_event(StepStartEvent().to_dict())
            messages_for_obs = messages.copy()  # Copy for observability
            llm_start_time = time.time()
            provider = self._get_provider()

            # Merge agent-level and per-run generation configs
            config = {**self.generation_config, **(generation_config or {})}

            # Track what happened during streaming
            full_text = []
            tool_calls = []  # list of {tool_call_id, tool_name, input}
            finish_reason = 'stop'
            usage = None
            response_metadata = None

            # Initialize incremental JSON parser for structured output streaming
            json_parser = None
            output_mode = OutputMode.TEXT
            if self.output_type:
                output_mode = detect_output_mode(self.output_type)
                if output_mode != OutputMode.TEXT:
                    element_type = get_element_type(self.output_type) if output_mode == OutputMode.ARRAY else None
                    json_parser = IncrementalJsonParser(self.output_type, element_type)

            # Stream from provider and forward events
            # Get schemas from instance tools + global registry
            tool_schemas = self._get_tool_schemas()
            async for event in provider.stream(messages, model=self.model_cfg['model'], tools=tool_schemas, generation_config=config):
                # Track metadata for finish events (don't forward finish-message)
                if event.type == 'finish-message':
                    finish_reason = event.finish_reason
                    continue  # Don't forward, consume internally

                # Track response metadata (usage, model info)
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
                    # Forward response-metadata so consumers can track usage
                    yield wrap_event(event.to_dict())
                    continue

                # Forward all other stream protocol events
                yield wrap_event(event.to_dict())

                # Track text content
                if event.type == 'text-delta':
                    full_text.append(event.delta)

                    # Feed incremental JSON parser for structured output streaming
                    if json_parser and event.delta:
                        for parsed in json_parser.feed(event.delta):
                            if isinstance(parsed, StreamedElement):
                                # Emit data-object-element for array items
                                yield ObjectElementEvent(
                                    index=parsed.index,
                                    element=parsed.element
                                ).to_dict()
                            elif isinstance(parsed, PartialObject):
                                # Emit data-object-partial for object updates
                                yield ObjectPartialEvent(
                                    partial=parsed.partial
                                ).to_dict()

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

            # Log LLM generation for observability (after streaming completes)
            if trace_ctx and observer:
                llm_end_time = time.time()
                llm_latency = (llm_end_time - llm_start_time) * 1000
                observer.log_generation(
                    current_step_ctx or trace_ctx,
                    GenerationData(
                        model=self.model_cfg.get('model', 'unknown'),
                        provider=self.model_cfg.get('provider', 'unknown'),
                        messages=messages_for_obs,
                        response=''.join(full_text) if full_text else None,
                        tool_calls=[{'tool': tc['tool_name'], 'args': tc['input']} for tc in tool_calls] if tool_calls else None,
                        usage=usage,
                        generation_config=config,
                        latency_ms=llm_latency,
                        start_time=llm_start_time,
                        end_time=llm_end_time,
                    )
                )

            # If we got text and no tool calls, we're done
            if full_text and not tool_calls:
                answer = ''.join(full_text)
                final_answer = answer  # Track for observability
                loop_state['final_answer'] = answer

                # Run output guardrails
                if self.guardrails.has_output_guardrails:
                    ctx = {'run_id': run_id, 'session_id': session_id}
                    passed, modified, error = await self.guardrails.check_output(answer, ctx)
                    if not passed:
                        error_event = ErrorEvent(error=f"Output guardrail failed: {error}")
                        yield error_event.to_dict()
                        yield {'type': 'finish'}
                        return
                    answer = modified

                # Validate structured output if output_type is set
                if self.output_type:
                    try:
                        validated_object = parse_structured_output(answer, self.output_type)
                        # Validation passed - emit data-object-complete event
                        yield ObjectCompleteEvent(
                            object=validated_object,
                            mode='array' if output_mode == OutputMode.ARRAY else 'object'
                        ).to_dict()
                    except Exception as e:
                        structured_output_attempts += 1
                        policy = self.structured_output_policy

                        if structured_output_attempts > policy.max_retries:
                            # Handle failure based on policy
                            if policy.on_failure == "raise":
                                error_event = ErrorEvent(
                                    error=f"Structured output validation failed: {e}"
                                )
                                yield error_event.to_dict()
                                yield {'type': 'finish'}
                                return
                            # For return_raw or return_last_valid, continue with answer
                        else:
                            # Retry: add error message and continue loop.
                            # Structured-output retries are bounded by
                            # StructuredOutputPolicy.max_retries, not the tool
                            # max_steps budget — offset the loop-top increment so
                            # this retry does not consume a tool step.
                            retry_prompt = get_retry_prompt(self.output_type, e)
                            self.ctxmgr._by_run[run_id].append({'role': 'system', 'content': retry_prompt})
                            steps -= 1
                            continue  # Go back to LLM

                self.ctxmgr.append_assistant_message(run_id, answer, session_id)

                # End step span for observability
                if current_step_ctx and observer:
                    observer.end_span(current_step_ctx, output=answer)
                    current_step_ctx = None

                # Emit finish-step event (AI SDK v5 spec: simple event, no fields)
                yield wrap_event({'type': 'finish-step'})

                # Emit finish event (AI SDK v5 spec: simple event, no fields)
                yield {'type': 'finish'}
                return

            # If we got tool calls, execute them and continue
            if tool_calls:
                # Add assistant's tool call to context BEFORE executing tools
                # This is critical - without this, LLM doesn't know it made tool calls
                # Use OpenAI's expected format with tool_calls array
                tool_calls_formatted = [
                    {
                        'id': tc['tool_call_id'],
                        'type': 'function',
                        'function': {
                            'name': tc['tool_name'],
                            'arguments': json.dumps(tc['input']) if isinstance(tc['input'], dict) else str(tc['input'])
                        }
                    }
                    for tc in tool_calls
                ]
                self.ctxmgr.append(run_id, {
                    'role': 'assistant',
                    'content': None,
                    'tool_calls': tool_calls_formatted
                }, session_id)

                # Harness durable-approval pre-pass: may snapshot + raise
                # SuspendRun for tools awaiting a human decision (no-op legacy).
                if approval_prepass is not None:
                    await approval_prepass(tool_calls, run_id, session_id, steps)

                async for _tc_event in self._run_tool_calls(
                    tool_calls,
                    run_id,
                    session_id,
                    steps=steps,
                    context=context,
                    wrap_event=wrap_event,
                    trace_ctx=trace_ctx,
                    current_step_ctx=current_step_ctx,
                    observer=observer,
                    loop_state=loop_state,
                    approval_resolver=approval_resolver,
                    on_tool_completed=on_tool_completed,
                ):
                    yield _tc_event
                current_step_ctx = None
                if loop_state.get('control') == 'terminate':
                    return

                # Continue loop to get next LLM response
                continue

            # If we got here with no text and no tool calls, something's wrong
            error_event = ErrorEvent(error='No response from LLM')
            yield error_event.to_dict()
            yield {'type': 'finish'}
            return

        # Max steps exceeded - synthesize a partial answer (reuse shared helper).
        async for event in self._synthesize_final(
            run_id,
            session_id,
            context=context,
            wrap_event=wrap_event,
            reason=f'max steps ({max_steps}) exceeded',
        ):
            yield event

    def _parallel_batch_eligible(
        self,
        tool_calls: List[Dict[str, Any]],
        *,
        approval_resolver: Optional[Callable[[str], bool]] = None,
        skip_tool_call_ids: Optional[set] = None,
    ) -> bool:
        """Whether this whole batch may run concurrently.

        All-or-nothing per batch. Partitioning a batch into a parallel group and
        a serial one is harder to reason about than it looks — the serial tools
        would still observe the parallel ones' side effects at an unpredictable
        point — so a single non-eligible call makes the entire step serial.

        Everything below is a reason a tool cannot safely overlap with another:

        - the policy is off (the default), or there is only one call
        - the tool has not opted in via ``parallel_safe``
        - the handler is an async generator: its interstitial events are
          stripped to ``{type, id, delta}`` with no tool_call_id, so a consumer
          could not tell two concurrent tools' events apart
        - the tool needs confirmation, or an approval hook is installed: a
          decision has to be made before the side effect happens, not after
        - a tool guardrail applies: it can rewrite the arguments, which has to
          happen before execution
        - the batch is being replayed after a crash, where results already exist
        """
        if self.policies.get('tool_execution') != 'parallel':
            return False
        if len(tool_calls) < 2:
            return False
        if skip_tool_call_ids:
            return False
        if approval_resolver is not None or self._tool_approval_callback:
            return False

        for tc in tool_calls:
            try:
                tool = self._get_tool(tc['tool_name'])
            except Exception:
                return False
            if not getattr(tool, 'parallel_safe', False):
                return False
            if getattr(tool, '_is_async_generator', False):
                return False
            if getattr(tool, 'requires_confirmation', False):
                return False
            if self.guardrails.has_tool_guardrails(tc['tool_name']):
                return False

        return True

    async def _prefetch_parallel_tools(
        self,
        tool_calls: List[Dict[str, Any]],
        *,
        approval_resolver: Optional[Callable[[str], bool]] = None,
        skip_tool_call_ids: Optional[set] = None,
    ) -> Dict[str, Any]:
        """Run an eligible batch's handlers concurrently, keyed by tool_call_id.

        Returns an empty dict when the batch is not eligible, which leaves the
        caller on its normal serial path. Failures come back as exception
        objects rather than raising, so each one can be re-raised at the point
        in the loop where that tool would have run and take the ordinary
        error path.
        """
        if not self._parallel_batch_eligible(
            tool_calls,
            approval_resolver=approval_resolver,
            skip_tool_call_ids=skip_tool_call_ids,
        ):
            return {}

        ids = [tc['tool_call_id'] for tc in tool_calls]
        results = await asyncio.gather(
            *(
                self._get_tool(tc['tool_name']).run(tc.get('input', {}), self.tool_context)
                for tc in tool_calls
            ),
            return_exceptions=True,
        )
        return dict(zip(ids, results))

    async def _run_tool_calls(
        self,
        tool_calls: List[Dict[str, Any]],
        run_id: str,
        session_id: Optional[str],
        *,
        steps: int,
        context: Optional[Dict[str, Any]] = None,
        wrap_event: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
        trace_ctx: Optional['SpanContext'] = None,
        current_step_ctx: Optional['SpanContext'] = None,
        observer: Optional[Any] = None,
        loop_state: Optional[Dict[str, Any]] = None,
        approval_resolver: Optional[Callable[[str], bool]] = None,
        on_tool_completed: Optional[Callable[..., Any]] = None,
        skip_tool_call_ids: Optional[set] = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Execute one step's tool calls, extracted from _step_loop (M3).

        Assumes the assistant tool_calls message is already in context (the
        caller appends it before the durable-approval pre-pass so a suspend
        checkpoint captures it). Shared by _step_loop and Agent.resume so a
        resumed run re-executes a suspended step without a spurious LLM call.

        Sets ``loop_state['control']`` to ``'terminate'`` if a terminal event
        sequence (finish) was already emitted, else ``'continue'``.

        Args:
            tool_calls: Collected tool calls for the step.
            run_id: Active run id.
            session_id: Optional session id.
            steps: Current step number (for custom tool-result handlers).
            context: Runtime context for callable system prompts.
            wrap_event: Event-metadata wrapper; identity if None.
            trace_ctx: Observability trace context.
            current_step_ctx: Observability step span.
            loop_state: Mutable dict; receives the control signal.
            approval_resolver: Maps a tool_call_id to an approve/deny bool
                (harness durable approvals). When None the inline
                ``_tool_approval_callback`` path is used unchanged.
            on_tool_completed: Optional async hook called after each tool result
                is committed to context, as
                ``on_tool_completed(run_id, session_id, steps, tool_calls,
                completed_ids)``. The harness uses it to persist a per-tool
                running checkpoint so a mid-step crash can recover without
                re-running completed tools. None (default) ⇒ no extra work.
            skip_tool_call_ids: Tool-call ids whose results are already in the
                rehydrated message window (crash recovery / resume replay); they
                are skipped instead of re-executed. None (default) ⇒ run all.

        Yields:
            Stream protocol event dicts for the executed tool calls.
        """
        from .integrations.base import ToolData

        if wrap_event is None:
            wrap_event = lambda event_dict: event_dict
        if loop_state is None:
            loop_state = {}
        if observer is None:
            observer = self._observer

        completed_ids: List[str] = []

        async def _mark_completed(tool_call_id: str) -> None:
            """Record a committed tool result and fire the checkpoint hook."""
            completed_ids.append(tool_call_id)
            if on_tool_completed is not None:
                await on_tool_completed(run_id, session_id, steps, tool_calls, completed_ids)

        # Opt-in concurrency. Only the handler calls overlap; every branch below
        # (approval, guardrails, directives, context appends, control
        # resolution) still runs in call order against the finished results, so
        # the emitted event sequence is identical either way and only the
        # wall-clock changes. That is deliberate: it buys the latency win —
        # measured 62.4s -> 31.1s on two 31s tools — without introducing a
        # second, concurrent copy of a hundred lines of branching.
        #
        # The visible trade-off, stated plainly: outputs appear together once
        # the slowest tool finishes, rather than each as it completes.
        prefetched = await self._prefetch_parallel_tools(
            tool_calls,
            approval_resolver=approval_resolver,
            skip_tool_call_ids=skip_tool_call_ids,
        )

        for tc in tool_calls:
            # Replay-skip (crash recovery / resume): this tool's result is
            # already in the rehydrated context — do not re-execute it.
            if skip_tool_call_ids and tc['tool_call_id'] in skip_tool_call_ids:
                completed_ids.append(tc['tool_call_id'])
                continue
            tool_args = tc.get('input', {})
            tool_start_time = time.time()
            try:
                # Get tool to check if it's streaming (instance or global)
                tool = self._get_tool(tc['tool_name'])
                result = None
                tool_error = None

                # Check tool approval (harness resolver takes precedence over inline callback)
                approved = True
                if approval_resolver is not None:
                    approved = approval_resolver(tc['tool_call_id'])
                elif self._tool_approval_callback:
                    approved = await self._tool_approval_callback(tc['tool_name'], tool_args, tc['tool_call_id'])
                if not approved:
                    # Tool was denied - add error result and continue
                    error_result = {'error': f"Tool '{tc['tool_name']}' was denied by user"}
                    output_event = ToolOutputAvailableEvent(
                        tool_call_id=tc['tool_call_id'],
                        output=error_result
                    )
                    yield wrap_event(output_event.to_dict())
                    self.ctxmgr.append_tool_result(run_id, tc['tool_name'], error_result, session_id, tool_call_id=tc['tool_call_id'])
                    await _mark_completed(tc['tool_call_id'])
                    continue  # Skip to next tool call

                # Run tool guardrails
                if self.guardrails.has_tool_guardrails(tc['tool_name']):
                    ctx = {'run_id': run_id, 'session_id': session_id, 'tool_name': tc['tool_name']}
                    passed, modified_args, error = await self.guardrails.check_tool(tc['tool_name'], tool_args, ctx)
                    if not passed:
                        error_event = ErrorEvent(error=f"Tool guardrail failed: {error}")
                        yield error_event.to_dict()
                        yield {'type': 'finish'}
                        loop_state['control'] = 'terminate'
                        return
                    tool_args = modified_args

                # Execute tool (streaming or non-streaming)
                # Track reasoning block ID for auto-injection
                _current_reasoning_id = None

                if tc['tool_call_id'] in prefetched:
                    # Already executed concurrently above. Re-raising the stored
                    # exception here rather than handling it separately means a
                    # parallel failure takes exactly the same path as a serial
                    # one — the except branch below turns it into a
                    # tool-output-error the model can recover from.
                    outcome = prefetched[tc['tool_call_id']]
                    if isinstance(outcome, BaseException):
                        raise outcome
                    result = outcome
                    yield wrap_event(
                        ToolOutputAvailableEvent(
                            tool_call_id=tc['tool_call_id'],
                            output=result,
                        ).to_dict()
                    )
                    tool_events = _empty_async_iter()
                else:
                    tool_events = tool.run_stream(tool_args, ctx=self.tool_context)

                async for event in tool_events:
                    if event.get('type') == 'tool-output':
                        # Final output from tool
                        result = event['output']
                        # Emit tool output event (V5 UI Stream Protocol)
                        output_event = ToolOutputAvailableEvent(
                            tool_call_id=tc['tool_call_id'],
                            output=result
                        )
                        yield wrap_event(output_event.to_dict())
                    else:
                        # Auto-inject ID for reasoning events (Vercel AI SDK requires it)
                        event_type = event.get('type', '')
                        if event_type.startswith('reasoning-'):
                            # Generate or reuse reasoning block ID
                            if 'id' not in event:
                                if event_type == 'reasoning-start' or _current_reasoning_id is None:
                                    _current_reasoning_id = str(uuid.uuid4())
                                event = {**event, 'id': _current_reasoning_id}

                            # Strip non-standard fields (AI SDK only allows type, id, delta)
                            # 'transient' is a Vel-internal hint, not part of AI SDK spec
                            allowed_keys = {'type', 'id', 'delta'}
                            event = {k: v for k, v in event.items() if k in allowed_keys}

                        # Custom artifact event (e.g., data-artifact-table-editor)
                        yield event

                # Validate final output (only if schema is non-empty)
                if result is not None and tool.output_schema:
                    validate_io(tool.output_schema, result)

                # Log tool execution for observability
                if trace_ctx and observer:
                    tool_latency = (time.time() - tool_start_time) * 1000
                    observer.log_tool(
                        current_step_ctx or trace_ctx,
                        ToolData(
                            tool_name=tc['tool_name'],
                            tool_call_id=tc['tool_call_id'],
                            input=tool_args,
                            output=result,
                            latency_ms=tool_latency,
                        )
                    )

                # Process inject_tools directive (dynamic tool injection)
                injected = self._process_inject_tools(result)
                if injected:
                    logger.debug(f"Injected tools for next LLM call: {injected}")

                # Process through custom handler if configured
                directive = self._process_tool_result(
                    tc['tool_name'], tool_args, result, run_id, steps, session_id
                )

                # Handle directive decision
                if directive.decision == ToolUseDecision.STOP:
                    yield wrap_event({'type': 'finish-step'})
                    yield {'type': 'finish'}
                    loop_state['control'] = 'terminate'
                    return
                elif directive.decision == ToolUseDecision.ERROR:
                    error_event = ErrorEvent(error=f"Tool handler returned ERROR for {tc['tool_name']}")
                    yield error_event.to_dict()
                    yield {'type': 'finish'}
                    loop_state['control'] = 'terminate'
                    return

                # Check if we should stop after this tool (non-custom behavior)
                if self.should_stop_after_tool(tc['tool_name']):
                    yield wrap_event({'type': 'finish-step'})
                    yield {'type': 'finish'}
                    # Every other terminal branch sets this, and the docstring
                    # above promises it. Without it `_step_loop` reads
                    # loop_state.get('control') as None, falls through to
                    # `continue`, and issues another LLM call — after `finish`
                    # has already gone out. On the harness resume/recover paths
                    # it is worse: 'control' is pre-seeded 'continue', so the
                    # run could never terminate here at all.
                    loop_state['control'] = 'terminate'
                    return  # Don't add to context or continue loop

                # Handle message modifications from directive
                if directive.replace_messages is not None:
                    self.ctxmgr._by_run[run_id] = directive.replace_messages
                elif directive.add_messages:
                    for msg in directive.add_messages:
                        if msg['role'] == 'system':
                            self.ctxmgr._by_run[run_id].insert(0, msg)
                        else:
                            self.ctxmgr._by_run[run_id].append(msg)

                # Add to context for next iteration (with tool_call_id for proper OpenAI format)
                self.ctxmgr.append_tool_result(run_id, tc['tool_name'], result, session_id, tool_call_id=tc['tool_call_id'])

                # Add reset tool choice message if enabled
                reset_msg = self._get_reset_tool_choice_message()
                if reset_msg:
                    self.ctxmgr._by_run[run_id].append(reset_msg)

                # Per-tool durable checkpoint (harness crash recovery; no-op when
                # the hook is unset, i.e. the non-harness path).
                await _mark_completed(tc['tool_call_id'])

            except Exception as e:
                failed_tool_args = locals().get('tool_args', tc.get('input', {}))
                tool_error_text = f"Tool execution failed: {str(e)}"
                # Log tool error for observability
                if trace_ctx and observer:
                    tool_latency = (time.time() - tool_start_time) * 1000
                    observer.log_tool(
                        current_step_ctx or trace_ctx,
                        ToolData(
                            tool_name=tc['tool_name'],
                            tool_call_id=tc['tool_call_id'],
                            input=failed_tool_args,
                            error=str(e),
                            latency_ms=tool_latency,
                        )
                    )
                # A raised tool is a recoverable outcome, not the end of the run.
                #
                # This used to emit a global `error` and terminate. Two things
                # went wrong with that. The tool call never reached a terminal
                # event, so every client left the tool part open — a spinner
                # that never resolves, made worse by ErrorEvent.to_dict()
                # stripping `details` so the toolCallId never reached the wire
                # at all. And the model never saw the failure, so it could not
                # retry, choose another approach, or explain the limitation.
                #
                # Both halves are needed and they solve different problems:
                # `tool-output-error` closes the visible tool part, and the
                # tool-result message lets the loop continue. This mirrors the
                # user-denial path above, which already did the right thing.
                #
                # An explicitly returned ToolUseDecision.ERROR stays terminal —
                # that is a deliberate directive, not a crash.
                error_result = {'error': tool_error_text}
                yield wrap_event(
                    ToolOutputErrorEvent(
                        tool_call_id=tc['tool_call_id'],
                        error_text=tool_error_text,
                    ).to_dict()
                )
                # Without this the assistant message announcing the tool call is
                # left with no matching `role: 'tool'` reply, which both OpenAI
                # and Anthropic reject if the transcript is ever resumed.
                self.ctxmgr.append_tool_result(
                    run_id, tc['tool_name'], error_result, session_id,
                    tool_call_id=tc['tool_call_id'],
                )
                await _mark_completed(tc['tool_call_id'])
                continue

        # End step span for observability
        if current_step_ctx and observer:
            observer.end_span(current_step_ctx, output={'status': 'continue', 'tools_executed': len(tool_calls)})
            current_step_ctx = None

        # Emit finish-step event (AI SDK v5 spec: simple event, no fields)
        yield wrap_event({'type': 'finish-step'})

        loop_state['control'] = 'continue'

    async def _stream_llm_call(
        self,
        messages: List[Dict[str, Any]],
        *,
        out: Dict[str, Any],
        tools: Optional[List[Dict[str, Any]]] = None,
        emit_as: str = 'text',
        reasoning_block_id: Optional[str] = None,
        generation_config: Optional[Dict[str, Any]] = None,
        model: Optional[str] = None,
        provider: Optional[Any] = None,
        wrap_event: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
        trace_ctx: Optional['SpanContext'] = None,
        current_step_ctx: Optional['SpanContext'] = None,
        observer: Optional[Any] = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """One LLM streaming call — the shared 'turn atom' for the loop engine.

        Streams a single provider call and forwards normalized stream events,
        optionally re-tagging the model's text output as ``reasoning-*`` for a
        thinking phase (``emit_as='reasoning'``). Because async generators cannot
        return values, results are reported via the mutable ``out`` dict:
        ``out['text']``, ``out['tool_calls']`` (``[{tool_call_id, tool_name,
        input}]``), ``out['usage']``, ``out['finish_reason']``,
        ``out['reasoning_block_id']``.

        This is deliberately NOT wired into ``_step_loop`` — the base loop keeps
        its battle-tested inline streaming (structured-output parsing, etc.). The
        atom exists so the reflection/loop engine can stop reimplementing a
        lower-fidelity LLM+event path and instead share this + the real
        ``_run_tool_calls``.
        """
        from .events import ReasoningStartEvent, ReasoningDeltaEvent, ReasoningEndEvent
        if wrap_event is None:
            wrap_event = lambda e: e
        if observer is None:
            observer = self._observer
        as_reasoning = emit_as == 'reasoning'
        rbid = reasoning_block_id or str(uuid.uuid4())

        provider = provider or self._get_provider()
        model = model or self.model_cfg['model']
        config = {**self.generation_config, **(generation_config or {})}

        full_text: List[str] = []
        tool_calls: List[Dict[str, Any]] = []
        usage = None
        finish_reason = 'stop'
        messages_for_obs = list(messages)
        llm_start = time.time()
        reasoning_open = False

        async for event in provider.stream(messages, model=model, tools=tools, generation_config=config):
            etype = event.type
            if etype == 'finish-message':
                finish_reason = event.finish_reason
                continue
            if etype == 'response-metadata':
                if getattr(event, 'usage', None):
                    usage = event.usage
                yield wrap_event(event.to_dict())
                continue
            if as_reasoning and etype in ('text-start', 'text-delta', 'text-end'):
                # Re-tag the model's text as reasoning for a thinking phase.
                if etype == 'text-delta':
                    if not reasoning_open:
                        reasoning_open = True
                        yield wrap_event(ReasoningStartEvent(block_id=rbid).to_dict())
                    full_text.append(event.delta)
                    yield wrap_event(ReasoningDeltaEvent(block_id=rbid, delta=event.delta).to_dict())
                elif etype == 'text-start':
                    if not reasoning_open:
                        reasoning_open = True
                        yield wrap_event(ReasoningStartEvent(block_id=rbid).to_dict())
                else:  # text-end
                    if reasoning_open:
                        reasoning_open = False
                        yield wrap_event(ReasoningEndEvent(block_id=rbid).to_dict())
                continue
            # Default: forward the event as-is and track content/tool calls.
            yield wrap_event(event.to_dict())
            if etype == 'text-delta':
                full_text.append(event.delta)
            elif etype == 'tool-input-available':
                tool_calls.append({
                    'tool_call_id': event.tool_call_id,
                    'tool_name': event.tool_name,
                    'input': event.input,
                })

        # Close a reasoning block the provider left open (defensive).
        if reasoning_open:
            yield wrap_event(ReasoningEndEvent(block_id=rbid).to_dict())

        out['text'] = ''.join(full_text)
        out['tool_calls'] = tool_calls
        out['usage'] = usage
        out['finish_reason'] = finish_reason
        out['reasoning_block_id'] = rbid

        # Log the generation under the step span (nesting-aware observer).
        if trace_ctx and observer:
            from .integrations.base import GenerationData
            observer.log_generation(
                current_step_ctx or trace_ctx,
                GenerationData(
                    model=model,
                    provider=self.model_cfg.get('provider', 'unknown'),
                    messages=messages_for_obs,
                    response=out['text'] or None,
                    tool_calls=[{'tool': tc['tool_name'], 'args': tc['input']} for tc in tool_calls] if tool_calls else None,
                    usage=usage,
                    latency_ms=(time.time() - llm_start) * 1000,
                ),
            )

    async def _synthesize_final(
        self,
        run_id: str,
        session_id: Optional[str],
        *,
        context: Optional[Dict[str, Any]] = None,
        wrap_event: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
        reason: str = 'max steps exceeded',
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Make one final tool-less LLM call to synthesize a partial answer.

        Shared by the legacy max-steps-exceeded path and Harness Mode budget
        exhaustion, so users always get a partial response instead of an error.

        Args:
            run_id: Active run identifier.
            session_id: Optional session id.
            context: Runtime context for a callable system_prompt.
            wrap_event: Event-metadata wrapper; identity if None.
            reason: Human-readable reason logged for the synthesis call.

        Yields:
            Stream protocol events for the final synthesized response.
        """
        if wrap_event is None:
            wrap_event = lambda event_dict: event_dict

        # This gives the user a partial answer rather than an error
        logger.warning(f'{reason}, making final synthesis call')

        # Add a system message to guide the final response
        synthesis_msg = {
            'role': 'user',
            'content': 'You have reached the maximum number of steps. Please synthesize a response based on the information you have gathered so far. Do not call any more tools.'
        }
        self.ctxmgr.append(run_id, synthesis_msg, session_id)

        # Make final LLM call without tools
        messages = self.ctxmgr.messages_for_llm(run_id, session_id)
        system_prompt = self._get_system_prompt(run_id, context=context)
        if system_prompt:
            messages = [{'role': 'system', 'content': system_prompt}] + messages

        provider = self._get_provider()

        # The synthesis call is a step like any other, so it opens one. Without
        # this the function still emitted `finish-step` at the end, leaving the
        # run with one more close than open — a client tracking step boundaries
        # sees a step end that never began, and groups the synthesized answer
        # into the previous step.
        yield wrap_event({'type': 'start-step'})

        # Stream the final response (no tools). Normalize provider event objects
        # to dicts so the synthesized output matches every other run_stream yield
        # (consumers/SSE receive dicts, never raw event objects).
        final_text = ''
        async for event in provider.stream(messages, self.model_cfg.get('model', 'gpt-4o'), tools=[]):
            event_type = event.type  # Events are objects with .type attribute, not dicts
            if event_type in ('text-delta', 'text-start', 'text-end'):
                yield event.to_dict() if hasattr(event, 'to_dict') else event
                if event_type == 'text-delta':
                    final_text += getattr(event, 'delta', '')
            elif event_type == 'finish-message':
                # Consumed, never forwarded — matching every other stream path
                # (`_step_loop` and `_stream_llm_call` both `continue` here).
                #
                # This one used to yield it, so a run that exhausted max_steps
                # emitted a part no other run does. `finish-message` is not in
                # the AI SDK UI Message Stream union at all, so a strict client
                # rejects it with unrecognized_keys -> invalid_union and throws
                # away the whole stream — the max-steps path, and only that
                # path, broke the browser.
                continue

        # Save final response to context
        if final_text:
            self.ctxmgr.append_assistant_message(run_id, final_text, session_id)

        yield wrap_event({'type': 'finish-step'})
        yield {'type': 'finish'}


    async def _run_with_thinking(
        self,
        input: Dict[str, Any],
        session_id: Optional[str],
        config: Any,  # ThinkingConfig
        generation_config: Optional[Dict[str, Any]] = None,
        context_refs: Optional[Any] = None,
        rlm: Optional[Dict[str, Any]] = None,
        observability_context: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None,
        stateless: bool = False,
        node_id: Optional[str] = None,
        external_run_id: Optional[str] = None,
        harness_cfg: Optional[Any] = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Execute with Extended Thinking enabled.

        Routes to ReflectionController for multi-pass reasoning:
        Analyze -> Critique -> Refine (adaptive) -> Conclude

        Args:
            input: Input dict with 'message' or 'messages'
            session_id: Optional session ID
            config: ThinkingConfig instance
            stateless: If True, don't mutate session state

        Yields:
            Stream protocol events
        """
        from .thinking import ReflectionController, ThinkingConfig
        from .thinking.router import effort_overrides, route_thinking

        if hasattr(config, 'to_dict'):
            config = ThinkingConfig(**config.to_dict())

        if getattr(config, 'routing', 'always') == 'never':
            async for event in self.run_stream(
                input,
                session_id=session_id,
                generation_config=generation_config,
                context_refs=context_refs,
                rlm=rlm,
                thinking=ThinkingConfig(mode='none'),
                observability_context=observability_context,
                context=context,
                stateless=stateless,
                node_id=node_id,
                external_run_id=external_run_id,
            ):
                yield event
            return

        question = self._extract_thinking_question(input)
        conversation_context = input.get('messages') if isinstance(input.get('messages'), list) else None

        if getattr(config, 'routing', 'always') == 'auto':
            router_model_cfg = getattr(config, 'router_model', None)
            router_provider = self._get_provider_for_model_config(router_model_cfg)
            router_model = (
                router_model_cfg.get('model', self.model_cfg['model'])
                if router_model_cfg
                else self.model_cfg['model']
            )
            try:
                decision = await route_thinking(
                    provider=router_provider,
                    model=router_model,
                    message=question,
                    context=conversation_context,
                    effort=getattr(config, 'effort', 'high'),
                    confidence_threshold=getattr(config, 'router_confidence_threshold', 0.8),
                )
            except Exception:
                decision = None
            if decision is None:
                async for event in self.run_stream(
                    input,
                    session_id=session_id,
                    generation_config=generation_config,
                    context_refs=context_refs,
                    rlm=rlm,
                    thinking=ThinkingConfig(mode='none'),
                    observability_context=observability_context,
                    context=context,
                    stateless=stateless,
                    node_id=node_id,
                    external_run_id=external_run_id,
                ):
                    yield event
                return
            if decision.mode == 'direct':
                async for event in self.run_stream(
                    input,
                    session_id=session_id,
                    generation_config=generation_config,
                    context_refs=context_refs,
                    rlm=rlm,
                    thinking=ThinkingConfig(mode='none'),
                    observability_context=observability_context,
                    context=context,
                    stateless=stateless,
                    node_id=node_id,
                    external_run_id=external_run_id,
                ):
                    yield event
                return

        effort = getattr(config, 'effort', 'high')
        overrides = effort_overrides(effort)
        config.max_refinements = overrides['max_refinements']
        config.confidence_threshold = overrides['confidence_threshold']

        run_id = str(uuid.uuid4())
        self.ctxmgr.set_input(run_id, input, session_id, stateless=stateless)

        # The unified reflection engine drives phases over the shared turn atom
        # (_stream_llm_call) + real tool round (_run_tool_calls), all nested under
        # one trace. Compose with Harness Mode: bound the refine loop by the
        # harness budget, and (when durable approval is on) suspend the whole
        # reflection run when a phase tool needs a human decision.
        from .harness.exceptions import SuspendRun
        budget = harness_cfg.budget if harness_cfg is not None else None
        harness_controller = None
        if harness_cfg is not None and harness_cfg.approval.enabled and harness_cfg.approval.mode == 'durable':
            from .harness import HarnessController
            harness_controller = HarnessController(agent=self, config=harness_cfg)
            harness_controller.bind_run(run_id=run_id, session_id=session_id, context=context)
        controller = ReflectionController(
            agent=self, config=config, budget=budget, harness_controller=harness_controller
        )

        # Track accumulated content for storage
        reasoning_parts = []
        answer_parts = []
        thinking_metadata = {}

        yield StartEvent().to_dict()
        yield StepStartEvent().to_dict()

        answer_step_started = False

        # Trace the whole reasoning run under one owned trace (phases nest as
        # THINKING spans). Fixes reflection being off the trace graph entirely.
        observer, trace_ctx = self._begin_leg_trace(run_id)
        suspended = False
        try:
            async for event in controller.run(
                question,
                context=conversation_context,
                trace_ctx=trace_ctx,
                observer=observer,
                parent_run_id=run_id,
                run_id=run_id,
            ):
                if event.get('type') == 'text-start' and not answer_step_started:
                    yield StepFinishEvent().to_dict()
                    yield StepStartEvent().to_dict()
                    answer_step_started = True
                yield event

                # Track for storage
                event_type = event.get('type')
                if event_type == 'reasoning-delta':
                    reasoning_parts.append(event.get('delta', ''))
                elif event_type == 'text-delta':
                    answer_parts.append(event.get('delta', ''))
                elif event_type == 'data-thinking-complete':
                    thinking_metadata = event.get('data', {})
        except SuspendRun as s:
            # A gated tool during a phase suspended the reflection run durably.
            suspended = True
            for ev in harness_controller.on_suspend(s):
                yield ev
        finally:
            self._end_leg_trace(observer, trace_ctx)

        if suspended:
            return

        # Save to context with multi-part message
        full_reasoning = ''.join(reasoning_parts)
        final_answer = ''.join(answer_parts)

        self.ctxmgr.append_assistant_with_reasoning(
            run_id,
            full_reasoning,
            final_answer,
            thinking_metadata,
            session_id
        )

        yield StepFinishEvent().to_dict()

        # Emit finish
        yield FinishEvent().to_dict()

    def _extract_thinking_question(self, input: Dict[str, Any]) -> str:
        """Extract the latest user prompt for thinking/routing."""
        message = input.get('message')
        if isinstance(message, str):
            return message
        messages = input.get('messages')
        if isinstance(messages, list):
            for item in reversed(messages):
                if isinstance(item, dict) and item.get('role') == 'user':
                    content = item.get('content', '')
                    if isinstance(content, str):
                        return content
                    if isinstance(content, list):
                        chunks = []
                        for part in content:
                            if isinstance(part, dict) and part.get('type') == 'text':
                                chunks.append(str(part.get('text') or ''))
                        return ''.join(chunks)
        return str(input)

    def _get_provider_for_model_config(self, model_config: Optional[Dict[str, Any]]):
        """Get a provider from an optional model config, falling back to the main provider."""
        if not model_config:
            return self._get_provider()

        provider_name = model_config.get('provider', self.model_cfg['provider'])
        api_key = model_config.get('api_key')

        if provider_name == self.model_cfg['provider'] and not api_key:
            return self._get_provider()

        from .providers import OpenAIProvider, GeminiProvider, AnthropicProvider

        if api_key:
            if provider_name == 'openai':
                return OpenAIProvider(api_key=api_key)
            if provider_name == 'google':
                return GeminiProvider(api_key=api_key)
            if provider_name == 'anthropic':
                return AnthropicProvider(api_key=api_key)

        return self.providers.get(provider_name) or self._get_provider()

async def run_stream(agent: 'Agent', input: Dict[str, Any]):
    """Helper function for streaming"""
    async for e in agent.run_stream(input):
        yield e
