from __future__ import annotations
import asyncio
import json
import warnings
import logging
import uuid
from typing import Any, AsyncGenerator, Dict, List, Optional, Literal, Union

# Configure logger for error surfacing
logger = logging.getLogger('vel.agent')
from .core import State, reduce, ContextManager
from .core.tool_behavior import (
    ToolUseBehavior, ToolUseDecision, ToolEvent, ToolUseDirective, HandoffConfig
)
from .core.guardrails import GuardrailEngine, GuardrailError
from .core.structured_output import (
    StructuredOutputPolicy, StructuredOutputValidationError,
    parse_structured_output, get_retry_prompt, get_json_mode_system_prompt
)
from .core.hooks import HookRegistry
from .providers import ProviderRegistry
from .tools import ToolRegistry, ToolSpec, validate_io
from .events import (
    StreamEvent, StartEvent, FinishEvent, ToolInputAvailableEvent, ToolOutputAvailableEvent,
    ErrorEvent, FinishMessageEvent, StepStartEvent, StepFinishEvent,
    ObjectElementEvent, ObjectPartialEvent, ObjectCompleteEvent
)
from .core.json_stream_parser import (
    IncrementalJsonParser, detect_output_mode, get_element_type, OutputMode,
    StreamedElement, PartialObject
)
from .prompts import PromptContextManager, PromptTemplate

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

        # Lifecycle hooks
        self.hooks = HookRegistry(hooks)

        # Dynamic instructions
        self.instruction = instruction

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
        system_prompt_override: Optional[str] = None
    ) -> 'ToolSpec':
        """
        Expose this agent as a tool that can be used by other agents.

        Args:
            name: Tool name (defaults to agent ID)
            description: Tool description
            system_prompt_override: Optional system prompt to override agent's default

        Returns:
            ToolSpec that wraps this agent
        """
        from .tools import ToolSpec

        tool_name = name or self.id
        tool_desc = description or f"Run the {self.id} agent"

        # Create handler that calls this agent
        async def agent_tool_handler(input: Dict[str, Any], ctx: Dict[str, Any]) -> Dict[str, Any]:
            # Get message from input
            message = input.get('message', input.get('query', str(input)))

            # Apply system prompt override if provided
            original_ctxmgr = None
            if system_prompt_override:
                # Store original and create modified context
                original_ctxmgr = self.ctxmgr
                # This is a simplified approach - a full implementation would
                # properly inject the system prompt override
                pass

            try:
                # Run the agent
                result = await self.run({'message': message})

                # Return result in structured format
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
            finally:
                if original_ctxmgr:
                    self.ctxmgr = original_ctxmgr

        return ToolSpec(
            name=tool_name,
            input_schema={
                'type': 'object',
                'properties': {
                    'message': {
                        'type': 'string',
                        'description': f'Message to send to the {tool_name} agent'
                    }
                },
                'required': ['message']
            },
            output_schema={
                'type': 'object',
                'properties': {
                    'response': {'type': 'string'}
                }
            },
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

    async def _call_tool(self, tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a tool (instance or global registry)"""
        # Get tool from instance registry or global registry
        tool = self._get_tool(tool_name)

        # Always validate input
        validate_io(tool.input_schema, args)

        # Execute tool
        result = await tool.run(args, ctx=self.tool_context)

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
        rlm: Optional[Dict[str, Any]] = None
    ) -> Union[str, Dict[str, Any]]:
        """
        Non-streaming run - returns final answer or raw tool output.

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
                self.ctxmgr.set_input(run_id, input, session_id)

        state = State(run_id=run_id)
        event: Dict[str, Any] = {'kind':'start'}
        steps = 0
        final_answer = ''
        structured_output_attempts = 0
        last_valid_output = None

        try:
            while True:
                state, effects = reduce(state, event)
                for eff in effects:
                    if eff.kind == 'call_llm':
                        step = await self._call_llm_generate(run_id, session_id, generation_config)
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

                        result = await self._call_tool(tool_name, tool_args)

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
                                break

                        # Add assistant response to context
                        self.ctxmgr.append_assistant_message(run_id, final_answer, session_id)
                        return final_answer
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
                    system_prompt = self._get_system_prompt(run_id)
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
            raise

    async def run_stream(
        self,
        input: Dict[str, Any],
        session_id: Optional[str] = None,
        generation_config: Optional[Dict[str, Any]] = None,
        context_refs: Optional[Any] = None,
        rlm: Optional[Dict[str, Any]] = None,
        thinking: Optional[Any] = None
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Streaming run - yields stream protocol events as they occur.

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

        # If Extended Thinking is enabled, route to thinking controller
        if thinking_config and thinking_config.mode == 'reflection':
            async for event in self._run_with_thinking(input, session_id, thinking_config):
                yield event
            return

        run_id = str(uuid.uuid4())
        self.ctxmgr.set_input(run_id, input, session_id)

        # Clear any injected tools from previous runs
        self._clear_injected_tools()

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
                self.ctxmgr.set_input(run_id, input, session_id)

        # Emit start event (V5 UI Stream Protocol)
        yield StartEvent().to_dict()

        steps = 0
        max_steps = self.policies.get('max_steps', 24)
        structured_output_attempts = 0

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

                # If we got text and no tool calls, we're done
                if full_text and not tool_calls:
                    answer = ''.join(full_text)

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
                                # Retry: add error message and continue loop
                                retry_prompt = get_retry_prompt(self.output_type, e)
                                self.ctxmgr._by_run[run_id].append({'role': 'system', 'content': retry_prompt})
                                continue  # Go back to LLM

                    self.ctxmgr.append_assistant_message(run_id, answer, session_id)

                    # Emit finish-step event (AI SDK v5 spec: simple event, no fields)
                    yield {'type': 'finish-step'}

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

                    for tc in tool_calls:
                        try:
                            # Get tool to check if it's streaming (instance or global)
                            tool = self._get_tool(tc['tool_name'])
                            tool_args = tc['input']
                            result = None

                            # Run tool guardrails
                            if self.guardrails.has_tool_guardrails(tc['tool_name']):
                                ctx = {'run_id': run_id, 'session_id': session_id, 'tool_name': tc['tool_name']}
                                passed, modified_args, error = await self.guardrails.check_tool(tc['tool_name'], tool_args, ctx)
                                if not passed:
                                    error_event = ErrorEvent(error=f"Tool guardrail failed: {error}")
                                    yield error_event.to_dict()
                                    yield {'type': 'finish'}
                                    return
                                tool_args = modified_args

                            # Execute tool (streaming or non-streaming)
                            # Track reasoning block ID for auto-injection
                            _current_reasoning_id = None

                            async for event in tool.run_stream(tool_args, ctx=self.tool_context):
                                if event.get('type') == 'tool-output':
                                    # Final output from tool
                                    result = event['output']
                                    # Emit tool output event (V5 UI Stream Protocol)
                                    output_event = ToolOutputAvailableEvent(
                                        tool_call_id=tc['tool_call_id'],
                                        output=result
                                    )
                                    yield output_event.to_dict()
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
                                yield {'type': 'finish-step'}
                                yield {'type': 'finish'}
                                return
                            elif directive.decision == ToolUseDecision.ERROR:
                                error_event = ErrorEvent(error=f"Tool handler returned ERROR for {tc['tool_name']}")
                                yield error_event.to_dict()
                                yield {'type': 'finish'}
                                return

                            # Check if we should stop after this tool (non-custom behavior)
                            if self.should_stop_after_tool(tc['tool_name']):
                                yield {'type': 'finish-step'}
                                yield {'type': 'finish'}
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

            # Max steps exceeded - make one final LLM call WITHOUT tools to synthesize a response
            # This gives the user a partial answer rather than an error
            logger.warning(f'max steps ({max_steps}) exceeded, making final synthesis call')

            # Add a system message to guide the final response
            synthesis_msg = {
                'role': 'user',
                'content': 'You have reached the maximum number of steps. Please synthesize a response based on the information you have gathered so far. Do not call any more tools.'
            }
            self.ctxmgr.append(run_id, synthesis_msg, session_id)

            # Make final LLM call without tools
            messages = self.ctxmgr.messages_for_llm(run_id, session_id)
            system_prompt = self._get_system_prompt(run_id)
            if system_prompt:
                messages = [{'role': 'system', 'content': system_prompt}] + messages

            provider = self._get_provider()

            # Stream the final response (no tools)
            final_text = ''
            async for event in provider.stream(messages, self.model_cfg.get('model', 'gpt-4o'), tools=[]):
                event_type = event.get('type')
                if event_type in ('text-delta', 'text-start', 'text-end'):
                    yield event
                    if event_type == 'text-delta':
                        final_text += event.get('delta', '')
                elif event_type == 'finish-message':
                    yield event

            # Save final response to context
            if final_text:
                self.ctxmgr.append_assistant_message(run_id, final_text, session_id)

            yield {'type': 'finish-step'}
            yield {'type': 'finish'}

        except asyncio.CancelledError:
            raise
        except Exception as e:
            # Ensure error message is never empty
            error_msg = str(e) if str(e) else f"{type(e).__name__}: {repr(e)}"
            logger.error(f"Agent stream error: {error_msg}", exc_info=True)
            error_event = ErrorEvent(error=error_msg)
            yield error_event.to_dict()
            raise

    async def _run_with_thinking(
        self,
        input: Dict[str, Any],
        session_id: Optional[str],
        config: Any  # ThinkingConfig
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Execute with Extended Thinking enabled.

        Routes to ReflectionController for multi-pass reasoning:
        Analyze -> Critique -> Refine (adaptive) -> Conclude

        Args:
            input: Input dict with 'message' or 'messages'
            session_id: Optional session ID
            config: ThinkingConfig instance

        Yields:
            Stream protocol events
        """
        from .thinking import ReflectionController

        run_id = str(uuid.uuid4())
        self.ctxmgr.set_input(run_id, input, session_id)

        # Emit start event
        yield StartEvent().to_dict()

        # Get provider (use thinking_model if specified)
        thinking_provider = self._get_thinking_provider(config)
        model = (
            config.thinking_model.get('model', self.model_cfg['model'])
            if config.thinking_model
            else self.model_cfg['model']
        )

        # Get tool schemas if thinking_tools enabled
        tools = self._get_tool_schemas() if config.thinking_tools else None

        # Create controller
        controller = ReflectionController(
            provider=thinking_provider,
            model=model,
            config=config,
            tools=tools,
            tool_executor=self._call_tool
        )

        # Track accumulated content for storage
        reasoning_parts = []
        answer_parts = []
        thinking_metadata = {}

        # Extract question from input
        question = input.get('message', str(input))

        # Stream events from controller
        async for event in controller.run(question):
            yield event

            # Track for storage
            event_type = event.get('type')
            if event_type == 'reasoning-delta':
                reasoning_parts.append(event.get('delta', ''))
            elif event_type == 'text-delta':
                answer_parts.append(event.get('delta', ''))
            elif event_type == 'data-thinking-complete':
                thinking_metadata = event.get('data', {})

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

        # Emit finish
        yield FinishEvent().to_dict()

    def _get_thinking_provider(self, config: Any):
        """
        Get provider for thinking steps.

        If config.thinking_model is set and has a different provider,
        creates a new provider instance. Otherwise, uses the main provider.
        """
        if not config.thinking_model:
            return self._get_provider()

        thinking_provider_name = config.thinking_model.get('provider', self.model_cfg['provider'])
        thinking_api_key = config.thinking_model.get('api_key')

        # If same provider and no custom API key, use main provider
        if thinking_provider_name == self.model_cfg['provider'] and not thinking_api_key:
            return self._get_provider()

        # Create new provider instance for thinking
        from .providers import OpenAIProvider, GeminiProvider, AnthropicProvider

        if thinking_api_key:
            if thinking_provider_name == 'openai':
                return OpenAIProvider(api_key=thinking_api_key)
            elif thinking_provider_name == 'google':
                return GeminiProvider(api_key=thinking_api_key)
            elif thinking_provider_name == 'anthropic':
                return AnthropicProvider(api_key=thinking_api_key)
            else:
                # Fall back to main provider
                return self._get_provider()
        else:
            # Different provider but no custom key - use registry
            return self.providers.get(thinking_provider_name)


async def run_stream(agent: 'Agent', input: Dict[str, Any]):
    """Helper function for streaming"""
    async for e in agent.run_stream(input):
        yield e
