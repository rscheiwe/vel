from __future__ import annotations
import asyncio
import inspect
from typing import Any, Dict, Callable, Optional, AsyncGenerator, List
from jsonschema import validate, Draft202012Validator
from .schema_generator import (
    generate_input_schema_from_function,
    generate_output_schema_from_function
)

class ToolSpec:
    def __init__(
        self,
        name: str,
        input_schema: Dict[str,Any],
        output_schema: Dict[str,Any],
        handler: Callable,
        description: str = None,
        enabled: Any = True,  # bool or Callable[[dict], bool]
        # Per-tool policies
        timeout: Optional[float] = None,  # Timeout in seconds
        retries: int = 0,  # Number of retry attempts
        fallback: Optional[str] = None,  # "return_error" | "call_other_tool" | handler
        _unpack_args: bool = False  # Internal: whether to unpack input dict as kwargs
    ):
        self.name = name
        self.input_schema = input_schema
        self.output_schema = output_schema
        self._handler = handler
        # Use explicit description, or fall back to input_schema description, or generate from name
        self.description = description or input_schema.get('description', f'Tool: {name}')
        # Detect if handler is an async generator function
        self._is_async_generator = inspect.isasyncgenfunction(handler)
        # Conditional enablement
        self._enabled = enabled
        # Per-tool policies
        self.timeout = timeout
        self.retries = retries
        self.fallback = fallback
        # Whether handler expects **kwargs unpacking (from_function style)
        self._unpack_args = _unpack_args

    def is_enabled(self, ctx: Optional[Dict[str, Any]] = None) -> bool:
        """
        Check if this tool is enabled for the given context.

        Args:
            ctx: Context dict with user info, session data, etc.

        Returns:
            True if tool should be available, False otherwise
        """
        if callable(self._enabled):
            return self._enabled(ctx or {})
        return bool(self._enabled)

    def is_streaming(self) -> bool:
        """Returns True if this tool yields events during execution (async generator handler)"""
        return self._is_async_generator

    @classmethod
    def from_function(
        cls,
        fn: Callable,
        name: Optional[str] = None,
        input_schema: Optional[Dict[str, Any]] = None,
        output_schema: Optional[Dict[str, Any]] = None,
        description: Optional[str] = None,
        validate_output: bool = False,
        **kwargs
    ) -> 'ToolSpec':
        """
        Create ToolSpec from a Python function with auto-generated schemas.

        By default, output validation is disabled (permissive). Set validate_output=True
        to enable strict output validation based on return type hints.

        Args:
            fn: The function to wrap as a tool
            name: Tool name (defaults to function name)
            input_schema: Override auto-generated input schema
            output_schema: Override auto-generated output schema
            description: Tool description (defaults to function docstring)
            validate_output: If False (default), accept any output. If True, validate against schema.
            **kwargs: Additional ToolSpec parameters (enabled, timeout, retries, fallback)

        Returns:
            ToolSpec instance ready to be passed to Agent

        Examples:
            # Simple tool (no output validation)
            def get_weather(city: str) -> dict:
                '''Get weather for a city.'''
                return {'temp': 72, 'condition': 'sunny'}

            tool = ToolSpec.from_function(get_weather)

            # Tool with strict output validation
            tool = ToolSpec.from_function(
                get_weather,
                validate_output=True,
                output_schema={
                    'type': 'object',
                    'properties': {
                        'temp': {'type': 'number'},
                        'condition': {'type': 'string'}
                    },
                    'required': ['temp', 'condition']
                }
            )

            # Pass directly to agent (no registration!)
            agent = Agent(
                id='my-agent',
                model={'provider': 'openai', 'model': 'gpt-4o'},
                tools=[tool]
            )
        """
        tool_name = name or fn.__name__
        tool_desc = description or (fn.__doc__.strip() if fn.__doc__ else f'Tool: {tool_name}')

        # Auto-generate input schema from function signature
        if input_schema is None:
            input_schema = generate_input_schema_from_function(fn)

        # Output schema defaults to permissive (no validation)
        if output_schema is None:
            if validate_output:
                # Try to infer from return type hint
                output_schema = generate_output_schema_from_function(fn, permissive=False)
            else:
                # Default: accept anything (empty schema = no validation)
                output_schema = {}

        return cls(
            name=tool_name,
            input_schema=input_schema,
            output_schema=output_schema,
            handler=fn,
            description=tool_desc,
            _unpack_args=True,  # from_function handlers expect **kwargs, not (input, ctx)
            **kwargs
        )

    async def run(self, input: Dict[str,Any], ctx: Dict[str,Any]) -> Dict[str,Any]:
        """Execute non-streaming tool (returns single result)"""
        if self._unpack_args:
            # from_function style: handler expects **kwargs (e.g., email_send(to, subject, body))
            if asyncio.iscoroutinefunction(self._handler):
                return await self._handler(**input)
            return self._handler(**input)
        else:
            # Traditional style: handler expects (input, ctx) signature
            if asyncio.iscoroutinefunction(self._handler):
                return await self._handler(input, ctx)
            return self._handler(input, ctx)

    async def run_stream(self, input: Dict[str,Any], ctx: Dict[str,Any]) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Execute streaming tool (yields events during execution).

        Yields:
            - Custom artifact events (e.g., {"type": "data-artifact-table-editor", "data": {...}})
            - Final event: {"type": "tool-output", "output": {...}}
        """
        if not self._is_async_generator:
            # Non-streaming tool: wrap result in single yield
            result = await self.run(input, ctx)
            yield {"type": "tool-output", "output": result}
        else:
            # Streaming tool: yield all events from async generator
            if self._unpack_args:
                # from_function style: handler expects **kwargs
                async for event in self._handler(**input):
                    yield event
            else:
                # Traditional style: handler expects (input, ctx)
                async for event in self._handler(input, ctx):
                    yield event

class ToolRegistry:
    _global_instance: Optional['ToolRegistry'] = None

    def __init__(self):
        self._tools: Dict[str, ToolSpec] = {}

    @classmethod
    def default(cls) -> 'ToolRegistry':
        """Return the global shared registry."""
        if cls._global_instance is None:
            cls._global_instance = cls()
        return cls._global_instance

    def register(self, tool: ToolSpec):
        self._tools[tool.name] = tool

    def get(self, name: str) -> ToolSpec:
        if name not in self._tools:
            raise KeyError(f"Tool '{name}' not found in registry. Available tools: {list(self._tools.keys())}")
        return self._tools[name]

    def schemas(self, ctx: Optional[Dict[str, Any]] = None, filter_tools: Optional[List[str]] = None):
        """
        Get schemas for enabled tools, optionally filtered by name.

        Args:
            ctx: Context dict for evaluating conditional enablement
            filter_tools: Optional list of tool names to include.
                         - If None: Return all registered tools (backward compatible)
                         - If []: Return no tools (empty dict)
                         - If ['tool1', 'tool2']: Return only those tools

        Returns:
            Dict of tool schemas (only enabled tools matching filter_tools)
        """
        tools_to_include = self._tools.items()

        # If filter_tools is provided, only include those tools
        if filter_tools is not None:
            tools_to_include = [(name, t) for name, t in self._tools.items() if name in filter_tools]

        return {
            name: {'input': t.input_schema, 'output': t.output_schema, 'description': t.description}
            for name, t in tools_to_include
            if t.is_enabled(ctx)
        }

_registry = ToolRegistry.default()

def register_tool(tool: ToolSpec):
    """
    Register a tool in the global registry.

    .. deprecated:: 0.3.0
        Global tool registration is deprecated and will be removed in v2.0.
        Pass ToolSpec instances directly to Agent instead:

        Old (deprecated):
            register_tool(tool)
            agent = Agent(tools=['tool_name'])

        New (recommended):
            tool = ToolSpec.from_function(my_function)
            agent = Agent(tools=[tool])
    """
    import warnings
    warnings.warn(
        f"register_tool() is deprecated and will be removed in Vel v2.0. "
        f"Pass ToolSpec instances directly to Agent instead:\n"
        f"  tool = ToolSpec.from_function({tool.name})\n"
        f"  agent = Agent(tools=[tool])\n"
        f"See examples/dynamic_tools.py for migration examples.",
        DeprecationWarning,
        stacklevel=2
    )
    _registry.register(tool)

def validate_io(schema: Dict[str,Any], value: Dict[str,Any]):
    Draft202012Validator.check_schema(schema)
    validate(instance=value, schema=schema)
