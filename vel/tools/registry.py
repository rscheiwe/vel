from __future__ import annotations
import asyncio
import inspect
from typing import Any, Dict, Callable, Optional, AsyncGenerator
from jsonschema import validate, Draft202012Validator

class ToolSpec:
    def __init__(self, name: str, input_schema: Dict[str,Any], output_schema: Dict[str,Any], handler: Callable, description: str = None):
        self.name = name
        self.input_schema = input_schema
        self.output_schema = output_schema
        self._handler = handler
        # Use explicit description, or fall back to input_schema description, or generate from name
        self.description = description or input_schema.get('description', f'Tool: {name}')
        # Detect if handler is an async generator function
        self._is_async_generator = inspect.isasyncgenfunction(handler)

    def is_streaming(self) -> bool:
        """Returns True if this tool yields events during execution (async generator handler)"""
        return self._is_async_generator

    async def run(self, input: Dict[str,Any], ctx: Dict[str,Any]) -> Dict[str,Any]:
        """Execute non-streaming tool (returns single result)"""
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

    def schemas(self):
        return {name: {'input': t.input_schema, 'output': t.output_schema, 'description': t.description} for name,t in self._tools.items()}

_registry = ToolRegistry.default()

def register_tool(tool: ToolSpec):
    _registry.register(tool)

def validate_io(schema: Dict[str,Any], value: Dict[str,Any]):
    Draft202012Validator.check_schema(schema)
    validate(instance=value, schema=schema)
