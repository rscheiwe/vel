from __future__ import annotations
import asyncio
from typing import Any, Dict, Callable
from jsonschema import validate, Draft202012Validator

class ToolSpec:
    def __init__(self, name: str, input_schema: Dict[str,Any], output_schema: Dict[str,Any], handler: Callable):
        self.name = name
        self.input_schema = input_schema
        self.output_schema = output_schema
        self._handler = handler

    async def run(self, input: Dict[str,Any], ctx: Dict[str,Any]) -> Dict[str,Any]:
        if asyncio.iscoroutinefunction(self._handler):
            return await self._handler(input, ctx)
        return self._handler(input, ctx)

class ToolRegistry:
    def __init__(self):
        self._tools: Dict[str, ToolSpec] = {}

    @classmethod
    def default(cls) -> 'ToolRegistry':
        reg = cls()
        reg.register(ToolSpec(
            name='get_weather',
            input_schema={'type':'object','properties':{'city':{'type':'string'}}, 'required':['city']},
            output_schema={'type':'object','properties':{'temp_f':{'type':'number'}}, 'required':['temp_f']},
            handler=lambda inp, ctx: {'temp_f': 72.0}
        ))
        return reg

    def register(self, tool: ToolSpec):
        self._tools[tool.name] = tool

    def get(self, name: str) -> ToolSpec:
        if name not in self._tools:
            raise KeyError(f"Tool '{name}' not found in registry. Available tools: {list(self._tools.keys())}")
        return self._tools[name]

    def schemas(self):
        return {name: {'input': t.input_schema, 'output': t.output_schema} for name,t in self._tools.items()}

_registry = ToolRegistry.default()

def register_tool(tool: ToolSpec):
    _registry.register(tool)

def validate_io(schema: Dict[str,Any], value: Dict[str,Any]):
    Draft202012Validator.check_schema(schema)
    validate(instance=value, schema=schema)
