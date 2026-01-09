---
description: "Create a new tool with auto-generated schema"
argument-hint: "[tool-name]"
---

Scaffold a new tool for Vel agents.

## Instructions

1. Create a tool function with proper type hints and docstring:

   ```python
   from typing import Optional, List
   from vel import ToolSpec

   def $1(
       param1: str,
       param2: Optional[int] = None,
       _context: dict = None  # Filtered from schema, injected at runtime
   ) -> dict:
       """
       Brief description of what the tool does.

       Args:
           param1: Description of param1
           param2: Description of param2 (optional)

       Returns:
           Description of return value
       """
       # Implementation
       return {'result': ...}

   # Create ToolSpec
   ${1}_tool = ToolSpec.from_function($1)
   ```

2. For async tools:

   ```python
   async def $1(query: str) -> dict:
       """Async tool description."""
       result = await some_async_operation(query)
       return {'data': result}
   ```

3. For streaming tools:

   ```python
   async def $1(query: str):
       """Streaming tool description."""
       async for item in stream_source(query):
           yield {'item': item}
   ```

4. Add the tool to an agent:

   ```python
   from vel import Agent

   agent = Agent(
       model={'provider': 'openai', 'model': 'gpt-4o'},
       tools=[${1}_tool]
   )
   ```

5. Add tests in `tests/test_tools.py`:

   ```python
   @pytest.mark.asyncio
   async def test_$1():
       tool = ToolSpec.from_function($1)
       result = await tool.execute({'param1': 'test'})
       assert 'result' in result
   ```

## Best Practices

- Use type hints for all parameters (schema auto-generation)
- Write clear docstrings (used as tool descriptions)
- Use `_context` for accessing run context
- Prefer returning dicts over complex objects
- Handle errors gracefully with informative messages

## Reference

See `vel/tools/registry.py` and `.claude/rules/tools-development.md`
