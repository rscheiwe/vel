---
paths:
  - "vel/tools/**/*.py"
description: "Tool definition and development guidelines"
---

# Tool Development Guidelines

## Preferred: ToolSpec.from_function()

Always use `ToolSpec.from_function()` for new tools:

```python
from vel import ToolSpec

def get_weather(city: str, units: str = "celsius") -> dict:
    """
    Get current weather for a city.

    Args:
        city: City name (e.g., "New York")
        units: Temperature units ("celsius" or "fahrenheit")

    Returns:
        Weather data with temp, condition, humidity
    """
    return {'temp': 22, 'condition': 'sunny', 'humidity': 45}

tool = ToolSpec.from_function(get_weather)
agent = Agent(tools=[tool])
```

## Deprecated: String References

Avoid using global registry with string references:

```python
# Deprecated - don't use
agent = Agent(tools=['get_weather'])

# Preferred - direct ToolSpec
agent = Agent(tools=[ToolSpec.from_function(get_weather)])
```

## Type Hints for Schema Generation

Schema is auto-generated from type hints. Use explicit types:

```python
from typing import List, Optional, Literal
from pydantic import BaseModel

# Simple types
def search(query: str, limit: int = 10) -> List[dict]:
    ...

# Literal for enums
def sort(order: Literal["asc", "desc"]) -> None:
    ...

# Pydantic for complex inputs
class SearchParams(BaseModel):
    query: str
    filters: Optional[dict] = None

def advanced_search(params: SearchParams) -> List[dict]:
    ...
```

## Per-Tool Policies

Configure tool-specific behavior:

```python
tool = ToolSpec.from_function(
    slow_api_call,
    timeout=30.0,           # 30 second timeout
    retries=3,              # Retry up to 3 times
    fallback='cached_data'  # Return cached data on failure
)
```

## Streaming Tools

For long-running operations, use async generators:

```python
async def stream_results(query: str):
    """Stream search results as they arrive."""
    async for result in search_api.stream(query):
        yield {'type': 'result', 'data': result}

tool = ToolSpec.from_function(stream_results)
```

## Context Access

Tools can access execution context via special parameters:

```python
def my_tool(query: str, _context: dict = None) -> dict:
    """
    Tool with context access.

    Args:
        query: Search query
        _context: Injected by Vel (not in schema)
    """
    run_id = _context.get('run_id')
    session_id = _context.get('session_id')
    return {'query': query, 'run_id': run_id}
```

Parameters named `ctx`, `_context`, or `context` are:
- Filtered from JSON schema (not sent to LLM)
- Populated at runtime with execution context

## Agent as Tool

Compose agents hierarchically:

```python
sub_agent = Agent(
    id='researcher',
    model={'provider': 'openai', 'model': 'gpt-4o'},
    tools=[search_tool]
)

parent_agent = Agent(
    id='orchestrator',
    model={'provider': 'openai', 'model': 'gpt-4o'},
    tools=[sub_agent.as_tool()]  # Sub-agent callable as tool
)
```

## Conditional Tools

Enable tools based on runtime conditions:

```python
def is_premium_user(ctx: dict) -> bool:
    return ctx.get('user_tier') == 'premium'

tool = ToolSpec.from_function(
    premium_feature,
    enabled=is_premium_user  # Only available for premium users
)
```

## Testing Tools

```python
@pytest.mark.asyncio
async def test_tool_execution():
    tool = ToolSpec.from_function(get_weather)

    # Test schema generation
    assert 'city' in tool.input_schema['properties']

    # Test execution
    result = await tool.execute({'city': 'NYC'})
    assert 'temp' in result
```

## Reference

- `vel/tools/registry.py` - ToolSpec definition
- `vel/tools/schema_generator.py` - Auto schema generation
- `vel/agent.py` - Tool execution in agent loop
