# Tools

Complete guide to the Vel tool system for enabling function calling in agents.

## Overview

Tools allow agents to perform actions and retrieve information beyond text generation. The Vel tool system provides:

- **JSON Schema Validation**: Automatic input/output validation
- **Async Support**: Both sync and async tool handlers
- **Type Safety**: Schema-enforced parameter types
- **Provider Agnostic**: Works with OpenAI, Gemini, and Claude
- **Simple Registration**: Global tool registry

## Quick Start

```python
from agents import Agent, ToolSpec, register_tool

# 1. Define tool handler
def get_weather_handler(input: dict, ctx: dict) -> dict:
    city = input['city']
    # Your logic here
    return {'temp_f': 72, 'condition': 'sunny', 'city': city}

# 2. Create ToolSpec
weather_tool = ToolSpec(
    name='get_weather',
    input_schema={
        'type': 'object',
        'properties': {'city': {'type': 'string'}},
        'required': ['city']
    },
    output_schema={
        'type': 'object',
        'properties': {
            'temp_f': {'type': 'number'},
            'condition': {'type': 'string'},
            'city': {'type': 'string'}
        },
        'required': ['temp_f', 'condition', 'city']
    },
    handler=get_weather_handler
)

# 3. Register tool
register_tool(weather_tool)

# 4. Use with agent
agent = Agent(
    id='my-agent',
    model={'provider': 'openai', 'model': 'gpt-4o'},
    tools=['get_weather']  # Tool names
)

# Agent will automatically call tool when needed
answer = await agent.run({'message': 'What is the weather in San Francisco?'})
```

## ToolSpec

### Structure

```python
class ToolSpec:
    name: str                    # Unique tool identifier
    input_schema: Dict[str, Any] # JSON Schema for input validation
    output_schema: Dict[str, Any] # JSON Schema for output validation
    handler: Callable            # Function to execute (sync or async)
```

### Parameters

**name** (required)
- Unique identifier for the tool
- Used by agent to reference tool
- Convention: lowercase_with_underscores

**input_schema** (required)
- JSON Schema (Draft 2020-12) defining expected input
- Must include `type`, `properties`, and `required` fields
- Automatically validated before calling handler

**output_schema** (required)
- JSON Schema defining expected output structure
- Validates handler return value
- Ensures consistent tool behavior

**handler** (required)
- Function that executes the tool logic
- Signature: `(input: dict, ctx: dict) -> dict`
- Can be sync or async (auto-detected)

## Creating Tools

### Basic Tool

```python
from agents import ToolSpec, register_tool

def add_numbers_handler(input: dict, ctx: dict) -> dict:
    a = input['a']
    b = input['b']
    return {'result': a + b}

add_tool = ToolSpec(
    name='add_numbers',
    input_schema={
        'type': 'object',
        'properties': {
            'a': {'type': 'number'},
            'b': {'type': 'number'}
        },
        'required': ['a', 'b']
    },
    output_schema={
        'type': 'object',
        'properties': {
            'result': {'type': 'number'}
        },
        'required': ['result']
    },
    handler=add_numbers_handler
)

register_tool(add_tool)
```

### Async Tool

```python
import asyncio
from agents import ToolSpec, register_tool

async def fetch_data_handler(input: dict, ctx: dict) -> dict:
    """Async tool with I/O operations"""
    url = input['url']

    # Simulate async I/O
    await asyncio.sleep(0.1)

    return {
        'status': 200,
        'data': f"Fetched from {url}"
    }

fetch_tool = ToolSpec(
    name='fetch_data',
    input_schema={
        'type': 'object',
        'properties': {
            'url': {'type': 'string', 'format': 'uri'}
        },
        'required': ['url']
    },
    output_schema={
        'type': 'object',
        'properties': {
            'status': {'type': 'integer'},
            'data': {'type': 'string'}
        },
        'required': ['status', 'data']
    },
    handler=fetch_data_handler
)

register_tool(fetch_tool)
```

### Tool with Complex Schema

```python
def search_handler(input: dict, ctx: dict) -> dict:
    query = input['query']
    filters = input.get('filters', {})
    limit = input.get('limit', 10)

    # Your search logic
    results = [
        {'title': 'Result 1', 'score': 0.95},
        {'title': 'Result 2', 'score': 0.87}
    ]

    return {
        'results': results[:limit],
        'total': len(results)
    }

search_tool = ToolSpec(
    name='search',
    input_schema={
        'type': 'object',
        'properties': {
            'query': {
                'type': 'string',
                'description': 'Search query'
            },
            'filters': {
                'type': 'object',
                'properties': {
                    'category': {'type': 'string'},
                    'date_range': {'type': 'string'}
                }
            },
            'limit': {
                'type': 'integer',
                'minimum': 1,
                'maximum': 100,
                'default': 10
            }
        },
        'required': ['query']
    },
    output_schema={
        'type': 'object',
        'properties': {
            'results': {
                'type': 'array',
                'items': {
                    'type': 'object',
                    'properties': {
                        'title': {'type': 'string'},
                        'score': {'type': 'number'}
                    },
                    'required': ['title', 'score']
                }
            },
            'total': {'type': 'integer'}
        },
        'required': ['results', 'total']
    },
    handler=search_handler
)

register_tool(search_tool)
```

## Using Tools

### Single Tool

```python
agent = Agent(
    id='my-agent',
    model={'provider': 'openai', 'model': 'gpt-4o'},
    tools=['get_weather']  # Single tool
)

answer = await agent.run({'message': 'What is the weather in Tokyo?'})
```

### Multiple Tools

```python
agent = Agent(
    id='my-agent',
    model={'provider': 'openai', 'model': 'gpt-4o'},
    tools=['get_weather', 'search', 'add_numbers']  # Multiple tools
)

answer = await agent.run({'message': 'Search for weather APIs and add 5 + 3'})
```

### Tools with Streaming

```python
agent = Agent(
    id='my-agent',
    model={'provider': 'openai', 'model': 'gpt-4o'},
    tools=['get_weather']
)

async for event in agent.run_stream({'message': 'Weather in London?'}):
    if event['type'] == 'tool-input-available':
        print(f"Tool called: {event['tool_name']}")
        print(f"Input: {event['input']}")
    elif event['type'] == 'tool-output-available':
        print(f"Tool result: {event['output']}")
    elif event['type'] == 'text-delta':
        print(event['delta'], end='', flush=True)
```

## Built-in Tools

Vel includes a default `get_weather` tool for testing:

```python
# Automatically registered
default_tool = ToolSpec(
    name='get_weather',
    input_schema={
        'type': 'object',
        'properties': {'city': {'type': 'string'}},
        'required': ['city']
    },
    output_schema={
        'type': 'object',
        'properties': {'temp_f': {'type': 'number'}},
        'required': ['temp_f']
    },
    handler=lambda inp, ctx: {'temp_f': 72.0}
)
```

**Note:** Override by registering your own `get_weather` tool.

## Tool Context

The `ctx` parameter provides runtime context to tools:

```python
def context_aware_handler(input: dict, ctx: dict) -> dict:
    """Tool that uses context"""
    run_id = ctx.get('run_id')  # Current run ID
    session_id = ctx.get('session_id')  # Session ID (if any)
    agent_id = ctx.get('agent_id')  # Agent ID

    # Use context for logging, tracking, etc.
    print(f"Tool called in run {run_id} by agent {agent_id}")

    return {'status': 'ok'}
```

**Available Context Keys:**
- `run_id`: Unique run identifier
- `session_id`: Session ID (if using sessions)
- `agent_id`: Agent identifier
- `input`: Original user input

## JSON Schema Validation

### Input Validation

Automatic validation before calling handler:

```python
# Schema defines number
input_schema={
    'type': 'object',
    'properties': {'count': {'type': 'number'}},
    'required': ['count']
}

# If LLM provides string, validation fails
# {"count": "five"} ❌ ValidationError
# {"count": 5} ✓ Valid
```

### Output Validation

Automatic validation of handler return value:

```python
# Schema expects specific structure
output_schema={
    'type': 'object',
    'properties': {
        'success': {'type': 'boolean'},
        'message': {'type': 'string'}
    },
    'required': ['success', 'message']
}

# Handler must return matching structure
return {'success': True}  # ❌ Missing 'message'
return {'success': True, 'message': 'OK'}  # ✓ Valid
```

### Schema Best Practices

```python
# ✓ Good: Descriptive, constrained schemas
{
    'type': 'object',
    'properties': {
        'temperature': {
            'type': 'number',
            'description': 'Temperature in Fahrenheit',
            'minimum': -100,
            'maximum': 200
        },
        'units': {
            'type': 'string',
            'enum': ['fahrenheit', 'celsius'],
            'default': 'fahrenheit'
        }
    },
    'required': ['temperature']
}

# ✗ Bad: Vague, unconstrained
{
    'type': 'object',
    'properties': {
        'data': {'type': 'string'}  # Too generic
    }
}
```

## Error Handling

### Tool Execution Errors

```python
def safe_divide_handler(input: dict, ctx: dict) -> dict:
    try:
        a = input['a']
        b = input['b']
        result = a / b
        return {'result': result}
    except ZeroDivisionError:
        return {'error': 'Division by zero', 'result': None}
    except Exception as e:
        return {'error': str(e), 'result': None}

# Schema allows error field
output_schema={
    'type': 'object',
    'properties': {
        'result': {'type': ['number', 'null']},
        'error': {'type': 'string'}
    }
}
```

### Validation Errors

```python
from jsonschema.exceptions import ValidationError

try:
    answer = await agent.run({'message': 'Call the tool'})
except ValidationError as e:
    print(f"Tool validation failed: {e}")
```

## Advanced Usage

### Dynamic Tool Registration

```python
def create_api_tool(api_name: str, endpoint: str) -> ToolSpec:
    """Factory function to create API tools"""
    def handler(input: dict, ctx: dict) -> dict:
        # Call the API
        return {'response': f"Called {endpoint}"}

    return ToolSpec(
        name=f'call_{api_name}',
        input_schema={
            'type': 'object',
            'properties': {'params': {'type': 'object'}},
            'required': []
        },
        output_schema={
            'type': 'object',
            'properties': {'response': {'type': 'string'}},
            'required': ['response']
        },
        handler=handler
    )

# Register multiple API tools
for api in ['weather', 'maps', 'translate']:
    tool = create_api_tool(api, f'https://api.example.com/{api}')
    register_tool(tool)
```

### Tool Chaining

Agent automatically chains tools when needed:

```python
# Agent can call multiple tools in sequence
agent = Agent(
    id='my-agent',
    model={'provider': 'openai', 'model': 'gpt-4o'},
    tools=['get_weather', 'search', 'send_email'],
    policies={'max_steps': 10}  # Allow multi-step execution
)

# Agent might: search weather API → get weather → send email with results
answer = await agent.run({
    'message': 'Find the weather in Paris and email it to user@example.com'
})
```

### Tool Policies

Control tool execution with policies:

```python
agent = Agent(
    id='my-agent',
    model={'provider': 'openai', 'model': 'gpt-4o'},
    tools=['get_weather'],
    policies={
        'max_steps': 5,           # Maximum tool calls per run
        'timeout': 30,            # Timeout in seconds (future)
        'retry': True             # Retry failed tools (future)
    }
)
```

## Examples

See `examples/test_both_modes.py` for complete tool usage demonstration:

```bash
python examples/test_both_modes.py
```

## Troubleshooting

### Tool Not Found

**Error:**
```
KeyError: 'my_tool'
```

**Solution:**
- Ensure tool is registered before creating agent: `register_tool(tool)`
- Check tool name spelling in `tools=[]` parameter
- Verify tool name matches ToolSpec.name

### Validation Error

**Error:**
```
jsonschema.exceptions.ValidationError: 'city' is a required property
```

**Solution:**
- Check LLM is providing all required fields
- Verify schema matches handler expectations
- Add descriptions to help LLM understand parameters

### Tool Never Called

**Problem:** Agent generates text response instead of calling tool.

**Solutions:**
1. Make tool name and schema descriptive
2. Add explicit instructions in message: "Use the get_weather tool"
3. Verify tool is in `tools=[]` parameter
4. Check if provider supports function calling (all do)

### Async Tool Hangs

**Problem:** Async tool handler never completes.

**Solutions:**
1. Ensure all async operations use `await`
2. Add timeouts to async I/O operations
3. Check for deadlocks in async code
4. Use `asyncio.wait_for()` for timeout control

## Best Practices

### 1. Descriptive Schemas

```python
# ✓ Good: Helps LLM understand tool
input_schema={
    'type': 'object',
    'properties': {
        'city': {
            'type': 'string',
            'description': 'City name for weather lookup (e.g., "San Francisco")'
        }
    },
    'required': ['city']
}

# ✗ Bad: No context for LLM
input_schema={
    'type': 'object',
    'properties': {'city': {'type': 'string'}},
    'required': ['city']
}
```

### 2. Consistent Naming

```python
# ✓ Good: Verb_noun pattern
'get_weather', 'search_products', 'send_email'

# ✗ Bad: Unclear actions
'weather', 'products', 'email'
```

### 3. Error Fields

```python
# ✓ Good: Schema allows error responses
output_schema={
    'type': 'object',
    'properties': {
        'result': {'type': ['string', 'null']},
        'error': {'type': 'string'},
        'success': {'type': 'boolean'}
    },
    'required': ['success']
}
```

### 4. Idempotent Tools

```python
# ✓ Good: Safe to retry
def get_weather_handler(input: dict, ctx: dict) -> dict:
    # Read-only operation
    return fetch_weather(input['city'])

# ⚠ Caution: Side effects
def send_email_handler(input: dict, ctx: dict) -> dict:
    # May send duplicate emails if retried
    return send_email(input['to'], input['body'])
```

## Next Steps

- [Stream Protocol](stream-protocol.md) - Understand tool call events
- [API Reference](api-reference.md) - Complete API documentation
- [Providers](providers.md) - Provider-specific tool features
