---
layout: default
title: Tools
nav_order: 6
---

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
from vel import Agent, ToolSpec, register_tool

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
    description: str             # Tool description (optional, helps LLM decide when to use)
    input_schema: Dict[str, Any] # JSON Schema for input validation
    output_schema: Dict[str, Any] # JSON Schema for output validation
    handler: Callable            # Function to execute (sync or async)
```

### Parameters

**name** (required)
- Unique identifier for the tool
- Used by agent to reference tool
- Convention: lowercase_with_underscores

**description** (optional but recommended)
- Human-readable description of what the tool does and when to use it
- Helps the LLM decide when to invoke the tool
- If not provided, falls back to `input_schema['description']`, or defaults to `f'Tool: {name}'`
- **Best practice**: Be explicit and specific about the tool's purpose and use cases

**input_schema** (required)
- JSON Schema (Draft 2020-12) defining expected input
- Must include `type`, `properties`, and `required` fields
- Can include top-level `description` field as fallback for tool description
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
from vel import ToolSpec, register_tool

def add_numbers_handler(input: dict, ctx: dict) -> dict:
    a = input['a']
    b = input['b']
    return {'result': a + b}

add_tool = ToolSpec(
    name='add_numbers',
    description='Add two numbers together and return the sum',  # ← Explicit description
    input_schema={
        'type': 'object',
        'properties': {
            'a': {'type': 'number', 'description': 'First number'},
            'b': {'type': 'number', 'description': 'Second number'}
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

**Alternative: Description in input_schema**

```python
# If you don't provide explicit description parameter,
# Vel falls back to input_schema['description']
add_tool = ToolSpec(
    name='add_numbers',
    input_schema={
        'type': 'object',
        'description': 'Add two numbers together and return the sum',  # ← Fallback description
        'properties': {
            'a': {'type': 'number'},
            'b': {'type': 'number'}
        },
        'required': ['a', 'b']
    },
    output_schema={...},
    handler=add_numbers_handler
)
```

### Async Tool

```python
import asyncio
from vel import ToolSpec, register_tool

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

The `ctx` parameter provides runtime context to tools. It contains both **built-in context** (automatically provided by the agent) and **custom resources** (injected via `tool_context`).

### Built-in Context

Every tool automatically receives runtime metadata:

```python
def context_aware_handler(input: dict, ctx: dict) -> dict:
    """Tool that uses built-in context"""
    run_id = ctx.get('run_id')  # Current run ID
    session_id = ctx.get('session_id')  # Session ID (if any)
    agent_id = ctx.get('agent_id')  # Agent ID

    # Use context for logging, tracking, etc.
    print(f"Tool called in run {run_id} by agent {agent_id}")

    return {'status': 'ok'}
```

**Built-in Context Keys:**
- `run_id`: Unique run identifier
- `session_id`: Session ID (if using sessions)
- `agent_id`: Agent identifier
- `input`: Original user input

### Custom Resource Injection

Use the `tool_context` parameter to inject shared resources into tools (dependency injection pattern):

```python
from vel import Agent

# Create shared resources
db_connection = get_database_connection()
storage = MessageBasedStorage(messages)

# Inject resources via tool_context
agent = Agent(
    id='my-agent',
    model={'provider': 'openai', 'model': 'gpt-4o'},
    tools=['query_database', 'manage_storage'],
    tool_context={
        'db': db_connection,      # Database connection
        'storage': storage,        # Storage backend
        'user_id': 'user_123',    # User context
        'config': app_config      # Configuration
    }
)
```

**Tool accesses resources:**

```python
async def query_database_handler(input: dict, ctx: dict) -> dict:
    # Access injected database connection
    db = ctx.get('db')
    if not db:
        return {'error': 'Database not available'}

    results = await db.query(input['table'])
    return {'results': results}
```

### Why Use tool_context?

**✅ Flexibility**
- Same tool works with different backends
- Swap implementations without changing tool code

```python
# Development: Mock database
agent = Agent(tools=['query_db'], tool_context={'db': MockDatabase()})

# Production: Real database
agent = Agent(tools=['query_db'], tool_context={'db': PostgresDatabase()})
```

**✅ Per-Request Isolation**
- Different agent instances have different contexts
- Perfect for multi-tenant applications

```python
# User A's agent
agent_a = Agent(
    tools=['get_data'],
    tool_context={'user_id': 'user_a', 'tenant': 'acme_corp'}
)

# User B's agent (different context)
agent_b = Agent(
    tools=['get_data'],
    tool_context={'user_id': 'user_b', 'tenant': 'widget_inc'}
)
```

**✅ Testability**
- Easy to mock resources in tests
- No global state to manage

```python
# Test with mock
mock_storage = MockStorage()
agent = Agent(tools=['manage_data'], tool_context={'storage': mock_storage})
```

**✅ No Global Variables**
- Resources passed explicitly
- Better code organization and thread safety

### Common Use Cases

**1. Database Connections**

```python
db = await get_db_connection()
agent = Agent(
    tools=['query_users', 'update_record'],
    tool_context={'db': db}
)

async def query_users_handler(input, ctx):
    db = ctx.get('db')
    return await db.query('users', input['filter'])
```

**2. Storage Backends**

```python
from server.llm.artifacts.storage import MessageBasedStorage

storage = MessageBasedStorage(messages)
agent = Agent(
    tools=['tableEditor'],
    tool_context={'storage': storage}
)

async def table_editor_handler(input, ctx):
    storage = ctx.get('storage')
    artifact = await storage.get_artifact()
    # ... work with artifact
```

**3. API Clients**

```python
api_client = ExternalAPIClient(api_key=settings.API_KEY)
agent = Agent(
    tools=['fetch_external_data'],
    tool_context={'api': api_client}
)
```

**4. User Context & Permissions**

```python
agent = Agent(
    tools=['delete_file', 'update_settings'],
    tool_context={
        'user_id': current_user.id,
        'permissions': current_user.permissions,
        'org_id': current_user.organization_id
    }
)

def delete_file_handler(input, ctx):
    if 'delete' not in ctx.get('permissions', []):
        return {'error': 'Permission denied'}
    # ... proceed with deletion
```

**5. Multiple Resources**

```python
agent = Agent(
    tools=['complex_operation'],
    tool_context={
        'db': db_connection,
        'cache': redis_client,
        'storage': s3_client,
        'config': app_config,
        'user_id': user_id
    }
)
```

### Example: Artifact Storage Tool

Real-world example from artifact streaming implementation:

```python
# Endpoint creates storage and injects it
from server.llm.artifacts.storage import MessageBasedStorage

async def generate_answer_endpoint():
    messages = chat.messages if chat else []
    storage = MessageBasedStorage(messages)

    agent = Agent(
        id='table-editor-agent:v1',
        model={'provider': 'openai', 'model': 'gpt-4o'},
        tools=['tableEditor'],
        tool_context={'storage': storage}  # Inject storage
    )

    async for event in agent.run_stream({'messages': messages}):
        yield event

# Tool accesses storage
async def table_editor_tool_handler(input, ctx):
    storage = ctx.get('storage')

    # Get existing artifact
    current_artifact = await storage.get_artifact()

    # Perform operations
    # ...

    # Return result
    return {'artifact_id': artifact_id, 'status': 'complete'}
```

### Best Practices

**1. Check for Required Resources**

```python
def my_tool_handler(input, ctx):
    db = ctx.get('db')
    if not db:
        return {'error': 'Database connection required but not provided'}

    # ... use db safely
```

**2. Provide Defaults**

```python
def my_tool_handler(input, ctx):
    config = ctx.get('config', {'env': 'dev', 'debug': False})
    # ... use config with defaults
```

**3. Document Dependencies**

```python
async def query_database_handler(input, ctx):
    """
    Query database tool.

    Required ctx keys:
    - db: Database connection instance with .query() method

    Optional ctx keys:
    - timeout: Query timeout in seconds (default: 30)
    """
    db = ctx.get('db')
    timeout = ctx.get('timeout', 30)
    # ...
```

**4. Keep Context Lean**

Only pass what tools actually need:

```python
# ❌ Too much
tool_context={'everything': entire_app_state}

# ✅ Specific resources
tool_context={'db': db, 'storage': storage}
```

**See Also:**
- Full example: `examples/tool_context_injection.py`
- Artifact storage implementation: `server/llm/artifacts/storage.py`

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

## Tool Organization & Imports

### How the Registry Works

Vel uses a **global tool registry**. When you call `register_tool()`, the tool is added to this global registry and becomes available to **all** agents in your application.

```python
from vel import register_tool, ToolSpec

# This registers the tool globally
register_tool(my_tool)

# Now ANY agent can use it by name
agent = Agent(tools=['my_tool'])
```

### Pattern 1: Inline Registration (Simple)

Register tools directly in your main file:

```python
# my_agent.py
from vel import Agent, ToolSpec, register_tool

# Define and register tool inline
weather_tool = ToolSpec(
    name='get_weather',
    input_schema={...},
    output_schema={...},
    handler=lambda inp, ctx: {'temp_f': 72}
)
register_tool(weather_tool)

# Create agent
agent = Agent(
    id='my-agent',
    model={'provider': 'openai', 'model': 'gpt-4o'},
    tools=['get_weather']  # Available immediately after registration
)
```

**When to use:**
- Simple applications
- Few tools (1-3)
- Quick prototyping

### Pattern 2: Separate Module (Recommended)

Organize tools in separate files and import them:

**File: `tools/weather.py`**
```python
from vel import ToolSpec, register_tool

def weather_handler(input: dict, ctx: dict) -> dict:
    return {'temp_f': 72, 'condition': 'sunny'}

weather_tool = ToolSpec(
    name='get_weather',
    input_schema={...},
    output_schema={...},
    handler=weather_handler
)

# Register tool when module is imported
register_tool(weather_tool)
```

**File: `tools/__init__.py`**
```python
# Import all tools to register them
from .weather import weather_tool
from .search import search_tool
from .email import email_tool

# Re-export for convenience
__all__ = ['weather_tool', 'search_tool', 'email_tool']
```

**File: `my_agent.py`**
```python
from vel import Agent

# Import tools module (registers all tools automatically)
import tools

# Or import specific tools
from tools.weather import weather_tool

# Create agent - tools are already registered
agent = Agent(
    id='my-agent',
    model={'provider': 'openai', 'model': 'gpt-4o'},
    tools=['get_weather', 'search', 'send_email']
)
```

**When to use:**
- Production applications
- Multiple tools (4+)
- Team collaboration
- Reusable tool libraries

### Pattern 3: Conditional Registration

Register tools only when needed:

```python
# tools/web_search.py
from vel import ToolSpec, register_tool
import os

def create_web_search_tool():
    """Only register if API key is available"""
    if not os.getenv('PERPLEXITY_API_KEY'):
        return None

    tool = ToolSpec(
        name='websearch',
        input_schema={...},
        output_schema={...},
        handler=web_search_handler
    )
    register_tool(tool)
    return tool

# Register on import (if key exists)
web_search_tool = create_web_search_tool()
```

### Important Rules

1. **Import Before Agent Creation**: Tools must be imported/registered **before** creating the agent

```python
# ✓ Good: Import first
from tools.weather import weather_tool
agent = Agent(tools=['get_weather'])

# ✗ Bad: Import after agent creation
agent = Agent(tools=['get_weather'])  # KeyError: 'get_weather' not found!
from tools.weather import weather_tool
```

2. **Registration Happens Once**: Tools are registered when the module is imported
   - First import: Tool is registered
   - Subsequent imports: Tool already registered (no duplicates)

3. **Global Registry**: All agents share the same tool registry
   - Registering a tool makes it available to **all** agents
   - You cannot have agent-specific tools (by design)

### Real-World Example

Here's how the Perplexity web search tool is organized:

```python
# examples/multi_step_tools/web_search.py
"""Web Search Tool - Perplexity API Integration"""
from vel import ToolSpec, register_tool

async def web_search_handler(input: dict, ctx: dict) -> dict:
    # Implementation here
    pass

web_search_tool = ToolSpec(
    name='websearch',
    input_schema={...},
    output_schema={...},
    handler=web_search_handler
)

# Register automatically when imported
register_tool(web_search_tool)
```

```python
# my_research_agent.py
from vel import Agent

# Import tool (registers it)
from examples.multi_step_tools.web_search import web_search_tool

# Create agent
agent = Agent(
    id='research-agent',
    model={'provider': 'openai', 'model': 'gpt-4o'},
    tools=['websearch']  # Tool is already registered
)

# Use agent
result = await agent.run({'message': 'Search for AI trends'})
```

### Troubleshooting Imports

**Problem:** `KeyError: 'my_tool'`

**Solution:**
```python
# Check: Did you import the tool module?
from tools.my_tool import my_tool  # This registers it

# Check: Did you import before creating agent?
# imports must be at top of file

# Check: Is tool name spelled correctly?
agent = Agent(tools=['my_tool'])  # Must match ToolSpec.name
```

**Problem:** Tool registered twice with different implementations

**Solution:**
```python
# Vel allows re-registration (last one wins)
# To prevent confusion, use unique names or check before registering:

from vel import get_tool_registry

registry = get_tool_registry()
if 'my_tool' not in registry:
    register_tool(my_tool)
```

## Examples

See `examples/test_both_modes.py` for complete tool usage demonstration:

```bash
python examples/test_both_modes.py
```

See `examples/perplexity_web_search_example.py` for real-world tool import example:

```bash
python examples/perplexity_web_search_example.py
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
