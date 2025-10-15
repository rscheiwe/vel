---
layout: default
title: API Reference
nav_order: 10
---

# API Reference

Complete API documentation for all Vel classes and functions.

## Table of Contents

- [Agent](#agent)
- [ContextManager](#contextmanager)
- [StatelessContextManager](#statelesscontextmanager)
- [ToolSpec](#toolspec)
- [Providers](#providers)
- [Stream Events](#stream-events)
- [Helper Functions](#helper-functions)

---

## Agent

Main orchestrator class for running agents with LLM providers.

### Constructor

```python
Agent(
    id: str,
    model: Dict[str, Any],
    prompt_env: str = 'prod',
    tools: List[str] | None = None,
    policies: Dict[str, Any] | None = None,
    context_manager: Optional[ContextManager] = None,
    session_storage: Literal['memory', 'database'] = 'memory',
    generation_config: Optional[Dict[str, Any]] = None
)
```

**Parameters:**

**id** (required)
- Type: `str`
- Agent identifier (unique name for this agent)
- Example: `'my-agent'`, `'chat-general:v1'`

**model** (required)
- Type: `Dict[str, Any]`
- Model configuration with `provider` and `model` keys
- Example: `{'provider': 'openai', 'model': 'gpt-4o'}`
- Supported providers: `'openai'`, `'google'`, `'anthropic'`

**prompt_env**
- Type: `str`
- Default: `'prod'`
- Environment for prompts (for future prompt versioning)

**tools**
- Type: `List[str] | None`
- Default: `None`
- List of tool names to enable
- Example: `['get_weather', 'search']`
- Tools must be registered before creating agent

**policies**
- Type: `Dict[str, Any] | None`
- Default: `{'max_steps': 24, 'retry': {'attempts': 2}}`
- Execution policies controlling agent behavior
- Available policies:
  - `max_steps`: Maximum tool calls per run (default: 24)
  - `retry`: Retry configuration (future feature)

**context_manager**
- Type: `Optional[ContextManager]`
- Default: `None` (uses `ContextManager()`)
- Custom context manager instance for conversation memory
- Options:
  - `None`: Default full memory
  - `ContextManager()`: Full memory (explicit)
  - `ContextManager(max_history=10)`: Limited history
  - `StatelessContextManager()`: No memory
  - Custom subclass

**session_storage**
- Type: `Literal['memory', 'database']`
- Default: `'memory'`
- Where to persist session context
- Options:
  - `'memory'`: In-memory only (fast, not persistent)
  - `'database'`: Postgres-backed (persistent, requires `POSTGRES_DSN`)

**generation_config**
- Type: `Optional[Dict[str, Any]]`
- Default: `None`
- Model generation parameters (temperature, max_tokens, etc.)
- Common parameters:
  - `temperature`: float (0-2) - Sampling temperature
  - `max_tokens`: int - Maximum output tokens
  - `top_p`: float (0-1) - Nucleus sampling
  - `top_k`: int - Top-K sampling (Gemini, Anthropic)
  - `presence_penalty`: float (-2 to 2) - Penalize new tokens (OpenAI)
  - `frequency_penalty`: float (-2 to 2) - Penalize repeated tokens (OpenAI)
  - `stop`: List[str] - Stop sequences
  - `seed`: int - Reproducibility seed (OpenAI, Anthropic)
- Can be overridden per-run using `generation_config` parameter in `run()` or `run_stream()`

**Example:**

```python
from vel import Agent

# Basic agent
agent = Agent(
    id='my-agent',
    model={'provider': 'openai', 'model': 'gpt-4o'}
)

# Agent with tools and limited memory
agent = Agent(
    id='my-agent',
    model={'provider': 'google', 'model': 'gemini-1.5-pro'},
    tools=['get_weather', 'search'],
    context_manager=ContextManager(max_history=20),
    session_storage='database',
    policies={'max_steps': 10}
)
```

---

### Methods

#### run()

Non-streaming execution - returns final answer only.

```python
async def run(
    input: Dict[str, Any],
    session_id: Optional[str] = None,
    generation_config: Optional[Dict[str, Any]] = None
) -> str
```

**Parameters:**

**input** (required)
- Type: `Dict[str, Any]`
- Input dictionary with `'message'` field
- Example: `{'message': 'Hello'}`

**session_id**
- Type: `Optional[str]`
- Default: `None`
- Session ID for multi-turn conversations
- If provided, context persists across calls
- Example: `'user-123'`, `'conv-abc'`

**generation_config**
- Type: `Optional[Dict[str, Any]]`
- Default: `None`
- Per-run generation config that overrides agent-level config
- Example: `{'temperature': 0, 'max_tokens': 100}`
- See Agent constructor for supported parameters

**Returns:**
- Type: `str`
- Final answer from the agent

**Raises:**
- `RuntimeError`: If max_steps exceeded
- `Exception`: On LLM or tool errors

**Example:**

```python
# Single-turn
answer = await agent.run({'message': 'What is 2+2?'})
print(answer)  # "4"

# Multi-turn with sessions
session_id = 'user-123'
answer1 = await agent.run({'message': 'My name is Alice'}, session_id=session_id)
answer2 = await agent.run({'message': 'What is my name?'}, session_id=session_id)
print(answer2)  # "Your name is Alice"

# With per-run generation config override
answer3 = await agent.run(
    {'message': 'Explain quantum computing'},
    generation_config={'temperature': 0, 'max_tokens': 100}  # Deterministic, brief
)
```

---

#### run_stream()

Streaming execution - yields stream protocol events in real-time.

```python
async def run_stream(
    input: Dict[str, Any],
    session_id: Optional[str] = None,
    generation_config: Optional[Dict[str, Any]] = None
) -> AsyncGenerator[Dict[str, Any], None]
```

**Parameters:**

**input** (required)
- Type: `Dict[str, Any]`
- Input dictionary with `'message'` field
- Example: `{'message': 'Tell me a story'}`

**session_id**
- Type: `Optional[str]`
- Default: `None`
- Session ID for multi-turn conversations
- Context persists across calls if provided

**generation_config**
- Type: `Optional[Dict[str, Any]]`
- Default: `None`
- Per-run generation config that overrides agent-level config
- Example: `{'temperature': 0.9, 'max_tokens': 1000}`
- See Agent constructor for supported parameters

**Yields:**
- Type: `Dict[str, Any]`
- Stream protocol events as they occur
- Event types: `text-delta`, `tool-input-available`, `finish-message`, etc.

**Example:**

```python
# Stream text to console
async for event in agent.run_stream({'message': 'Tell me a joke'}):
    if event['type'] == 'text-delta':
        print(event['delta'], end='', flush=True)
    elif event['type'] == 'finish-message':
        print()  # Newline
        break

# With sessions
session_id = 'user-123'
async for event in agent.run_stream({'message': 'My name is Bob'}, session_id=session_id):
    if event['type'] == 'text-delta':
        print(event['delta'], end='', flush=True)

# With per-run generation config override
async for event in agent.run_stream(
    {'message': 'Write a haiku'},
    generation_config={'temperature': 0.9, 'max_tokens': 50}  # Creative, brief
):
    if event['type'] == 'text-delta':
        print(event['delta'], end='', flush=True)
```

---

## ContextManager

Manages conversation history with configurable memory behavior.

### Constructor

```python
ContextManager(
    max_history: Optional[int] = None,
    summarize: bool = False
)
```

**Parameters:**

**max_history**
- Type: `Optional[int]`
- Default: `None` (unlimited)
- Maximum number of messages to retain
- Implements sliding window (keeps last N messages)
- Example: `max_history=10` keeps last 10 messages (~5 turns)

**summarize**
- Type: `bool`
- Default: `False`
- Whether to summarize old messages (future feature, not implemented)

**Example:**

```python
from vel import Agent, ContextManager

# Full memory (default)
agent = Agent(
    id='my-agent',
    model={'provider': 'openai', 'model': 'gpt-4o'},
    context_manager=ContextManager()
)

# Limited memory (last 20 messages)
agent = Agent(
    id='my-agent',
    model={'provider': 'openai', 'model': 'gpt-4o'},
    context_manager=ContextManager(max_history=20)
)
```

---

### Methods

#### get_session_context()

Get all messages for a session.

```python
def get_session_context(session_id: str) -> List[Dict[str, Any]]
```

**Parameters:**
- `session_id`: Session identifier

**Returns:**
- List of message dictionaries with `role` and `content` fields

**Example:**

```python
context = agent.ctxmgr.get_session_context('user-123')
print(context)
# [
#   {'role': 'user', 'content': 'Hello'},
#   {'role': 'assistant', 'content': 'Hi there!'}
# ]
```

---

#### set_session_context()

Set messages for a session (used when loading from storage).

```python
def set_session_context(session_id: str, messages: List[Dict[str, Any]])
```

**Parameters:**
- `session_id`: Session identifier
- `messages`: List of message dictionaries

**Example:**

```python
messages = [
    {'role': 'user', 'content': 'Hello'},
    {'role': 'assistant', 'content': 'Hi there!'}
]
agent.ctxmgr.set_session_context('user-123', messages)
```

---

#### clear_session()

Clear all messages for a session.

```python
def clear_session(session_id: str)
```

**Parameters:**
- `session_id`: Session identifier

**Example:**

```python
agent.ctxmgr.clear_session('user-123')
```

---

## StatelessContextManager

Stateless context manager - no memory between calls.

### Constructor

```python
StatelessContextManager()
```

**No parameters.** Each call is completely independent.

**Example:**

```python
from vel import Agent, StatelessContextManager

agent = Agent(
    id='my-agent',
    model={'provider': 'openai', 'model': 'gpt-4o'},
    context_manager=StatelessContextManager()
)

# First call
answer1 = await agent.run({'message': 'My name is Alice'}, session_id='user-1')

# Second call - does NOT remember Alice
answer2 = await agent.run({'message': 'What is my name?'}, session_id='user-1')
# Agent will say it doesn't know
```

---

## ToolSpec

Specification for a custom tool with JSON schema validation.

### Constructor

```python
ToolSpec(
    name: str,
    input_schema: Dict[str, Any],
    output_schema: Dict[str, Any],
    handler: Callable[[Dict[str, Any], Dict[str, Any]], Dict[str, Any]]
)
```

**Parameters:**

**name** (required)
- Type: `str`
- Unique tool identifier
- Convention: `lowercase_with_underscores`
- Example: `'get_weather'`, `'search_products'`

**input_schema** (required)
- Type: `Dict[str, Any]`
- JSON Schema (Draft 2020-12) for input validation
- Must include `type`, `properties`, `required`
- Validated before calling handler

**output_schema** (required)
- Type: `Dict[str, Any]`
- JSON Schema for output validation
- Validates handler return value
- Ensures consistent behavior

**handler** (required)
- Type: `Callable[[Dict[str, Any], Dict[str, Any]], Dict[str, Any]]`
- Function that executes tool logic
- Signature: `(input: dict, ctx: dict) -> dict`
- Can be sync or async (auto-detected)

**Example:**

```python
from vel import ToolSpec, register_tool

def get_weather_handler(input: dict, ctx: dict) -> dict:
    city = input['city']
    # Your logic here
    return {
        'temp_f': 72,
        'condition': 'sunny',
        'city': city
    }

weather_tool = ToolSpec(
    name='get_weather',
    input_schema={
        'type': 'object',
        'properties': {
            'city': {'type': 'string', 'description': 'City name'}
        },
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

register_tool(weather_tool)
```

---

### Methods

#### run()

Execute the tool handler with validation.

```python
async def run(input: Dict[str, Any], ctx: Dict[str, Any]) -> Dict[str, Any]
```

**Parameters:**
- `input`: Tool input (validated against input_schema)
- `ctx`: Runtime context with `run_id`, `session_id`, `agent_id`

**Returns:**
- Tool output (validated against output_schema)

**Note:** Usually called internally by Agent, not directly.

---

## Providers

### BaseProvider

Abstract base class for LLM providers.

```python
class BaseProvider(ABC):
    name: str

    @abstractmethod
    async def stream(
        messages: List[LLMMessage],
        model: str,
        tools: Dict[str, Any]
    ) -> AsyncGenerator[StreamEvent, None]

    @abstractmethod
    async def generate(
        messages: List[LLMMessage],
        model: str,
        tools: Dict[str, Any]
    ) -> Dict[str, Any]
```

**Note:** Implement this to create custom providers.

---

### OpenAIProvider

OpenAI provider implementation.

**Environment Variables:**
- `OPENAI_API_KEY` (required)
- `OPENAI_API_BASE` (optional, default: `https://api.openai.com/v1`)

**Supported Models:**
- `gpt-4o`
- `gpt-4-turbo`
- `gpt-4`
- `gpt-3.5-turbo`

**Example:**

```python
# Set in .env
OPENAI_API_KEY=sk-...
OPENAI_API_BASE=https://api.openai.com/v1  # Optional

# Use with agent
agent = Agent(
    id='my-agent',
    model={'provider': 'openai', 'model': 'gpt-4o'}
)
```

---

### GeminiProvider

Google Gemini provider implementation.

**Environment Variables:**
- `GOOGLE_API_KEY` (required)

**Supported Models:**
- `gemini-1.5-pro`
- `gemini-1.5-flash`
- `gemini-pro`

**Example:**

```python
# Set in .env
GOOGLE_API_KEY=AIza...

# Use with agent
agent = Agent(
    id='my-agent',
    model={'provider': 'google', 'model': 'gemini-1.5-pro'}
)
```

---

### AnthropicProvider

Anthropic Claude provider implementation.

**Environment Variables:**
- `ANTHROPIC_API_KEY` (required)
- `ANTHROPIC_API_BASE` (optional, default: `https://api.anthropic.com`)

**Supported Models:**
- `claude-opus-4-20250514`
- `claude-sonnet-4-20250514`
- `claude-3-5-sonnet-20241022`
- `claude-3-5-haiku-20241022`
- `claude-3-opus-20240229`
- `claude-3-sonnet-20240229`
- `claude-3-haiku-20240307`

**Example:**

```python
# Set in .env
ANTHROPIC_API_KEY=sk-ant-...

# Use with agent
agent = Agent(
    id='my-agent',
    model={'provider': 'anthropic', 'model': 'claude-sonnet-4-20250514'}
)
```

---

## Stream Events

All stream events have a `type` field and extend the base `StreamEvent` class.

### Event Types

| Event | Fields |
|-------|--------|
| `start` | `messageId` (optional) |
| `text-start` | `id` (block ID) |
| `text-delta` | `id`, `delta` (text chunk) |
| `text-end` | `id` |
| `tool-input-start` | `toolCallId`, `toolName` |
| `tool-input-delta` | `toolCallId`, `inputTextDelta` |
| `tool-input-available` | `toolCallId`, `toolName`, `input` (object) |
| `tool-output-available` | `toolCallId`, `output` (any) |
| `finish-message` | `finishReason` |
| `error` | `error` (string) |

See [Stream Protocol](stream-protocol.md) for complete event documentation.

---

## Helper Functions

### register_tool()

Register a tool in the global registry.

```python
def register_tool(tool: ToolSpec)
```

**Parameters:**
- `tool`: ToolSpec instance

**Example:**

```python
from vel import ToolSpec, register_tool

tool = ToolSpec(
    name='my_tool',
    input_schema={...},
    output_schema={...},
    handler=my_handler
)

register_tool(tool)
```

---

### validate_io()

Validate data against a JSON schema.

```python
def validate_io(schema: Dict[str, Any], value: Dict[str, Any])
```

**Parameters:**
- `schema`: JSON Schema dictionary
- `value`: Data to validate

**Raises:**
- `jsonschema.exceptions.ValidationError`: If validation fails

**Example:**

```python
from vel import validate_io

schema = {
    'type': 'object',
    'properties': {'count': {'type': 'number'}},
    'required': ['count']
}

validate_io(schema, {'count': 5})  # OK
validate_io(schema, {'count': 'five'})  # Raises ValidationError
```

---

## Type Definitions

### LLMMessage

Message format for LLM providers.

```python
LLMMessage = Dict[str, Any]

# Structure:
{
    'role': 'user' | 'assistant' | 'system',
    'content': str
}
```

---

## Default Values

### Agent Defaults

```python
{
    'prompt_env': 'prod',
    'tools': None,
    'policies': {'max_steps': 24, 'retry': {'attempts': 2}},
    'context_manager': ContextManager(),
    'session_storage': 'memory'
}
```

### ContextManager Defaults

```python
{
    'max_history': None,  # Unlimited
    'summarize': False
}
```

### Policy Defaults

```python
{
    'max_steps': 24,
    'retry': {'attempts': 2}  # Future feature
}
```

---

## Complete Example

```python
import asyncio
from dotenv import load_dotenv
from vel import Agent, ContextManager, ToolSpec, register_tool

load_dotenv()

# Create custom tool
def calculate_handler(input: dict, ctx: dict) -> dict:
    a = input['a']
    b = input['b']
    op = input['operation']

    if op == 'add':
        result = a + b
    elif op == 'multiply':
        result = a * b
    else:
        result = 0

    return {'result': result}

calc_tool = ToolSpec(
    name='calculate',
    input_schema={
        'type': 'object',
        'properties': {
            'a': {'type': 'number'},
            'b': {'type': 'number'},
            'operation': {'type': 'string', 'enum': ['add', 'multiply']}
        },
        'required': ['a', 'b', 'operation']
    },
    output_schema={
        'type': 'object',
        'properties': {'result': {'type': 'number'}},
        'required': ['result']
    },
    handler=calculate_handler
)

register_tool(calc_tool)

# Create agent
agent = Agent(
    id='calculator-agent',
    model={'provider': 'openai', 'model': 'gpt-4o'},
    tools=['calculate'],
    context_manager=ContextManager(max_history=10),
    session_storage='memory',
    policies={'max_steps': 5}
)

async def main():
    session_id = 'session-1'

    # Turn 1
    answer1 = await agent.run(
        {'message': 'What is 5 + 3?'},
        session_id=session_id
    )
    print(answer1)

    # Turn 2: Streaming
    print("\nTurn 2:")
    async for event in agent.run_stream(
        {'message': 'Now multiply that by 2'},
        session_id=session_id
    ):
        if event['type'] == 'text-delta':
            print(event['delta'], end='', flush=True)
        elif event['type'] == 'tool-input-available':
            print(f"\n[Tool: {event['toolName']}({event['input']})]")

if __name__ == '__main__':
    asyncio.run(main())
```

---

## Next Steps

- [Getting Started](getting-started.md) - Quick start guide
- [Stream Protocol](stream-protocol.md) - Streaming event details
- [Tools](tools.md) - Tool system in depth
- [Providers](providers.md) - Provider configuration
