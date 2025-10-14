---
layout: default
title: Translation API
nav_order: 8
---

# Translation API

## Overview

The Vel Translation API provides a clean interface for external libraries to use Vel's event translation without reimplementing provider-specific logic. This allows libraries like Mesh to leverage Vel's standardized stream protocol events across multiple LLM providers (OpenAI, Anthropic, Google).

## Installation

```bash
pip install vel
```

## Quick Start

```python
from vel import get_translator

# Get translator for OpenAI
translator = get_translator("openai")

# Prepare messages
messages = [
    {"role": "user", "content": "Tell me a joke"}
]

# Stream events in Vel's standardized format
async for event in translator.translate_stream(messages, model="gpt-4"):
    if event.type == "text-delta":
        print(event.delta, end="", flush=True)
```

## API Reference

### `get_translator(provider_name: str) -> EventTranslator`

Get an event translator for a specific provider.

**Parameters:**
- `provider_name`: Name of the provider (`"openai"`, `"anthropic"`, `"google"`)

**Returns:**
- `EventTranslator` instance

**Example:**
```python
from vel import get_translator

translator = get_translator("openai")
```

### `available_providers() -> list[str]`

Get list of available provider names.

**Returns:**
- List of provider names

**Example:**
```python
from vel import available_providers

providers = available_providers()
print(providers)  # ['openai', 'anthropic', 'google']
```

### `EventTranslator`

Main class for translating provider events to Vel's stream protocol.

#### `translate_stream(messages, model, tools=None)`

Translate provider's native stream to Vel stream protocol events.

**Parameters:**
- `messages`: List of chat messages in format `[{"role": "user", "content": "..."}]`
- `model`: Model identifier (e.g., `"gpt-4"`, `"claude-3-opus"`)
- `tools`: Optional tool definitions (default: `None`)

**Yields:**
- `StreamEvent` objects

**Example:**
```python
translator = get_translator("openai")

messages = [{"role": "user", "content": "Hello!"}]

async for event in translator.translate_stream(messages, "gpt-4"):
    print(event.type, event.to_dict())
```

#### `translate_stream_to_dicts(messages, model, tools=None)`

Same as `translate_stream()` but yields dictionaries instead of StreamEvent objects.

**Example:**
```python
async for event_dict in translator.translate_stream_to_dicts(messages, "gpt-4"):
    print(event_dict)
    # {'type': 'text-delta', 'id': '...', 'delta': 'Hello'}
```

## Event Types

Vel's stream protocol includes the following event types:

| Event Type | Description |
|------------|-------------|
| `start` | Message generation begins |
| `text-start` | Text block starts |
| `text-delta` | Text chunk arrives |
| `text-end` | Text block completes |
| `tool-input-start` | Tool call begins |
| `tool-input-delta` | Tool argument chunk |
| `tool-input-available` | Tool arguments ready |
| `tool-output-available` | Tool execution result |
| `finish-message` | Message complete |
| `error` | Error occurred |

See [Stream Protocol Documentation](https://rscheiwe.github.io/vel/stream-protocol.html) for complete details.

## Use Cases

### Integration with Graph Orchestration Libraries

```python
# In an orchestration library like Mesh
from vel import get_translator

class AgentNode:
    def __init__(self, provider: str, model: str):
        self.translator = get_translator(provider)
        self.model = model

    async def execute(self, messages):
        full_response = ""

        async for event in self.translator.translate_stream(messages, self.model):
            if event.type == "text-delta":
                full_response += event.delta
                # Emit to graph executor
                await self.emit_token(event.delta)

        return full_response
```

### Multi-Provider Support

```python
from vel import get_translator, available_providers

# Support all available providers
for provider_name in available_providers():
    translator = get_translator(provider_name)

    # Use the same code for all providers
    async for event in translator.translate_stream(messages, model):
        handle_event(event)
```

### Custom Event Processing

```python
from vel import get_translator

translator = get_translator("anthropic")

messages = [{"role": "user", "content": "Analyze this data"}]

# Collect different event types
text_chunks = []
tool_calls = []

async for event in translator.translate_stream(messages, "claude-3-opus"):
    if event.type == "text-delta":
        text_chunks.append(event.delta)
    elif event.type == "tool-input-available":
        tool_calls.append({
            "name": event.tool_name,
            "args": event.input
        })
    elif event.type == "finish-message":
        print(f"Completed with reason: {event.finish_reason}")

full_text = "".join(text_chunks)
```

## Supported Providers

### OpenAI
- Models: GPT-4, GPT-3.5, etc.
- Requires: `OPENAI_API_KEY` environment variable

### Anthropic
- Models: Claude 3 (Opus, Sonnet, Haiku)
- Requires: `ANTHROPIC_API_KEY` environment variable

### Google
- Models: Gemini Pro, Gemini Flash
- Requires: `GOOGLE_API_KEY` environment variable

## Error Handling

```python
from vel import get_translator

try:
    translator = get_translator("openai")

    async for event in translator.translate_stream(messages, "gpt-4"):
        if event.type == "error":
            print(f"Error occurred: {event.error}")
            break
        # Handle other events

except ValueError as e:
    print(f"Provider not available: {e}")
```

## Advanced Usage

### With Tool Calls

```python
from vel import get_translator

translator = get_translator("openai")

messages = [{"role": "user", "content": "What's the weather in Paris?"}]

tools = {
    "get_weather": {
        "input": {
            "type": "object",
            "properties": {
                "city": {"type": "string"}
            },
            "required": ["city"]
        }
    }
}

async for event in translator.translate_stream(messages, "gpt-4", tools=tools):
    if event.type == "tool-input-available":
        print(f"Tool: {event.tool_name}")
        print(f"Args: {event.input}")
```

## Benefits

1. **Single Source of Truth**: All provider translation logic in one place
2. **Automatic Updates**: New providers and features propagate automatically
3. **Consistent Interface**: Same event structure across all providers
4. **Battle-Tested**: Used in production by Vel agents
5. **Minimal Dependencies**: Only requires Vel package

## See Also

- [Vel Stream Protocol](https://rscheiwe.github.io/vel/stream-protocol.html)
- [Vel Documentation](https://github.com/rscheiwe/vel)
- [Vercel AI SDK Stream Protocol](https://ai-sdk.dev/docs/ai-sdk-ui/stream-protocol)
