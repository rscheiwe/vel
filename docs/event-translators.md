---
layout: default
title: Event Translators
nav_order: 8
---

# Event Translators

## Overview

Vel Event Translators provide a clean API for translating native provider events to Vel's standardized stream protocol. These translators support:

1. **Direct API usage** (OpenAI Chat API, Anthropic Messages API, Google Gemini API)
2. **Agent SDKs** (OpenAI Agents SDK, and more)
3. **Consistent event formatting** across all providers
4. **Single source of truth** - no duplicated translation logic

Translators can be used:
- **Internally** by Vel providers (already integrated)
- **Externally** by orchestration libraries like Mesh

**Important:** Translators only translate events - they do NOT make API calls.

## Installation

```bash
pip install vel
```

## Available Translators

| Translator | Source | Use Case |
|------------|--------|----------|
| `OpenAIAPITranslator` | OpenAI Chat Completions API | Direct API calls to OpenAI |
| `OpenAIAgentsSDKTranslator` | OpenAI Agents SDK | Using OpenAI's agent framework |
| `AnthropicAPITranslator` | Anthropic Messages API | Direct API calls to Claude |
| `GeminiAPITranslator` | Google Gemini API | Direct API calls to Gemini |

## Quick Start

### OpenAI Chat API

```python
from vel import get_openai_api_translator
import httpx
import json

translator = get_openai_api_translator()

# Make API call (example with httpx)
async with httpx.AsyncClient() as client:
    async with client.stream('POST', 'https://api.openai.com/v1/chat/completions', ...) as response:
        async for line in response.aiter_lines():
            if line.startswith('data: '):
                chunk = json.loads(line[6:])
                vel_event = translator.translate_chunk(chunk)
                if vel_event:
                    print(vel_event.to_dict())
```

### OpenAI Agents SDK

```python
from vel import get_openai_agents_translator
from agents import Agent, Runner

translator = get_openai_agents_translator()

# Use OpenAI Agents SDK
result = Runner.run_streamed(agent, "Hello!")
async for native_event in result.stream_events():
    vel_event = translator.translate(native_event)
    if vel_event:
        print(vel_event.to_dict())
```

### Anthropic Messages API

```python
from vel import get_anthropic_translator

translator = get_anthropic_translator()

# Stream from Anthropic API (SSE format)
async for line in response.aiter_lines():
    if line.startswith('data: '):
        data = json.loads(line[6:])
        vel_event = translator.translate_event(data)
        if vel_event:
            yield vel_event
```

### Google Gemini API

```python
from vel import get_gemini_translator

translator = get_gemini_translator()

# Stream from Gemini
async for chunk in response:
    vel_event = translator.translate_chunk(chunk)
    if vel_event:
        yield vel_event
```

## API Reference

### Convenience Functions

All translators can be instantiated via convenience functions:

```python
from vel import (
    get_openai_api_translator,        # OpenAI Chat API
    get_openai_agents_translator,     # OpenAI Agents SDK
    get_anthropic_translator,         # Anthropic Messages API
    get_gemini_translator,            # Google Gemini API
)
```

### `get_openai_api_translator()`

Get a translator for OpenAI Chat Completions API.

**Returns:**
- `OpenAIAPITranslator` instance

**Example:**
```python
from vel import get_openai_api_translator
translator = get_openai_api_translator()
```

### `get_anthropic_translator()`

Get a translator for Anthropic Messages API.

**Returns:**
- `AnthropicAPITranslator` instance

**Example:**
```python
from vel import get_anthropic_translator
translator = get_anthropic_translator()
```

### `get_gemini_translator()`

Get a translator for Google Gemini API.

**Returns:**
- `GeminiAPITranslator` instance

**Example:**
```python
from vel import get_gemini_translator
translator = get_gemini_translator()
```

### `get_openai_agents_translator()`

Get a translator for OpenAI Agents SDK events.

**Returns:**
- `OpenAIAgentsSDKTranslator` instance

**Example:**
```python
from vel import get_openai_agents_translator

translator = get_openai_agents_translator()
```

### `OpenAIAgentsSDKTranslator`

Translates OpenAI Agents SDK native events to Vel stream protocol events.

#### `translate(native_event)`

Translate a native OpenAI Agents SDK event to Vel format.

**Parameters:**
- `native_event`: Native event from `Runner.run_streamed().stream_events()`

**Returns:**
- `StreamEvent` in Vel format, or `None` if event should be skipped

**Example:**
```python
translator = get_openai_agents_translator()

result = Runner.run_streamed(agent, "Hello")

async for native_event in result.stream_events():
    vel_event = translator.translate(native_event)
    if vel_event:
        # Handle Vel-formatted event
        if vel_event.type == "text-delta":
            print(vel_event.delta, end="", flush=True)
```

#### `reset()`

Reset translator state between messages.

**Example:**
```python
# After processing one message, reset for next message
translator.reset()
```

## Event Translation Mapping

### OpenAI Agents SDK → Vel

| OpenAI Event | Vel Event | Description |
|--------------|-----------|-------------|
| `raw_response_event` (delta) | `text-delta` | Token-by-token streaming |
| `run_item_stream_event` (message completed) | `text-end` | Text block completes |
| `run_item_stream_event` (tool in_progress) | `tool-input-start` | Tool call begins |
| `run_item_stream_event` (tool completed) | `tool-output-available` | Tool result |
| `agent_updated_stream_event` | (skipped) | Agent state changes |

## Complete Example

```python
from vel import get_openai_agents_translator
from agents import Agent, Runner

async def chat_with_translation():
    # Create agent
    agent = Agent(
        name="Assistant",
        instructions="You are a helpful assistant"
    )

    # Get translator
    translator = get_openai_agents_translator()

    # Run agent (using actual OpenAI Agents SDK)
    result = Runner.run_streamed(agent, "Tell me a joke")

    # Collect response
    full_response = ""

    # Translate events
    async for native_event in result.stream_events():
        vel_event = translator.translate(native_event)

        if vel_event:
            if vel_event.type == "text-delta":
                # Token streaming
                print(vel_event.delta, end="", flush=True)
                full_response += vel_event.delta

            elif vel_event.type == "text-end":
                # Text complete
                print()

            elif vel_event.type == "tool-input-start":
                # Tool call starting
                print(f"\n[Tool: {vel_event.tool_name}]")

            elif vel_event.type == "tool-output-available":
                # Tool result
                print(f"[Result: {vel_event.output}]")

    return full_response
```

## Use Case: Orchestration Libraries

This is designed for libraries like Mesh that want to:
1. Support multiple agent SDKs
2. Provide consistent event handling
3. Not reimplement translation logic

```python
# In an orchestration library (e.g., Mesh)
from vel import get_openai_agents_translator
from agents import Agent, Runner

class AgentNode:
    def __init__(self, agent, use_native_events=False):
        self.agent = agent
        self.translator = None if use_native_events else get_openai_agents_translator()

    async def execute(self, input):
        # Use the actual agent/SDK
        result = Runner.run_streamed(self.agent, input)

        async for native_event in result.stream_events():
            if self.translator:
                # Translate to Vel format for consistency
                vel_event = self.translator.translate(native_event)
                if vel_event:
                    await self.emit_event(vel_event)
            else:
                # Use native events
                await self.emit_native_event(native_event)
```

## Benefits

1. **Single Source of Truth**
   - Translation logic lives in Vel
   - Libraries don't reimplement it

2. **Use Actual SDKs**
   - Respects user's agent configuration
   - Doesn't bypass their chosen SDK

3. **Consistent Events**
   - Same event structure across providers
   - Easier integration code

4. **Optional**
   - Can still use native events
   - Translation is opt-in

## What This Is NOT

❌ **Not a provider** - Doesn't make API calls
❌ **Not a replacement** - Doesn't replace your SDK
❌ **Not execution** - Just translates events

✅ **Pure translation** - Native events → Vel events
✅ **Stateless** - No side effects
✅ **Focused** - One job, done well

## Supported SDKs

Currently supported:
- ✅ OpenAI Agents SDK

Coming soon:
- Google Agents SDK (when available)
- Other agent frameworks

## Event Protocol

For complete Vel stream protocol documentation, see:
- [Vel Stream Protocol](https://rscheiwe.github.io/vel/stream-protocol.html)

## See Also

- [Vel Documentation](https://github.com/rscheiwe/vel)
- [OpenAI Agents SDK](https://openai.github.io/openai-agents-python/)
