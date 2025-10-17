---
layout: default
title: Providers
nav_order: 5
---

# Providers

Complete guide to LLM providers in Vel: OpenAI, Google Gemini, and Anthropic Claude.

## Overview

Vel uses a provider abstraction layer that allows you to switch between different LLM providers without changing your application code. All providers implement the same `BaseProvider` interface and emit standardized stream protocol events.

**Supported Providers:**
- OpenAI (gpt-4o, gpt-4-turbo, gpt-3.5-turbo, etc.)
- Google Gemini (gemini-1.5-pro, gemini-1.5-flash, etc.)
- Anthropic Claude (claude-opus-4, claude-sonnet-4, claude-3.5-sonnet, etc.)

## Provider Selection

Specify the provider when creating an agent:

```python
from vel import Agent

# OpenAI
agent = Agent(
    id='my-agent',
    model={'provider': 'openai', 'model': 'gpt-4o'}
)

# Google Gemini
agent = Agent(
    id='my-agent',
    model={'provider': 'google', 'model': 'gemini-1.5-pro'}
)

# Anthropic Claude
agent = Agent(
    id='my-agent',
    model={'provider': 'anthropic', 'model': 'claude-sonnet-4-20250514'}
)
```

## OpenAI Provider

### Configuration

**Required Environment Variable:**
```bash
OPENAI_API_KEY=sk-...
```

**Optional Environment Variable:**
```bash
OPENAI_API_BASE=https://api.openai.com/v1  # Custom endpoint
```

### Available Models

```python
# GPT-4 models
model={'provider': 'openai', 'model': 'gpt-4o'}
model={'provider': 'openai', 'model': 'gpt-4-turbo'}
model={'provider': 'openai', 'model': 'gpt-4'}

# GPT-3.5 models
model={'provider': 'openai', 'model': 'gpt-3.5-turbo'}
```

### Custom Endpoint

Use a custom OpenAI-compatible endpoint:

```bash
# .env file
OPENAI_API_BASE=https://my-custom-endpoint.com/v1
OPENAI_API_KEY=your-api-key
```

**Use cases:**
- Azure OpenAI Service
- OpenAI-compatible local models (LM Studio, Ollama with OpenAI adapter)
- Proxy services

### Features

**Streaming:**
- ✓ Text streaming with delta events
- ✓ Tool call streaming with incremental arguments
- ✓ Multiple tool calls per response
- ✓ Function calling support

**Non-streaming:**
- ✓ Single response generation
- ✓ Tool calling
- ✓ JSON mode support (via model config)

### Example

```python
import asyncio
from dotenv import load_dotenv
from vel import Agent

load_dotenv()

async def main():
    agent = Agent(
        id='openai-agent',
        model={'provider': 'openai', 'model': 'gpt-4o'}
    )

    # Non-streaming
    answer = await agent.run({'message': 'Hello!'})
    print(answer)

    # Streaming
    async for event in agent.run_stream({'message': 'Tell me a story'}):
        if event['type'] == 'text-delta':
            print(event['delta'], end='', flush=True)

asyncio.run(main())
```

## Google Gemini Provider

### Configuration

**Required Environment Variable:**
```bash
GOOGLE_API_KEY=...
```

**Required Package:**
```bash
pip install google-generativeai
```

### Available Models

```python
# Gemini 1.5 models
model={'provider': 'google', 'model': 'gemini-1.5-pro'}
model={'provider': 'google', 'model': 'gemini-1.5-flash'}

# Gemini 1.0 models
model={'provider': 'google', 'model': 'gemini-pro'}
```

### Features

**Streaming:**
- ✓ Text streaming with delta events
- ✓ Function calling support
- ✓ Multi-turn conversations

**Non-streaming:**
- ✓ Single response generation
- ✓ Function calling
- ✓ Multimodal support (text, images)

**Differences from OpenAI:**
- Uses `user` and `model` roles (not `assistant`)
- Function calls are not streamed incrementally (emit complete arguments)
- Different tool schema format (handled automatically)

### Example

```python
import asyncio
from dotenv import load_dotenv
from vel import Agent

load_dotenv()

async def main():
    agent = Agent(
        id='gemini-agent',
        model={'provider': 'google', 'model': 'gemini-1.5-pro'}
    )

    # Non-streaming
    answer = await agent.run({'message': 'Explain quantum computing'})
    print(answer)

    # Streaming
    async for event in agent.run_stream({'message': 'Write a poem'}):
        if event['type'] == 'text-delta':
            print(event['delta'], end='', flush=True)

asyncio.run(main())
```

---

## Anthropic Claude Provider

### Configuration

**Required Environment Variable:**
```bash
ANTHROPIC_API_KEY=sk-ant-...
```

**Optional Environment Variable:**
```bash
ANTHROPIC_API_BASE=https://api.anthropic.com  # Custom endpoint
```

### Available Models

```python
# Claude 4 models (latest)
model={'provider': 'anthropic', 'model': 'claude-opus-4-20250514'}
model={'provider': 'anthropic', 'model': 'claude-sonnet-4-20250514'}

# Claude 3.5 models
model={'provider': 'anthropic', 'model': 'claude-3-5-sonnet-20241022'}
model={'provider': 'anthropic', 'model': 'claude-3-5-haiku-20241022'}

# Claude 3 models
model={'provider': 'anthropic', 'model': 'claude-3-opus-20240229'}
model={'provider': 'anthropic', 'model': 'claude-3-sonnet-20240229'}
model={'provider': 'anthropic', 'model': 'claude-3-haiku-20240307'}
```

### Features

**Streaming:**
- ✓ Text streaming with delta events
- ✓ Tool call streaming with incremental arguments
- ✓ Extended thinking support (reasoning blocks)
- ✓ Multi-turn conversations

**Non-streaming:**
- ✓ Single response generation
- ✓ Tool calling
- ✓ Multimodal support (text, images, PDFs)
- ✓ Extended context window (200K tokens)

**Differences from OpenAI:**
- Supports system messages separately (via `system` parameter)
- Tool arguments streamed incrementally like OpenAI
- Supports extended thinking/reasoning blocks
- More explicit role structure (`user` and `assistant`)

### Example

```python
import asyncio
from dotenv import load_dotenv
from vel import Agent

load_dotenv()

async def main():
    agent = Agent(
        id='claude-agent',
        model={'provider': 'anthropic', 'model': 'claude-sonnet-4-20250514'}
    )

    # Non-streaming
    answer = await agent.run({'message': 'Explain quantum entanglement'})
    print(answer)

    # Streaming
    async for event in agent.run_stream({'message': 'Write a haiku about AI'}):
        if event['type'] == 'text-delta':
            print(event['delta'], end='', flush=True)

asyncio.run(main())
```

---

## Provider Comparison

| Feature | OpenAI | Gemini | Claude |
|---------|--------|--------|--------|
| Streaming text | ✓ | ✓ | ✓ |
| Streaming tool args | ✓ Incremental | ✓ Complete | ✓ Incremental |
| Multiple tool calls | ✓ | ✓ | ✓ |
| Custom endpoint | ✓ | ✗ | ✓ |
| Multimodal input | ✓ (Vision models) | ✓ (Native) | ✓ (Native) |
| Extended thinking | ✗ | ✗ | ✓ |
| JSON mode | ✓ | ✗ | ✗ |
| Max context | 128K | 2M | 200K |
| Cost | $$$ | $$ | $$$ |

---

## Generation Configuration

Control model behavior with fine-grained generation parameters. Vel provides full parity with Vercel AI SDK's `streamText()` flexibility.

### Agent-Level Configuration

Set default generation parameters when creating an agent:

```python
from vel import Agent

agent = Agent(
    id='my-agent',
    model={'provider': 'openai', 'model': 'gpt-4o'},
    generation_config={
        'temperature': 0.7,      # Creativity (0-2)
        'max_tokens': 500,       # Output limit
        'top_p': 0.9,            # Nucleus sampling
        'presence_penalty': 0.6, # Encourage new topics (OpenAI)
        'frequency_penalty': 0.3,# Reduce repetition (OpenAI)
        'stop': ['END'],         # Stop sequences
        'seed': 42               # Reproducible outputs (OpenAI, Anthropic)
    }
)
```

### Per-Run Override

Override generation config for specific runs:

```python
# Use agent's default config
result1 = await agent.run({'message': 'Write a creative story'})

# Override for deterministic response
result2 = await agent.run(
    {'message': 'What is 2+2?'},
    generation_config={'temperature': 0}  # Override for this run only
)

# Works with streaming too
async for event in agent.run_stream(
    {'message': 'Explain AI'},
    generation_config={'max_tokens': 100}  # Brief response
):
    print(event)
```

### Supported Parameters by Provider

#### Common Parameters (All Providers)

| Parameter | Type | Description | Example |
|-----------|------|-------------|---------|
| `temperature` | float | Sampling temperature (0-2) | `0.7` |
| `max_tokens` | int | Maximum output tokens | `500` |
| `top_p` | float | Nucleus sampling (0-1) | `0.9` |
| `stop` | List[str] | Stop sequences | `['END']` |

#### OpenAI-Specific

| Parameter | Type | Description | Example |
|-----------|------|-------------|---------|
| `presence_penalty` | float | Penalize new tokens (-2 to 2) | `0.6` |
| `frequency_penalty` | float | Penalize repeated tokens (-2 to 2) | `0.3` |
| `seed` | int | Reproducibility seed | `42` |
| `logit_bias` | Dict[int, float] | Token probability adjustments | `{50256: -100}` |
| `user` | str | User identifier for tracking | `'user-123'` |

#### Anthropic-Specific

| Parameter | Type | Description | Example |
|-----------|------|-------------|---------|
| `top_k` | int | Top-K sampling | `40` |
| `stop_sequences` | List[str] | Alternative to `stop` | `['END']` |

#### Google Gemini-Specific

| Parameter | Type | Description | Example |
|-----------|------|-------------|---------|
| `top_k` | int | Top-K sampling | `40` |
| `max_output_tokens` | int | Alternative to `max_tokens` | `2048` |
| `stop_sequences` | List[str] | Alternative to `stop` | `['END']` |

### Examples

#### Deterministic Code Generation

```python
agent = Agent(
    id='code-gen',
    model={'provider': 'openai', 'model': 'gpt-4o'},
    generation_config={
        'temperature': 0,
        'seed': 42,  # Same output every time
        'max_tokens': 2000
    }
)
```

#### Creative Writing

```python
agent = Agent(
    id='creative',
    model={'provider': 'anthropic', 'model': 'claude-sonnet-4-20250514'},
    generation_config={
        'temperature': 0.9,  # High creativity
        'top_p': 0.95,
        'top_k': 50,
        'max_tokens': 4000
    }
)
```

#### Concise Responses

```python
agent = Agent(
    id='brief',
    model={'provider': 'google', 'model': 'gemini-1.5-pro'},
    generation_config={
        'max_tokens': 100,
        'temperature': 0.7,
        'stop_sequences': ['\n\n']  # Stop at double newline
    }
)
```

See `examples/generation_config_example.py` for comprehensive examples.

## Environment Variables

### OpenAI

```bash
# Required
OPENAI_API_KEY=sk-...

# Optional
OPENAI_API_BASE=https://api.openai.com/v1
```

### Google Gemini

```bash
# Required
GOOGLE_API_KEY=...
```

### Anthropic Claude

```bash
# Required
ANTHROPIC_API_KEY=sk-ant-...

# Optional
ANTHROPIC_API_BASE=https://api.anthropic.com
```

### Example .env File

```bash
# Copy from .env.example
cp .env.example .env

# Edit .env
OPENAI_API_KEY=sk-proj-...
GOOGLE_API_KEY=AIza...
ANTHROPIC_API_KEY=sk-ant-...

# Optional
OPENAI_API_BASE=https://api.openai.com/v1
ANTHROPIC_API_BASE=https://api.anthropic.com
POSTGRES_DSN=postgresql+psycopg://user:pass@localhost:5432/vel
```

## Error Handling

### Missing API Key

**Error:**
```
ValueError: OPENAI_API_KEY environment variable is not set
```

**Solution:**
1. Add key to `.env` file
2. Ensure `load_dotenv()` is called before creating agent
3. Or export environment variable: `export OPENAI_API_KEY=sk-...`

### Import Error (Gemini)

**Error:**
```
ImportError: google-generativeai not installed
```

**Solution:**
```bash
pip install google-generativeai
```

### Rate Limit Errors

**Error:**
```
httpx.HTTPStatusError: 429 Too Many Requests
```

**Solution:**
- Implement retry logic (future feature)
- Add delays between requests
- Upgrade API tier with provider

### Invalid Model Name

**Error:**
```
httpx.HTTPStatusError: 404 Not Found
```

**Solution:**
- Check model name spelling
- Verify model access with your API key
- See available models section above

## Creating Custom Providers

### Implement BaseProvider

```python
from vel.providers.base import BaseProvider, LLMMessage
from vel.events import StreamEvent, TextDeltaEvent, FinishMessageEvent
from typing import Any, AsyncGenerator, Dict, List

class CustomProvider(BaseProvider):
    """Custom LLM provider"""
    name = 'custom'

    def __init__(self):
        # Validate API key, set endpoint, etc.
        pass

    async def stream(
        self,
        messages: List[LLMMessage],
        model: str,
        tools: Dict[str, Any]
    ) -> AsyncGenerator[StreamEvent, None]:
        """Stream response as stream protocol events"""
        # Connect to your LLM API
        # Translate native events to StreamEvent objects
        # Yield TextStartEvent, TextDeltaEvent, TextEndEvent, etc.
        yield TextDeltaEvent(block_id='1', delta='Hello')
        yield FinishMessageEvent(finish_reason='stop')

    async def generate(
        self,
        messages: List[LLMMessage],
        model: str,
        tools: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Non-streaming generation"""
        # Call your LLM API
        # Return {'done': True, 'answer': '...'}
        return {'done': True, 'answer': 'Hello from custom provider'}
```

### Register Provider

```python
from vel.providers import register_provider

register_provider(CustomProvider())

# Use it
agent = Agent(
    id='my-agent',
    model={'provider': 'custom', 'model': 'my-model'}
)
```

## Best Practices

### API Key Security

```python
# ✓ Good: Use environment variables
load_dotenv()
agent = Agent(
    id='my-agent',
    model={'provider': 'openai', 'model': 'gpt-4o'}
)

# ✗ Bad: Hardcode keys
api_key = 'sk-...'  # Never do this!
```

### Model Selection

```python
# Production: Use reliable, tested models
model={'provider': 'openai', 'model': 'gpt-4o'}

# Development: Use faster, cheaper models
model={'provider': 'openai', 'model': 'gpt-3.5-turbo'}
model={'provider': 'google', 'model': 'gemini-1.5-flash'}
```

### Error Handling

```python
try:
    agent = Agent(
        id='my-agent',
        model={'provider': 'openai', 'model': 'gpt-4o'}
    )
    answer = await agent.run({'message': 'Hello'})
except ValueError as e:
    # Missing API key
    print(f"Configuration error: {e}")
except httpx.HTTPStatusError as e:
    # API error (rate limit, invalid model, etc.)
    print(f"API error: {e}")
```

### Provider Fallback

```python
async def get_agent(preferred_provider='openai'):
    """Create agent with fallback"""
    try:
        if preferred_provider == 'openai':
            return Agent(
                id='my-agent',
                model={'provider': 'openai', 'model': 'gpt-4o'}
            )
    except ValueError:
        pass

    # Fallback to Gemini
    return Agent(
        id='my-agent',
        model={'provider': 'google', 'model': 'gemini-1.5-pro'}
    )
```

## Troubleshooting

### "Illegal header value b'Bearer '"

API key is empty or not loaded.

**Check:**
1. `.env` file exists and has `OPENAI_API_KEY=sk-...`
2. `load_dotenv()` called before creating agent
3. No whitespace or quotes around key in `.env`

### Streaming Not Working

**Check:**
1. Using `run_stream()` not `run()`
2. Iterating over async generator: `async for event in agent.run_stream(...)`
3. Network supports streaming (some proxies buffer)

### Different Behavior Between Providers

Providers have different capabilities:
- OpenAI streams tool arguments incrementally
- Gemini emits complete tool arguments at once
- Message role names differ (`assistant` vs `model`)

These differences are normalized by the stream protocol, but may affect performance characteristics.

## Using Translators Directly

Vel's providers use **translators** internally to convert provider-specific events to standardized stream protocol events. You can also use translators directly for custom orchestration.

### When to Use Translators

- Building custom orchestrator with specific control flow
- Integrating with external frameworks (Mesh, LangGraph)
- Single-shot LLM calls without multi-step logic
- Protocol testing and validation

### Quick Example

```python
from vel.providers.translators import OpenAIAPITranslator
from openai import AsyncOpenAI

translator = OpenAIAPITranslator()
client = AsyncOpenAI()

stream = await client.chat.completions.create(
    model='gpt-4o',
    messages=[{'role': 'user', 'content': 'Hello'}],
    stream=True
)

async for chunk in stream:
    event = translator.translate_chunk(chunk.model_dump())
    if event:
        print(event.type, event.delta if hasattr(event, 'delta') else '')
```

### Important: Filling the Gaps

Translators only emit **content events** (text-delta, tool-input-available, etc.). They don't emit:
- `start` / `start-step` / `finish-step` / `finish` (orchestration events)
- `tool-output-available` (requires tool execution)

**If using with AI SDK frontend components**, you must manually emit these events. See the complete guides:

**👉 [Event Translators](event-translators)** - Architecture overview and internal composition
**👉 [Using Translators Directly](using-translators)** - Full guide with working examples

### Available Translators

```python
from vel.providers.translators import (
    OpenAIAPITranslator,           # OpenAI Chat Completions API
    OpenAIResponsesAPITranslator,  # OpenAI Responses API (o1/o3)
    OpenAIAgentsSDKTranslator,     # OpenAI Agents SDK
    AnthropicAPITranslator,        # Anthropic Messages API
    GeminiAPITranslator,           # Google Gemini API
)
```

## Next Steps

- [Event Translators](event-translators) - Protocol adapter architecture and internal composition
- [Tools](tools.md) - Add function calling to your agents
- [Stream Protocol](stream-protocol.md) - Understand streaming events
- [Session Management](sessions.md) - Multi-turn conversations
