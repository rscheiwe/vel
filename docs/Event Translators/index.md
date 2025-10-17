---
layout: default
title: Event Translators
nav_order: 8
has_children: true
---

# Event Translators

Vel's event translator system provides a clean separation between protocol conversion and orchestration, enabling composability and framework interoperability.

## What Are Translators?

Translators convert provider-specific streaming events into Vel's standardized stream protocol events. They implement the **Protocol Adapter Pattern**, focusing solely on format conversion without orchestration logic.

## Architecture Overview

### Single Responsibility Principle

Vel separates concerns across two layers:

#### Layer 1: Translators (Protocol Adapters)

- **Job:** Convert provider-specific → standard protocol
- **Scope:** Single LLM response stream
- **Stateful:** Only tracks current response (text blocks, tool calls)
- **Reusable:** Works with any orchestrator (Vel Agent, Mesh, LangGraph, custom)

#### Layer 2: Agent (Orchestrator)

- **Job:** Multi-step execution, tool calling, context management
- **Scope:** Full agentic workflow
- **Stateful:** Sessions, context, run history
- **Opinionated:** Implements specific orchestration pattern

**Why This Matters:** This separation enables **composability**. You can use Vel's translators in any orchestration framework without adopting Vel's Agent pattern.

## How Translators Work

### Internal Composition (Most Common)

When you use an Agent with a provider, the translator is used **internally** via composition:

```python
# 1. User creates Agent with OpenAI
agent = Agent(
    id='my-agent',
    model={'provider': 'openai', 'model': 'gpt-4o'}
)

# 2. Agent gets OpenAIProvider from registry
provider = self.providers.get('openai')

# 3. OpenAIProvider has translator internally
class OpenAIProvider(BaseProvider):
    def __init__(self):
        self.translator = OpenAIAPITranslator()  # ← Composition

# 4. Provider uses translator to convert chunks
async def stream(self, messages, model, tools):
    async for chunk in openai_api.stream():
        vel_event = self.translator.translate_chunk(chunk)  # ← Translation
        yield vel_event  # ← Yields to Agent

# 5. Agent adds orchestration events
yield {'type': 'start-step'}         # ← Agent adds
async for event in provider.stream():  # ← Provider/translator events
    yield event.to_dict()
yield {'type': 'finish-step', ...}   # ← Agent adds
```

### Data Flow Diagram

```
┌─────────────────────────────────────────────────────┐
│                      Agent                          │
│  - Emits: start, start-step, finish-step, finish   │
│  - Executes tools                                   │
│  - Manages sessions/context                         │
└──────────────────┬────────────────────────────────┘
                   │ calls provider.stream()
                   ↓
┌─────────────────────────────────────────────────────┐
│                 OpenAIProvider                       │
│  - Makes HTTP calls to OpenAI API                   │
│  - Uses translator internally (composition)         │
└──────────────────┬────────────────────────────────┘
                   │ uses translator.translate_chunk()
                   ↓
┌─────────────────────────────────────────────────────┐
│             OpenAIAPITranslator                      │
│  - Converts OpenAI chunks → Vel events              │
│  - Emits: text-delta, tool-input-available, etc.    │
└─────────────────────────────────────────────────────┘
                   │
                   ↓
              Vel Events
        (text-delta, tool-input-available, etc.)
```

### Provider-Translator Mapping

Each provider uses its corresponding translator via composition:

```python
# OpenAI Provider
class OpenAIProvider(BaseProvider):
    name = 'openai'
    def __init__(self):
        self.translator = OpenAIAPITranslator()

# Anthropic Provider
class AnthropicProvider(BaseProvider):
    name = 'anthropic'
    def __init__(self):
        self.translator = AnthropicAPITranslator()

# Google Gemini Provider
class GeminiProvider(BaseProvider):
    name = 'google'
    def __init__(self):
        self.translator = GeminiAPITranslator()

# OpenAI Responses API (o1/o3 models)
class OpenAIResponsesProvider(BaseProvider):
    name = 'openai-responses'
    def __init__(self):
        self.translator = OpenAIResponsesAPITranslator()
```

## What Translators Emit

Translators handle **content-level events** from a single LLM response:

### Always Emitted ✅

- `text-start`, `text-delta`, `text-end` - Text streaming
- `tool-input-start`, `tool-input-delta`, `tool-input-available` - Tool calls
- `response-metadata` - Usage, model ID (consumed by Agent)
- `finish-message` - Completion signal (consumed by Agent)
- `error` - Error events

### Provider-Specific ✅

- `reasoning-start`, `reasoning-delta`, `reasoning-end` - OpenAI o1/o3, Anthropic thinking
- `source` - Gemini grounding citations
- `file` - Inline file attachments

### Never Emitted ❌

- `start` / `start-step` / `finish-step` / `finish` - Orchestration (Agent's job)
- `tool-output-available` - Requires tool execution (Agent's job)

## Using Translators Directly

You can also use translators **standalone** for custom orchestration:

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

# You must add orchestration events manually
yield {'type': 'start'}
yield {'type': 'start-step'}

async for chunk in stream:
    event = translator.translate_chunk(chunk.model_dump())
    if event and event.type not in ('response-metadata', 'finish-message'):
        yield event.to_dict()

yield {'type': 'finish-step', 'finishReason': 'stop', ...}
yield {'type': 'finish', 'finishReason': 'stop', ...}
```

**See:** [Using Translators Directly](using-translators) for complete examples with multi-step and tool calling.

## Available Translators

Import from `vel.providers.translators`:

```python
from vel.providers.translators import (
    OpenAIAPITranslator,           # OpenAI Chat Completions API
    OpenAIResponsesAPITranslator,  # OpenAI Responses API (o1/o3)
    OpenAIAgentsSDKTranslator,     # OpenAI Agents SDK (experimental)
    AnthropicAPITranslator,        # Anthropic Messages API
    GeminiAPITranslator,           # Google Gemini API
)
```

## Design Benefits

### 1. Composability

Translators can be used:
- **Inside Vel providers** (default, automatic)
- **With external frameworks** (Mesh, LangGraph)
- **In custom orchestrators** (full control)

### 2. Testability

Each layer can be tested independently:
- Test translator without HTTP
- Test provider without Agent
- Test Agent with mock provider

### 3. Maintainability

Changes are localized:
- OpenAI API changes → Update translator only
- Orchestration logic → Update Agent only
- Stream protocol → Update both (rare)

### 4. Reusability

Same translator works in multiple contexts:
```python
# Used by Vel Agent
agent = Agent(model={'provider': 'openai', ...})

# Used by custom orchestrator
translator = OpenAIAPITranslator()
# Your custom logic
```

## When to Use Each Approach

### Use Agent (Recommended)

**When you need:**
- Multi-step execution
- Tool calling with execution
- Session management
- Turn-key agentic workflows

```python
agent = Agent(id='my-agent', model={...}, tools=[...])
async for event in agent.run_stream({'message': '...'}):
    yield event  # Complete stream with orchestration
```

### Use Translator Directly

**When you need:**
- Custom orchestration logic
- Integration with external frameworks
- Single-shot LLM calls
- Protocol testing

```python
translator = OpenAIAPITranslator()
# Your orchestration logic
async for chunk in provider_stream:
    event = translator.translate_chunk(chunk)
    # Your handling
```

## Next Steps

- **[Using Translators Directly](using-translators)** - Complete guide with working examples
- **[Providers](providers)** - Available providers and configuration
- **[Stream Protocol](stream-protocol)** - Event reference
