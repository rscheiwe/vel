---
layout: default
title: Getting Started
nav_order: 2
---

# Getting Started with Vel

Complete guide to installing and using Vel for the first time.

## Installation

### Prerequisites

- Python 3.10 or higher
- pip

### Install from Source

```bash
# Clone the repository
git clone <repo-url>
cd vel

# Install in development mode
pip install -e .

# Optional: Install dev dependencies
pip install -e ".[dev]"
```

## Configuration

### Environment Variables

Create a `.env` file in the project root:

```bash
# Copy the example
cp .env.example .env

# Edit with your API keys
vim .env
```

#### Required Variables

```bash
# For OpenAI
OPENAI_API_KEY=sk-...

# OR for Google Gemini
GOOGLE_API_KEY=...

# OR for Anthropic Claude
ANTHROPIC_API_KEY=sk-ant-...
```

#### Optional Variables

```bash
# OpenAI Custom Endpoint
OPENAI_API_BASE=https://api.openai.com/v1
```

### API Key Configuration Methods

Vel supports **two ways** to provide API keys, making it suitable for both applications and libraries:

#### Method 1: Environment Variables (recommended for applications)

Set environment variables as shown above. Agents will automatically use them:

```python
agent = Agent(
    id='my-agent',
    model={'provider': 'openai', 'model': 'gpt-4o'}
)
# Uses OPENAI_API_KEY from environment
```

**Pros:**
- ✓ Secure (not in code)
- ✓ Easy for development
- ✓ Standard practice

**Cons:**
- ✗ Not suitable for libraries imported by others
- ✗ Can't use different keys for different agents

#### Method 2: Explicit API Keys (recommended for libraries/production)

Pass API keys directly in the model config:

```python
agent = Agent(
    id='my-agent',
    model={
        'provider': 'openai',
        'model': 'gpt-4o',
        'api_key': 'sk-...'  # Override environment variable
    }
)
```

**Pros:**
- ✓ Works in any environment
- ✓ Suitable for installable packages
- ✓ Different agents can use different keys
- ✓ Perfect for multi-tenant applications

**Cons:**
- ✗ Must manage secrets carefully

#### Use Cases

| Scenario | Method |
|----------|--------|
| Building an application | Environment variables |
| Building a library/package | Explicit API keys |
| Multi-tenant SaaS | Explicit API keys (per-tenant) |
| Development/testing | Environment variables |
| CI/CD | Environment variables |

## Quick Start

### Basic Usage

```python
import asyncio
from dotenv import load_dotenv
from vel import Agent

load_dotenv()

async def main():
    # Create an agent
    agent = Agent(
        id='my-agent',
        model={'provider': 'openai', 'model': 'gpt-4o'}
    )

    # Non-streaming mode
    answer = await agent.run({'message': 'Hello, how are you?'})
    print(answer)

if __name__ == '__main__':
    asyncio.run(main())
```

### Streaming Mode

```python
async def streaming_example():
    agent = Agent(
        id='my-agent',
        model={'provider': 'openai', 'model': 'gpt-4o'}
    )

    # Stream events as they arrive
    async for event in agent.run_stream({'message': 'Tell me a joke'}):
        print(event)
```

### With Sessions (Multi-turn)

```python
async def session_example():
    agent = Agent(
        id='my-agent',
        model={'provider': 'openai', 'model': 'gpt-4o'}
    )

    session_id = 'user-123'

    # First turn
    answer1 = await agent.run(
        {'message': 'My name is Alice'},
        session_id=session_id
    )
    print(answer1)

    # Second turn - remembers Alice
    answer2 = await agent.run(
        {'message': 'What is my name?'},
        session_id=session_id
    )
    print(answer2)  # "Your name is Alice"
```

### With Tools

```python
from vel import Agent, ToolSpec

# Define a custom tool (no registration needed!)
def get_weather(city: str) -> dict:
    """Get weather for a city."""
    return {'temp_f': 72, 'condition': 'sunny', 'city': city}

# Create tool from function
weather_tool = ToolSpec.from_function(get_weather)

# Use the agent with tools
async def tool_example():
    agent = Agent(
        id='my-agent',
        model={'provider': 'openai', 'model': 'gpt-4o'},
        tools=[weather_tool]  # Pass directly, no registration!
    )

    answer = await agent.run({'message': 'What is the weather in New York?'})
    print(answer)  # Agent will call the tool and respond
```

### With Prompt Templates

```python
from vel import Agent, PromptTemplate

# Define a prompt template (no registration needed!)
template = PromptTemplate(
    id="assistant:v1",
    system="""
    <system_instructions>
      <role>You are {{role_name}}, an expert in {{domain}}.</role>
      <guidelines>
        - Be concise and accurate
        - Cite sources when possible
      </guidelines>
    </system_instructions>
    """
)

async def prompt_example():
    agent = Agent(
        id='assistant',
        model={'provider': 'openai', 'model': 'gpt-4o'},
        prompt=template,  # Pass directly, no registration!
        prompt_vars={
            'role_name': 'Dr. Smith',
            'domain': 'medical information'
        }
    )

    answer = await agent.run({'message': 'What causes headaches?'})
    print(answer)
```

This enables powerful use cases:
- **Prompts from database/API** - Load templates at runtime
- **User-created prompts** - Let users customize agent behavior
- **A/B testing** - Test different prompt versions
- **Per-tenant customization** - Different prompts for different customers

See [Prompt Templates](prompts.md) for more details.

### Message Aggregation

Use **MessageReducer** to aggregate streaming events into structured messages compatible with the Vercel AI SDK format:

```python
from vel import Agent, MessageReducer

async def message_aggregation_example():
    """Aggregate streaming events into structured messages"""
    # Create reducer
    reducer = MessageReducer()

    # Add user message
    user_msg = reducer.add_user_message(
        "What's the weather in San Francisco?",
        metadata={"user_id": "user-123", "timestamp": "2024-01-15T10:00:00Z"}
    )

    # Stream agent response
    agent = Agent(
        id='weather-agent',
        model={'provider': 'openai', 'model': 'gpt-4o'},
        tools=['get_weather']
    )

    async for event in agent.run_stream({'message': "What's the weather in SF?"}):
        reducer.process_event(event)

    # Get messages in Vercel AI SDK format
    messages = reducer.get_messages(
        assistant_metadata={"model": "gpt-4o"}
    )
    # [
    #   {user message},
    #   {assistant message with parts: [tool-call, tool-result, text]}
    # ]

    # Use messages however you need (store in DB, return to client, etc.)
    return messages
```

**Key Features:**
- ✓ Compatible with Vercel AI SDK `useChat` hook
- ✓ Aggregates text, tool calls, and tool results into parts array
- ✓ Includes provider metadata (OpenAI message/call IDs)
- ✓ Supports custom message IDs and metadata

See [Stream Protocol - Message Aggregation](stream-protocol.md#message-aggregation) for complete details.

### With Generation Configuration

Control model behavior with fine-grained parameters:

```python
from vel import Agent

async def generation_config_example():
    # Agent with default config
    agent = Agent(
        id='my-agent',
        model={'provider': 'openai', 'model': 'gpt-4o'},
        generation_config={
            'temperature': 0.7,  # Creativity
            'max_tokens': 500    # Output limit
        }
    )

    # Use default config
    creative = await agent.run({'message': 'Write a poem'})
    print(creative)

    # Override for specific run
    factual = await agent.run(
        {'message': 'What is 2+2?'},
        generation_config={'temperature': 0}  # Deterministic for this run
    )
    print(factual)
```

**Common Parameters:**
- `temperature` - Creativity (0-2)
- `max_tokens` - Output length limit
- `top_p` - Nucleus sampling (0-1)
- `seed` - Reproducible outputs (OpenAI, Anthropic)

See [Providers](providers.md#generation-configuration) for all parameters.

## Architecture Overview

Vel uses a two-layer architecture based on the **Single Responsibility Principle**:

### Layer 1: Translators (Protocol Adapters)

**Responsibility:** Convert provider-specific events → standard stream protocol

```python
from vel.providers.translators import OpenAIAPITranslator

translator = OpenAIAPITranslator()
# Converts OpenAI chunks → Vel protocol events
```

- **Job:** Protocol translation only
- **Scope:** Single LLM response stream
- **Stateful:** Only tracks current response (text blocks, tool calls)
- **Reusable:** Works with any orchestrator (Vel Agent, Mesh, LangGraph)

### Layer 2: Agent (Orchestrator)

**Responsibility:** Multi-step execution, tool calling, context management

```python
from vel import Agent

agent = Agent(id='my-agent', model={...}, tools=[...])
# Handles orchestration, tool execution, sessions
```

- **Job:** Full agentic workflow
- **Scope:** Multi-step execution with tools
- **Stateful:** Sessions, context, run history
- **Opinionated:** Implements specific orchestration pattern

### Why Two Layers?

This separation enables **composability**:

1. **Use Agent** for turnkey agentic workflows (most common)
2. **Use Translator** when integrating with external frameworks or building custom orchestrators

The translator layer can be reused across different orchestration strategies without modification.

**Learn more:** [Event Translators](event-translators) - Complete architecture details and usage guide

## Next Steps

- [Session Management](sessions.md) - Learn about multi-turn conversations
- [Prompt Templates](prompts.md) - Dynamic system prompts with Jinja2 templating
- [Providers](providers.md) - Configure OpenAI, Gemini, and Claude
- [Tools](tools.md) - Create custom tools
- [Stream Protocol](stream-protocol.md) - Understand streaming events and custom data-* events for RAG, progress tracking, and analytics
- [Event Translators](event-translators) - Protocol adapter architecture and custom orchestration
- [API Reference](api-reference.md) - Complete API documentation

## Troubleshooting

### "Illegal header value b'Bearer '"

Your `OPENAI_API_KEY`, `GOOGLE_API_KEY`, or `ANTHROPIC_API_KEY` is not set. Check your `.env` file.

### Import Errors

Make sure you installed the package:

```bash
pip install -e .
```

And that you're loading environment variables:

```python
from dotenv import load_dotenv
load_dotenv()
```
