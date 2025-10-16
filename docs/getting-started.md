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
# Database (for persistent sessions)
POSTGRES_DSN=postgresql+psycopg://user:pass@localhost:5432/vel

# Redis (for caching)
REDIS_URL=redis://localhost:6379/0

# OpenAI Custom Endpoint
OPENAI_API_BASE=https://api.openai.com/v1
```

**Note:** If `POSTGRES_DSN` and `REDIS_URL` are not set, Vel will use in-memory storage (fine for development).

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
        model={'provider': 'openai', 'model': 'gpt-4o'},
        session_storage='memory'  # or 'database'
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
from vel import Agent, ToolSpec, register_tool

# Define a custom tool
def get_weather_handler(input: dict, ctx: dict) -> dict:
    city = input['city']
    return {'temp_f': 72, 'condition': 'sunny', 'city': city}

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

register_tool(weather_tool)

# Use the agent with tools
async def tool_example():
    agent = Agent(
        id='my-agent',
        model={'provider': 'openai', 'model': 'gpt-4o'},
        tools=['get_weather']
    )

    answer = await agent.run({'message': 'What is the weather in New York?'})
    print(answer)  # Agent will call the tool and respond
```

### Storing Messages for Database

Use **MessageReducer** to aggregate streaming events into structured messages compatible with the Vercel AI SDK format:

```python
from vel import Agent, MessageReducer

async def database_storage_example():
    """Aggregate streaming events for database storage"""
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

    # Store in database
    for msg in messages:
        await db.insert_message(msg)
```

**Key Features:**
- ✓ Compatible with Vercel AI SDK `useChat` hook
- ✓ Aggregates text, tool calls, and tool results into parts array
- ✓ Includes provider metadata (OpenAI message/call IDs)
- ✓ Supports custom message IDs and metadata
- ✓ Ready for database storage

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

## REST API

### Start the Service

```bash
# Start with uvicorn
uvicorn agents_service.main:app --reload

# Or with custom host/port
uvicorn agents_service.main:app --host 0.0.0.0 --port 8000
```

### Streaming Endpoint

```bash
curl -X POST http://localhost:8000/runs \
  -H "Content-Type: application/json" \
  -d '{
    "agent_id": "chat-general:v1",
    "provider": "openai",
    "model": "gpt-4o",
    "input": {"message": "hello"}
  }'
```

### Non-Streaming Endpoint

```bash
curl -X POST http://localhost:8000/runs/sync \
  -H "Content-Type: application/json" \
  -d '{
    "agent_id": "chat-general:v1",
    "provider": "google",
    "model": "gemini-1.5-pro",
    "input": {"message": "hello"}
  }'
```

## Next Steps

- [Session Management](sessions.md) - Learn about multi-turn conversations
- [Providers](providers.md) - Configure OpenAI, Gemini, and Claude
- [Tools](tools.md) - Create custom tools
- [Stream Protocol](stream-protocol.md) - Understand streaming events and custom data-* events for RAG, progress tracking, and analytics
- [API Reference](api-reference.md) - Complete API documentation

## Troubleshooting

### "Illegal header value b'Bearer '"

Your `OPENAI_API_KEY`, `GOOGLE_API_KEY`, or `ANTHROPIC_API_KEY` is not set. Check your `.env` file.

### "Connection refused" (Postgres/Redis)

If you see connection errors for Postgres or Redis, comment out `POSTGRES_DSN` and `REDIS_URL` in your `.env` file to use in-memory storage.

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
