# Vel — Agent Runtime (12-Factor Agents aligned)

A 12-Factor inspired AI agent runtime with streaming responses, multiple LLM providers, and event-driven architecture.

## Features

- **Dual Execution Modes**: Streaming (SSE) and non-streaming (JSON) responses
- **Multiple LLM Providers**: OpenAI and Google Gemini with plug-and-play architecture
- **Stream Protocol**: Vercel AI SDK-compatible event system for provider-agnostic streaming
- **Tool System**: JSON schema-validated tools with async support
- **Persistent Storage**: PostgreSQL for durability, Redis for caching
- **FastAPI Service**: Production-ready REST API with health checks

## Documentation

**📚 [Complete Documentation](docs/README.md)**

- [Getting Started](docs/getting-started.md) - Installation and quick start
- [Session Management](docs/sessions.md) - Multi-turn conversations
- [Providers](docs/providers.md) - OpenAI and Gemini configuration
- [Tools](docs/tools.md) - Custom tool creation
- [Stream Protocol](docs/stream-protocol.md) - Event streaming reference
- [API Reference](docs/api-reference.md) - Complete API docs

## Project Structure

```
agents/
├── providers/      # LLM provider implementations (OpenAI, Gemini)
├── storage/        # Storage layer (Postgres, Redis)
├── tools/          # Tool registry and specifications
├── core/           # State management, reducer, context
├── events.py       # Stream protocol event definitions
└── agent.py        # Main Agent class

agents_service/
└── main.py         # FastAPI service with streaming & sync endpoints
```

## Installation

```bash
# Clone and install
git clone <repo-url>
cd vel
pip install -e .

# Set up environment
cp .env.example .env
# Edit .env with your API keys
```

## Quick Start

### Python SDK

```python
import asyncio
from agents import Agent

async def main():
    # Non-streaming mode
    agent = Agent(
        id='chat-general:v1',
        model={'provider': 'openai', 'model': 'gpt-4o'},
        tools=['get_weather'],
        policies={'max_steps': 8}
    )
    answer = await agent.run({'message': 'What is the weather?'})
    print(answer)

    # Streaming mode
    async for event in agent.run_stream({'message': 'Tell me a story'}):
        print(event)

if __name__ == '__main__':
    asyncio.run(main())
```

### REST API

```bash
# Start the service
uvicorn agents_service.main:app --reload

# Streaming endpoint (SSE)
curl -X POST http://localhost:8000/runs \
  -H "Content-Type: application/json" \
  -d '{
    "agent_id": "chat-general:v1",
    "provider": "openai",
    "model": "gpt-4o",
    "input": {"message": "hello"}
  }'

# Non-streaming endpoint (JSON)
curl -X POST http://localhost:8000/runs/sync \
  -H "Content-Type: application/json" \
  -d '{
    "agent_id": "chat-general:v1",
    "provider": "google",
    "model": "gemini-1.5-pro",
    "input": {"message": "hello"}
  }'
```

## Stream Protocol

Vel uses the [Vercel AI SDK stream protocol](https://ai-sdk.dev/docs/ai-sdk-ui/stream-protocol) for provider-agnostic event streaming:

- `text-start`, `text-delta`, `text-end` - Text content chunks
- `tool-input-start`, `tool-input-delta`, `tool-input-available` - Tool call inputs
- `tool-output-available` - Tool execution results
- `error`, `finish-message` - Error handling and completion

Each provider translates native events into these standardized events.

## Providers

### OpenAI
```python
agent = Agent(
    id='my-agent',
    model={'provider': 'openai', 'model': 'gpt-4o'}
)
```

### Google Gemini
```python
agent = Agent(
    id='my-agent',
    model={'provider': 'google', 'model': 'gemini-1.5-pro'}
)
```

## Session Management (Multi-Turn Conversations)

Sessions enable multi-turn conversations where the agent remembers context across multiple calls.

### Basic Session Usage
```python
agent = Agent(
    id='my-agent',
    model={'provider': 'openai', 'model': 'gpt-4o'},
    session_storage='memory'  # or 'database'
)

# Multi-turn conversation - same session_id = shared memory
session_id = 'user-123'

answer1 = await agent.run({'message': 'My name is Alice'}, session_id=session_id)
# "Hello Alice! How can I help you?"

answer2 = await agent.run({'message': 'What is my name?'}, session_id=session_id)
# "Your name is Alice."
```

### Session Storage Modes

#### In-Memory (Default - Fast, Not Persistent)
```python
agent = Agent(
    id='my-agent',
    model={'provider': 'openai', 'model': 'gpt-4o'},
    session_storage='memory'  # Sessions lost on restart
)
```

#### Database (Persistent, Survives Restarts)
```python
# Requires POSTGRES_DSN configured
agent = Agent(
    id='my-agent',
    model={'provider': 'openai', 'model': 'gpt-4o'},
    session_storage='database'  # Sessions stored in Postgres
)
```

### Context Manager Modes

Control how much history is retained:

```python
from agents import ContextManager, StatelessContextManager

# Full memory (default)
agent = Agent(..., context_manager=ContextManager())

# No memory (stateless)
agent = Agent(..., context_manager=StatelessContextManager())

# Limited history (last 10 messages)
agent = Agent(..., context_manager=ContextManager(max_history=10))

# Custom logic
class CustomContextManager(ContextManager):
    def messages_for_llm(self, run_id: str, session_id: Optional[str] = None):
        # Your custom retrieval (e.g., RAG, summarization)
        return your_logic()

agent = Agent(..., context_manager=CustomContextManager())
```

See `examples/context_modes.py` for a full demonstration.

## Configuration

Environment variables (see `.env.example`):

```bash
# Database & Cache
POSTGRES_DSN=postgresql+psycopg://user:pass@localhost:5432/vel
REDIS_URL=redis://localhost:6379/0

# OpenAI
OPENAI_API_KEY=sk-...
OPENAI_API_BASE=https://api.openai.com/v1

# Google Gemini
GOOGLE_API_KEY=...

# Runner mode
VEL_RUNNER=local-async
```

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Format code
black agents/ agents_service/
ruff check agents/ agents_service/

# Type checking
mypy agents/
```

## Architecture

- **Agent**: Main orchestrator with dual execution modes (streaming/non-streaming)
- **ContextManager**: Memory layer for conversation history (configurable: full/stateless/limited)
- **Reducer**: Pure function for state transitions and effect generation (used in non-streaming mode)
- **Providers**: LLM-specific implementations with stream protocol translation
- **Tools**: Validated, async-capable function execution
- **Storage**: Dual-layer persistence (Postgres + Redis) with in-memory fallback

## License

MIT
