# VEL

## Agent Runtime (12-Factor Agents Aligned)

A production-ready AI agent runtime aligned with [12-Factor Agent principles](https://github.com/humanlayer/12-factor-agents) by Dex and contributors. Built for reliability, scalability, and maintainability with streaming responses, multiple LLM providers, and event-driven architecture.

## Features

- **Dual Execution Modes**: Streaming (SSE) and non-streaming (JSON) responses
- **Multiple LLM Providers**: OpenAI, Google Gemini, and Anthropic Claude with plug-and-play architecture
- **Stream Protocol**: Vercel AI SDK V3-compatible event system for provider-agnostic streaming (100% parity)
  - Exact event naming (`tool-call`, `tool-result`, etc.)
  - Response metadata (token usage tracking)
  - Source events (citations and grounding)
  - File events (inline data support)
  - Anthropic thinking blocks
  - Enhanced error details
- **Tool System**: JSON schema-validated tools with async support
- **Flexible Prompts**: Jinja2 templating with XML formatting, environment-based configuration, and version control
- **Persistent Storage**: PostgreSQL for durability, Redis for caching
- **FastAPI Service**: Production-ready REST API with health checks

## Documentation

**📚 [Complete Documentation](https://rscheiwe.github.io/vel)**

- [Getting Started](https://rscheiwe.github.io/vel/getting-started) - Installation and quick start
- [Session Management](https://rscheiwe.github.io/vel/sessions) - Multi-turn conversations
- [Prompt Templates](https://rscheiwe.github.io/vel/prompts) - Flexible prompt management with Jinja2 and XML
- [Providers](https://rscheiwe.github.io/vel/providers) - OpenAI, Gemini, and Claude configuration
- [Tools](https://rscheiwe.github.io/vel/tools) - Custom tool creation
- [Stream Protocol](https://rscheiwe.github.io/vel/stream-protocol) - Event streaming reference
- [Event Translators](https://rscheiwe.github.io/vel/event-translators) - Translate provider events to Vel format (OpenAI, Anthropic, Gemini)
- [Memory System](https://rscheiwe.github.io/vel/memory) - Optional memory with Fact Store and ReasoningBank
- [API Reference](https://rscheiwe.github.io/vel/api-reference) - Complete API docs
- [12-Factor Alignment](https://rscheiwe.github.io/vel/12-factor-alignment) - Production-ready agent principles
- [Stream Protocol Parity](PARITY_STATUS.md) - Vercel AI SDK V3 compatibility status (100% parity)

## Project Structure

```
vel/
├── providers/      # LLM provider implementations (OpenAI, Gemini, Anthropic)
├── storage/        # Storage layer (Postgres, Redis)
├── tools/          # Tool registry and specifications
├── prompts/        # Prompt templates with Jinja2 and XML formatting
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
from vel import Agent

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

### Anthropic Claude

```python
agent = Agent(
    id='my-agent',
    model={'provider': 'anthropic', 'model': 'claude-sonnet-4-20250514'}
)
```

## Session Management (Multi-Turn Conversations)

Sessions enable multi-turn conversations where the agent remembers context across multiple calls.

### Basic Session Usage

```python
agent = Agent(
    id='my-agent',
    model={'provider': 'openai', 'model': 'gpt-4o'},
    session_persistence='transient'  # or 'persistent'
)

# Multi-turn conversation - same session_id = shared history
session_id = 'user-123'

answer1 = await agent.run({'message': 'My name is Alice'}, session_id=session_id)
# "Hello Alice! How can I help you?"

answer2 = await agent.run({'message': 'What is my name?'}, session_id=session_id)
# "Your name is Alice."
```

### Session Persistence Modes

#### Transient (Default - Fast, Not Persistent)

```python
agent = Agent(
    id='my-agent',
    model={'provider': 'openai', 'model': 'gpt-4o'},
    session_persistence='transient'  # Message history lost on restart
)
```

#### Persistent (Database-Backed, Survives Restarts)

```python
# Requires POSTGRES_DSN configured
agent = Agent(
    id='my-agent',
    model={'provider': 'openai', 'model': 'gpt-4o'},
    session_persistence='persistent'  # Message history stored in Postgres
)
```

### Message History Modes

Control how much conversation history is retained:

```python
from vel import ContextManager, StatelessContextManager

# Full message history (default)
agent = Agent(..., context_manager=ContextManager())

# No message history (stateless)
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

# Anthropic Claude
ANTHROPIC_API_KEY=sk-ant-...

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
black vel/ agents_service/
ruff check vel/ agents_service/

# Type checking
mypy vel/
```

## Architecture

Vel is designed following the [12-Factor Agent principles](https://github.com/humanlayer/12-factor-agents) (by Dex and contributors) for production-ready AI applications. See our [implementation guide](docs/12-factor-alignment.md) for details.

- **Agent**: Main orchestrator with dual execution modes (streaming/non-streaming)
- **ContextManager**: Message history layer for conversation turns (configurable: full/stateless/limited)
- **Reducer**: Pure function for state transitions and effect generation (stateless, reproducible)
- **Providers**: LLM-specific implementations with stream protocol translation
- **Tools**: Validated, async-capable function execution (structured outputs)
- **Storage**: Dual-layer persistence (Postgres + Redis) with in-memory fallback
- **Memory** (optional): Fact store and ReasoningBank for long-term structured data and strategy learning

**Key Principles:**

- ✓ Own your prompts - Direct control, no abstractions
- ✓ Own your context window - Custom context managers
- ✓ Stateless reducer - Predictable, reproducible behavior
- ✓ Small, focused agents - Composable design

## TODO

- [ ] Add features from OpenAI Agent SDK (tool responses, e.g.)
- [ ] Test Gemini tool calling
- [ ] Finish Postgres integration
- [ ] Add knowledge-graph memory layer
- [ ] Add example of how to create Vel agents via a tool
- [ ] Add guardrails
- [x] ~~Update ReasoningBank to include e2e implementation as described in Google's paper~~ (Phase 1 complete, see `docs/Memory/reasoningbank-phase2-roadmap.md` for Phase 2)

## License

MIT
