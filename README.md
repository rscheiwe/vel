# VEL

## Agent Runtime (12-Factor Agents Aligned)

A production-ready AI agent runtime aligned with [12-Factor Agent principles](https://github.com/humanlayer/12-factor-agents) by Dex and contributors. Built for reliability, scalability, and maintainability with streaming responses, multiple LLM providers, and event-driven architecture.

## Features

- **Dual Execution Modes**: Streaming (SSE) and non-streaming (JSON) responses
- **Multiple LLM Providers**: OpenAI, Google Gemini, and Anthropic Claude with plug-and-play architecture
- **Generation Configuration**: Full control over model parameters (temperature, max_tokens, top_p, etc.) with per-run override support - matches Vercel AI SDK flexibility
- **Stream Protocol**: Vercel AI SDK **V5 UI Stream Protocol** compatible - works seamlessly with React `useChat()` and frontend components (100% parity)
  - Exact event naming (`tool-call`, `tool-result`, etc.)
  - Response metadata (token usage tracking)
  - Source events (citations and grounding)
  - File events (inline data support)
  - Anthropic thinking blocks
  - Enhanced error details
- **Message Aggregation**: MessageReducer for converting streaming events to Vercel AI SDK message format for database storage
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
- [Stream Protocol Parity](PARITY_STATUS.md) - Vercel AI SDK V5 UI Stream Protocol compatibility status (100% parity)

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

Vel uses the [Vercel AI SDK V5 UI Stream Protocol](https://ai-sdk.dev/docs/ai-sdk-ui/stream-protocol) for frontend-compatible event streaming:

- `text-start`, `text-delta`, `text-end` - Text content chunks
- `tool-input-start`, `tool-input-delta` - Tool input streaming
- `tool-input-available` - Complete tool input ready for execution
- `tool-output-available` - Tool execution result
- `response-metadata` - Token usage and model info
- `source` - Citations and grounding (Gemini)
- `file` - Inline file attachments
- `error`, `finish-message` - Error handling and completion

**Frontend Compatible:** Works seamlessly with React's `useChat()`, `useCompletion()`, and other Vercel AI SDK frontend components. Each provider translates native events into V5-compatible standardized events.

### Message Aggregation

**MessageReducer** aggregates streaming events into structured messages for database storage:

```python
from vel import Agent, MessageReducer

# Create reducer
reducer = MessageReducer()
reducer.add_user_message("What's the weather in San Francisco?")

# Stream agent response
agent = Agent(
    id='weather-agent',
    model={'provider': 'openai', 'model': 'gpt-4o'},
    tools=['get_weather']
)

async for event in agent.run_stream({'message': "What's the weather in SF?"}):
    reducer.process_event(event)

# Get Vercel AI SDK compatible messages
messages = reducer.get_messages()
# [
#   {user message},
#   {assistant message with parts: [tool-call, tool-result, text]}
# ]

# Store in database
for msg in messages:
    await db.insert_message(msg)
```

**Features:**
- ✓ Vercel AI SDK `useChat` hook compatible
- ✓ Aggregates text, tool calls, and results into parts array
- ✓ Provider metadata (OpenAI message/call IDs)
- ✓ Custom message IDs and metadata support

See [Message Aggregation docs](https://rscheiwe.github.io/vel/stream-protocol#message-aggregation) for complete details.

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

## Generation Configuration

Control model behavior with fine-grained generation parameters. Matches the flexibility of Vercel AI SDK's `streamText()` function.

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
    generation_config={'temperature': 0}  # Override to 0 for this run only
)

# Works with streaming too
async for event in agent.run_stream(
    {'message': 'Explain AI'},
    generation_config={'max_tokens': 100}  # Brief response
):
    print(event)
```

### Supported Parameters

#### Common (All Providers)
- `temperature` - Sampling temperature (0-2, default varies by provider)
- `max_tokens` - Maximum output tokens
- `top_p` - Nucleus sampling (0-1)
- `stop` - Stop sequences (list of strings)

#### OpenAI
- `presence_penalty` - Penalize new tokens (-2 to 2)
- `frequency_penalty` - Penalize repeated tokens (-2 to 2)
- `seed` - Reproducibility seed (integer)
- `logit_bias` - Token probability adjustments (dict)

#### Anthropic
- `top_k` - Top-K sampling (integer)
- `stop_sequences` - Alternative to `stop` (list of strings)

#### Google Gemini
- `top_k` - Top-K sampling (integer)
- `max_output_tokens` - Alternative to `max_tokens` (integer)
- `stop_sequences` - Alternative to `stop` (list of strings)

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

## Examples

Vel includes comprehensive examples demonstrating various patterns:

**Core Examples:**
- `examples/quickstart.py` - Basic agent usage (streaming & non-streaming)
- `examples/message_reducer_example.py` - MessageReducer for database storage
- `examples/context_modes.py` - Different context management strategies
- `examples/generation_config_example.py` - Model parameter control
- `examples/prompt_templates.py` - Prompt template system

**Multi-Step Agent Examples:**
- `examples/multi_step_simple.py` - Basic multi-step pattern (websearch + news)
- `examples/multi_step_analysis.py` - Problem analysis with analyze tool
- `examples/multi_step_decision.py` - Decision-making with decide tool
- `examples/multi_step_complex.py` - Complex reasoning with all tools
- `examples/comprehensive_multi_step_agent.py` - Full multi-step demonstration

**Run with:**
```bash
python examples/quickstart.py
python examples/message_reducer_example.py
python examples/multi_step_simple.py
```

Or use VS Code debug configurations (see `.vscode/launch.json`).

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
