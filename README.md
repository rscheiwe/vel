# VEL

## Agent Runtime (12-Factor Agents Aligned)

A production-ready AI agent runtime aligned with [12-Factor Agent principles](https://github.com/humanlayer/12-factor-agents) by Dex and contributors. Built for reliability, scalability, and maintainability with streaming responses, multiple LLM providers, and event-driven architecture.

## Features

- **Dual Execution Modes**: Streaming (SSE) and non-streaming (JSON) responses
- **Multiple LLM Providers**: OpenAI, Google Gemini, and Anthropic Claude with plug-and-play architecture
- **RLM (Recursive Language Model)**: Handle 5MB+ documents through iterative reasoning, context probing, and budget-controlled execution
- **Generation Configuration**: Full control over model parameters (temperature, max_tokens, top_p, etc.) with per-run override support - matches Vercel AI SDK flexibility
- **Stream Protocol**: Vercel AI SDK **V5 UI Stream Protocol** compatible - works seamlessly with React `useChat()` and frontend components (100% parity)
  - Exact event naming (`tool-call`, `tool-result`, etc.)
  - Custom `data-*` events with transient flag for RAG citations, progress tracking, and analytics
  - Response metadata (token usage tracking)
  - Source events (citations and grounding)
  - File events (inline data support)
  - Reasoning events (OpenAI o1/o3 chain-of-thought streaming)
  - Anthropic thinking blocks
  - Enhanced error details
- **Message Aggregation**: MessageReducer for converting streaming events (text, reasoning, tools) to Vercel AI SDK message format
- **Tool System**: JSON schema-validated tools with async support
- **Flexible Prompts**: Jinja2 templating with XML formatting, environment-based configuration, and version control
- **Message Format Compatibility**: Works with Vercel AI SDK's `convertToModelMessages()` and includes Python converter for UIMessage → ModelMessage
- **Automatic Provider Translation**: Converts ModelMessage format to provider-specific formats (OpenAI/Anthropic/Gemini) automatically

## Message Format Compatibility

Vel supports multiple message format patterns for seamless integration:

### React Frontend + Vel Backend
```typescript
// Frontend (React with Vercel AI SDK)
import { useChat, convertToModelMessages } from 'ai';

const { messages } = useChat();
const modelMessages = convertToModelMessages(messages);

fetch('/api/chat', {
  body: JSON.stringify({ messages: modelMessages })
});
```

```python
# Backend (FastAPI with Vel)
from vel import Agent

@app.post("/api/chat")
async def chat(request: dict):
    agent = Agent(
        id='chat',
        model={'provider': 'openai', 'model': 'gpt-4o'}
    )

    # Vel translates ModelMessage → OpenAI format automatically
    response = await agent.run({'messages': request['messages']})
    return {'response': response}
```

### Python-Only Applications
```python
from vel import Agent
from vel.utils import convert_to_model_messages

# Option 1: Build ModelMessages manually
messages = [
    {'role': 'user', 'content': 'Hello'},
    {'role': 'assistant', 'content': 'Hi!'}
]

# Option 2: Convert UIMessages from database
ui_messages = db.get_conversation(user_id)
messages = convert_to_model_messages(ui_messages)

# Use with any provider - translation is automatic
agent = Agent(
    id='chat',
    model={'provider': 'anthropic', 'model': 'claude-3-5-sonnet-20241022'}
)
response = await agent.run({'messages': messages})
```

**See [Message Formats Documentation](https://rscheiwe.github.io/vel/message-formats) for detailed patterns and examples.**

## Documentation

**📚 [Complete Documentation](https://rscheiwe.github.io/vel)**

- [Getting Started](https://rscheiwe.github.io/vel/getting-started) - Installation and quick start
- [Message Formats](https://rscheiwe.github.io/vel/message-formats) - UIMessage, ModelMessage, and automatic provider translation
- [Session Management](https://rscheiwe.github.io/vel/sessions) - Multi-turn conversations
- [RLM (Recursive Language Model)](https://rscheiwe.github.io/vel/rlm) - Long context support (5MB+) with iterative reasoning
- [Prompt Templates](https://rscheiwe.github.io/vel/prompts) - Flexible prompt management with Jinja2 and XML
- [Providers](https://rscheiwe.github.io/vel/providers) - OpenAI, Gemini, and Claude configuration
- [Tools](https://rscheiwe.github.io/vel/tools) - Custom tool creation
- [Stream Protocol](https://rscheiwe.github.io/vel/stream-protocol) - Event streaming reference
- [Event Translators](https://rscheiwe.github.io/vel/event-translators) - Protocol adapter architecture and usage guide
  - [Using Translators Directly](https://rscheiwe.github.io/vel/using-translators) - Custom orchestration with frontend compatibility
- [Memory System](https://rscheiwe.github.io/vel/memory) - Optional memory with Fact Store and ReasoningBank
- [API Reference](https://rscheiwe.github.io/vel/api-reference) - Complete API docs
- [12-Factor Alignment](https://rscheiwe.github.io/vel/12-factor-alignment) - Production-ready agent principles
- [Stream Protocol Parity](PARITY_STATUS.md) - Vercel AI SDK V5 UI Stream Protocol compatibility status (100% parity)

## Project Structure

```
vel/
├── providers/      # LLM provider implementations (OpenAI, Gemini, Anthropic)
├── rlm/            # RLM (Recursive Language Model) for long context support
├── tools/          # Tool registry and specifications
├── prompts/        # Prompt templates with Jinja2 and XML formatting
├── core/           # State management, reducer, context
├── events.py       # Stream protocol event definitions
└── agent.py        # Main Agent class
```

## Architecture

Vel uses a **two-layer architecture** based on the Single Responsibility Principle:

### Layer 1: Translators (Protocol Adapters)

- **Job:** Convert provider-specific → standard protocol
- **Scope:** Single LLM response stream
- **Stateful:** Only tracks current response (text blocks, tool calls)
- **Reusable:** Works with any orchestrator (Vel Agent, Mesh, LangGraph, custom)

### Layer 2: Agent (Orchestrator)

- **Job:** Multi-step execution, tool calling, context management
- **Scope:** Full agentic workflow
- **Stateful:** Sessions, context, run history
- **Opinionated:** Implements specific orchestration pattern

This separation enables **composability**: use Agent for turnkey workflows, or use Event Translators directly with custom orchestrators. See [Event Translators](https://rscheiwe.github.io/vel/event-translators) for complete architecture details and integration examples.

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

## ⚠️ Deprecation Notice

**Global tool registration is deprecated in v0.3.0 and will be removed in v2.0.**

**Old (deprecated):**
```python
from vel import ToolSpec, register_tool

register_tool(ToolSpec(...))  # ⚠️ DEPRECATED
agent = Agent(tools=['tool_name'])  # ⚠️ DEPRECATED
```

**New (recommended):**
```python
from vel import ToolSpec

tool = ToolSpec.from_function(your_function)
agent = Agent(tools=[tool])  # ✅ No registration needed!
```

See [Migration Guide](#migration-guide-global-registry--instance-tools-v20) below for details.

---

## Quick Start

### API Key Configuration

Vel supports two ways to provide API keys:

**1. Environment Variables (recommended for development)**
```bash
export OPENAI_API_KEY='sk-...'
export ANTHROPIC_API_KEY='sk-ant-...'
export GOOGLE_API_KEY='...'
```

**2. Explicit API Keys (recommended for libraries/production)**
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

This makes Vel suitable for:
- **Applications**: Use environment variables
- **Libraries**: Pass API keys programmatically
- **Multi-tenant**: Different agents can use different API keys

### Python SDK

```python
import asyncio
from vel import Agent, ToolSpec

# Define a tool
def get_weather(city: str) -> dict:
    """Get weather for a city."""
    return {'temp': 72, 'condition': 'sunny'}

weather_tool = ToolSpec.from_function(get_weather)

async def main():
    # Option 1: Use environment variable (OPENAI_API_KEY)
    agent = Agent(
        id='chat-general:v1',
        model={'provider': 'openai', 'model': 'gpt-4o'},
        tools=[weather_tool],  # Pass ToolSpec directly
        policies={'max_steps': 8}
    )

    # Option 2: Explicit API key
    agent = Agent(
        id='chat-general:v1',
        model={'provider': 'openai', 'model': 'gpt-4o', 'api_key': 'sk-...'},
        tools=[weather_tool],  # Pass ToolSpec directly
        policies={'max_steps': 8}
    )

    # Non-streaming mode
    answer = await agent.run({'message': 'What is the weather?'})
    print(answer)

    # Streaming mode
    async for event in agent.run_stream({'message': 'Tell me a story'}):
        print(event)

if __name__ == '__main__':
    asyncio.run(main())
```

## Stream Protocol

Vel uses the [Vercel AI SDK V5 UI Stream Protocol](https://ai-sdk.dev/docs/ai-sdk-ui/stream-protocol) for frontend-compatible event streaming:

- `text-start`, `text-delta`, `text-end` - Text content chunks
- `reasoning-start`, `reasoning-delta`, `reasoning-end` - Reasoning/chain-of-thought (o1/o3 models)
- `tool-input-start`, `tool-input-delta` - Tool input streaming
- `tool-input-available` - Complete tool input ready for execution
- `tool-output-available` - Tool execution result
- `start-step`, `finish-step` - Multi-step agent progress
- `data-*` - Custom application events (notifications, progress, metrics) with transient flag support
- `response-metadata` - Token usage and model info
- `source` - Citations and grounding (Gemini)
- `file` - Inline file attachments
- `error`, `finish-message` - Error handling and completion

**Frontend Compatible:** Works seamlessly with React's `useChat()`, `useCompletion()`, and other Vercel AI SDK frontend components. Each provider translates native events into V5-compatible standardized events.

#### Enhanced Error Handling

Vel automatically surfaces detailed error information without requiring manual print statements. Error events include:

```python
{
  'type': 'error',
  'error': 'max_tokens must be greater than thinking.budget_tokens',
  'errorCode': 'invalid_request_error',
  'errorType': 'InvalidRequestError',
  'statusCode': 400,
  'provider': 'anthropic',
  'details': {
    'type': 'error',
    'message': 'max_tokens must be greater than thinking.budget_tokens'
  }
}
```

**Automatic Logging:** Errors are automatically logged with full context:
```python
# Errors are logged automatically
agent = Agent(id='agent:v1', model={'provider': 'openai', 'model': 'gpt-4o'})

# If an error occurs, it's logged with full context
# No manual print statements needed!
async for event in agent.run_stream({'message': 'test'}):
    if event['type'] == 'error':
        # Full error context is available in the event
        print(f"Error from {event['provider']}: {event['error']}")
        if event.get('statusCode'):
            print(f"HTTP {event['statusCode']}")
```

**Python Logging:** Configure logging to see detailed error traces:
```python
import logging
logging.basicConfig(level=logging.ERROR)
# vel.agent logger will now output detailed error information
```

### Message Aggregation

**MessageReducer** aggregates streaming events into structured messages (Vercel AI SDK format):

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

# Use messages however you need (store in DB, return to client, etc.)
print(messages)
```

**With Reasoning (o1/o3 models):**

```python
# Create reducer for reasoning model
reducer = MessageReducer()
reducer.add_user_message("What is sqrt(169)?")

agent = Agent(
    id='reasoning-agent',
    model={'provider': 'openai-responses', 'model': 'o1'}
)

async for event in agent.run_stream({'message': 'What is sqrt(169)?'}):
    reducer.process_event(event)

messages = reducer.get_messages()
# assistant message parts: [
#   {'type': 'start-step'},
#   {'type': 'reasoning', 'text': '', 'state': 'done', 'providerMetadata': {...}},
#   {'type': 'text', 'text': 'The answer is 13', 'state': 'done'}
# ]
```

**Features:**
- ✓ Vercel AI SDK `useChat` hook compatible
- ✓ Aggregates text, reasoning, tool calls, and results into parts array
- ✓ Reasoning parts with provider metadata (o1/o3 models)
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

### Reasoning Models (o1/o3)

Vel supports OpenAI's reasoning models. **Use the Responses API provider** for reasoning event indicators:

```python
agent = Agent(
    id='reasoning-agent',
    model={
        'provider': 'openai-responses',  # Use Responses API for reasoning events
        'model': 'o1'  # or 'o1-mini', 'o3-mini'
    }
)

async for event in agent.run_stream({'message': 'Solve: sqrt(169)'}):
    if event['type'] == 'reasoning-start':
        print("🧠 Reasoning begins...")
    elif event['type'] == 'reasoning-delta':
        # Note: OpenAI often encrypts reasoning content, so deltas may be empty
        delta = event.get('delta', '')
        if delta:
            print(f"💭 {delta}", end='', flush=True)
    elif event['type'] == 'reasoning-end':
        print("\n✅ Reasoning complete")
    elif event['type'] == 'text-delta':
        print(event['delta'], end='', flush=True)
```

**Event Flow**:
1. `reasoning-start` - Reasoning block begins
2. `reasoning-delta` - Reasoning content (often empty/encrypted by OpenAI)
3. `reasoning-end` - Reasoning block ends
4. `text-start` → `text-delta`* → `text-end` - Final answer

**Note**: OpenAI encrypts reasoning content for o1/o3 models in most cases. You'll receive `reasoning-start` and `reasoning-end` events to indicate reasoning occurred, but `reasoning-delta` events may be empty. This matches the AI SDK behavior.

**See**: [examples/responses_api.py](examples/responses_api.py) for Responses API examples, [examples/reasoning_o1.py](examples/reasoning_o1.py) for Chat Completions API

## Session Management (Multi-Turn Conversations)

Sessions enable multi-turn conversations where the agent remembers context across multiple calls.

### Basic Session Usage

```python
agent = Agent(
    id='my-agent',
    model={'provider': 'openai', 'model': 'gpt-4o'}
)

# Multi-turn conversation - same session_id = shared history
session_id = 'user-123'

answer1 = await agent.run({'message': 'My name is Alice'}, session_id=session_id)
# "Hello Alice! How can I help you?"

answer2 = await agent.run({'message': 'What is my name?'}, session_id=session_id)
# "Your name is Alice."

# Note: Sessions are in-memory. For persistent storage, save/load messages yourself.
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

## RLM (Recursive Language Model) - Long Context Support

RLM is a middleware that enables agents to handle very long contexts (5MB+) through recursive reasoning and iterative context probing.

### How It Works

Instead of loading the entire context into the prompt, RLM:
1. **Probes context iteratively** using tools (search, read, summarize)
2. **Accumulates notes** in a scratchpad
3. **Reasons recursively** until reaching a FINAL() answer
4. **Enforces budgets** for cost and performance control

```python
from vel import Agent

# Enable RLM for long-context reasoning
agent = Agent(
    id='doc-analyzer:v1',
    model={'provider': 'openai', 'model': 'gpt-4o-mini'},
    rlm={
        'enabled': True,
        'depth': 1,  # Allow recursive sub-queries
        'control_model': {'provider': 'openai', 'model': 'gpt-4o-mini'},
        'writer_model': {'provider': 'openai', 'model': 'gpt-4o'},  # Optional
        'budgets': {
            'max_steps_root': 12,
            'max_tokens_total': 120000,
            'max_cost_usd': 0.50
        }
    }
)

# Use with large documents (5MB+)
with open('large_document.txt') as f:
    large_doc = f.read()

answer = await agent.run(
    input={'message': 'Summarize the key findings and recommendations.'},
    context_refs=large_doc  # RLM activates automatically
)
```

### Key Features

- **No context window limits** - Handle documents beyond model limits
- **Cost efficient** - Use cheap models for iteration, strong models for synthesis
- **Budget controls** - Hard limits on steps, tokens, and cost
- **Streaming support** - Emit RLM events (probes, notes, budget status)
- **REPL-style execution** - Optional `python_exec` for complex data processing (disabled by default)

### Tools

RLM provides three tools for context interaction:

- **context_probe** - Safe search/read/summarize operations (always enabled)
- **rlm_call** - Spawn recursive sub-queries for decomposition
- **python_exec** - Execute Python code with CONTEXT variable (⚠️ security risk, disabled by default)

### Documentation

See the [complete RLM guide](https://rscheiwe.github.io/vel/rlm) for:
- Detailed architecture and control flow
- Configuration options and tuning
- Security considerations for `python_exec`
- Streaming events
- Examples and best practices

### Example Output

```bash
python examples/rlm_basic.py
```

Inspired by [Alex Zhang's RLM approach](https://alexzhang13.github.io/blog/2025/rlm/).

## Configuration

Environment variables (see `.env.example`):

```bash
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
- `examples/rlm_basic.py` - RLM for long contexts (5MB+ documents)
- `examples/message_reducer_example.py` - MessageReducer for message aggregation
- `examples/custom_data_events.py` - Custom data-* events with transient flag
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
black vel/
ruff check vel/

# Type checking
mypy vel/
```

## Architecture

Vel is designed following the [12-Factor Agent principles](https://github.com/humanlayer/12-factor-agents) (by Dex and contributors) for production-ready AI applications. See our [implementation guide](docs/12-factor-alignment.md) for details.

- **Agent**: Main orchestrator with dual execution modes (streaming/non-streaming)
- **RLM**: Middleware for long-context reasoning (5MB+) with iterative probing and budget controls
- **ContextManager**: Message history layer for conversation turns (configurable: full/stateless/limited)
- **Reducer**: Pure function for state transitions and effect generation (stateless, reproducible)
- **Providers**: LLM-specific implementations with stream protocol translation
- **Tools**: Validated, async-capable function execution (structured outputs)
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
- [ ] Stress test RLM with real-world large documents
- [x] ~~Update ReasoningBank to include e2e implementation as described in Google's paper~~ (Phase 1 complete, see `docs/Memory/reasoningbank-phase2-roadmap.md` for Phase 2)
- [x] ~~Add RLM (Recursive Language Model) support for long contexts~~ (Complete - see `docs/rlm.md`)

## Migration Guide: Global Registry → Instance Tools (v2.0)

**Status:** Global tool registration is deprecated in v0.3.0 and will be removed in v2.0.

### What's Changing

**Before (v0.x - Deprecated):**
```python
from vel import ToolSpec, register_tool, Agent

# Register globally
tool = ToolSpec(name='get_weather', input_schema={...}, output_schema={...}, handler=my_handler)
register_tool(tool)  # ⚠️ DEPRECATED

# Use by string
agent = Agent(tools=['get_weather'])  # ⚠️ DEPRECATED
```

**After (v2.0 - Recommended):**
```python
from vel import ToolSpec, Agent

# Define function
def get_weather(city: str) -> dict:
    """Get weather for a city."""
    return {'temp': 72, 'condition': 'sunny'}

# Wrap in ToolSpec (auto-generates schemas)
tool = ToolSpec.from_function(get_weather)

# Pass directly to agent
agent = Agent(tools=[tool])  # ✅ No registration needed!
```

### Why?

1. **No Global State** - Tools scoped to agent instances
2. **Type Safety** - No string magic, IDE autocomplete works
3. **Better Testing** - No need to mock global registries
4. **Runtime Tools** - Create tools dynamically (perfect for UIs)
5. **Industry Standard** - Matches OpenAI Agents SDK pattern

### Migration Steps

1. **Replace `register_tool()` calls:**
   ```python
   # Before
   register_tool(ToolSpec(...))

   # After
   tool = ToolSpec.from_function(your_function)
   ```

2. **Update Agent initialization:**
   ```python
   # Before
   agent = Agent(tools=['tool_name'])

   # After
   agent = Agent(tools=[tool])
   ```

3. **For shared tools, define once and reuse:**
   ```python
   shared_tool = ToolSpec.from_function(my_function)
   agent1 = Agent(tools=[shared_tool])
   agent2 = Agent(tools=[shared_tool])
   ```

### Timeline

- **v0.3.0** (Current): Deprecation warnings added, old code still works
- **v1.x**: Warnings continue, old code still works
- **v2.0**: Breaking changes - `register_tool()` removed, `Agent` only accepts `List[ToolSpec]`

### Examples

See `examples/dynamic_tools.py` for complete migration examples.

---

## License

MIT
