# Vel Documentation

Comprehensive documentation for the Vel agent runtime.

## Table of Contents

- [Getting Started](getting-started.md) - Installation and quick start
- [Session Management](sessions.md) - Multi-turn conversations and memory
- [Providers](providers.md) - LLM providers (OpenAI, Gemini)
- [Tools](tools.md) - Tool system and custom tools
- [Stream Protocol](stream-protocol.md) - Event streaming specification
- [API Reference](api-reference.md) - Complete API documentation

## Quick Links

### Core Concepts

- **Agent** - Main orchestrator with dual execution modes (streaming/non-streaming)
- **Sessions** - Multi-turn conversation management with hybrid storage
- **Providers** - LLM-specific implementations with agnostic stream protocol
- **Tools** - Validated, async-capable function execution
- **Context Manager** - Memory layer for conversation history
- **Storage** - Dual-layer persistence (Postgres + Redis) with in-memory fallback

### Key Features

✓ **Dual Execution Modes** - Streaming (SSE) and non-streaming (JSON)
✓ **Multiple Providers** - OpenAI and Google Gemini
✓ **Session Management** - Persistent or in-memory multi-turn conversations
✓ **Stream Protocol** - Vercel AI SDK-compatible events
✓ **Tool System** - JSON schema-validated tools
✓ **Configurable Memory** - Full, stateless, or limited history

## Architecture Overview

```
┌─────────────────────────────────────────┐
│              Agent                      │
│  ┌─────────────┐     ┌──────────────┐  │
│  │ run()       │     │ run_stream() │  │
│  │ (sync)      │     │ (streaming)  │  │
│  └─────────────┘     └──────────────┘  │
└─────────────────────────────────────────┘
           │                    │
           ├────────────────────┤
           │                    │
    ┌──────▼──────┐      ┌─────▼─────┐
    │  Provider   │      │  Context  │
    │  Registry   │      │  Manager  │
    └─────────────┘      └───────────┘
           │                    │
    ┌──────▼──────┐      ┌─────▼─────┐
    │   OpenAI    │      │  Session  │
    │   Gemini    │      │  Storage  │
    └─────────────┘      └───────────┘
                               │
                        ┌──────▼──────┐
                        │   Postgres  │
                        │   Redis     │
                        │   Memory    │
                        └─────────────┘
```

## Examples

All examples are in the `examples/` directory:

- `quickstart.py` - Basic agent usage
- `test_both_modes.py` - Streaming vs non-streaming with tools
- `context_modes.py` - Session management demonstrations

## Contributing

See the main [README.md](../README.md) for development setup and guidelines.

## License

MIT
