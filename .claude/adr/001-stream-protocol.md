# ADR-001: Vercel AI SDK V5 Stream Protocol

**Status:** Accepted
**Date:** 2025-01-08
**Decision Makers:** Vel Core Team

---

## Context

Vel needs a unified event format for streaming LLM responses across multiple providers (OpenAI, Anthropic, Gemini). Each provider has its own native streaming format:

- **OpenAI**: Server-Sent Events with `delta` objects
- **Anthropic**: `content_block_delta` events with index tracking
- **Gemini**: Streaming `GenerateContentResponse` chunks

The client application needs a consistent interface regardless of which provider is used.

## Decision

Implement the **Vercel AI SDK V5 Stream Protocol** as Vel's canonical event format.

### Event Types

```python
EventType = Literal[
    'start', 'text-start', 'text-delta', 'text-end',
    'reasoning-start', 'reasoning-delta', 'reasoning-end',
    'tool-input-start', 'tool-input-delta', 'tool-input-available',
    'tool-output-available', 'start-step', 'finish-step',
    'finish-message', 'finish', 'error'
]
```

### Translation Layer

Each provider has a dedicated translator class in `vel/providers/translators.py`:
- `OpenAIStreamTranslator`
- `AnthropicStreamTranslator`
- `GeminiStreamTranslator`

Translators convert native provider events to Vel events in real-time.

## Consequences

### Positive

1. **React `useChat()` Compatibility**: Direct integration with Vercel AI SDK frontend hooks
2. **Provider Agnosticism**: Clients never see provider-specific formats
3. **Standardized Tooling**: Tool calls, reasoning blocks, and metadata follow consistent patterns
4. **Ecosystem Alignment**: Leverages established, well-documented protocol

### Negative

1. **Translator Maintenance**: Each provider update may require translator changes (~1600 lines total)
2. **Protocol Coupling**: Breaking changes in V5 spec would require coordinated updates
3. **Feature Parity**: Some provider-specific features may not map cleanly to V5 events

## Alternatives Considered

1. **Provider-Specific Protocols**: Rejected due to client complexity and maintenance burden
2. **Custom Protocol**: Rejected to avoid reinventing established patterns
3. **GraphQL Subscriptions**: Rejected due to overhead and complexity for streaming use case

## References

- [Vercel AI SDK Stream Protocol](https://sdk.vercel.ai/docs/ai-sdk-ui/stream-protocol)
- `vel/events.py` - Event definitions
- `vel/providers/translators.py` - Translator implementations
