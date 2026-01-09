# ADR-003: Multi-Provider Translation Architecture

**Status:** Accepted
**Date:** 2025-01-08
**Decision Makers:** Vel Core Team

---

## Context

Vel supports multiple LLM providers (OpenAI, Anthropic, Gemini), each with:
- Different message formats (roles, content structure)
- Different streaming event formats
- Different tool call representations
- Different API authentication patterns

## Decision

Implement a **three-layer translation architecture**:

### Layer 1: Provider Interface

```python
class BaseProvider(ABC):
    async def stream(...) -> AsyncGenerator[StreamEvent, None]
    async def generate(...) -> Dict[str, Any]
```

All providers implement this interface, enabling interchangeable use.

### Layer 2: Message Translation

`vel/providers/message_translator.py` handles:
- Vel `ModelMessage` → Provider-specific message format
- Role mapping (assistant, user, tool, system)
- Tool result formatting
- Thinking/reasoning block handling

### Layer 3: Stream Translation

`vel/providers/translators.py` contains per-provider translators:
- `OpenAIStreamTranslator`: OpenAI delta events → Vel events
- `AnthropicStreamTranslator`: Anthropic content blocks → Vel events
- `GeminiStreamTranslator`: Gemini chunks → Vel events

### Per-Instance API Keys

Providers support per-instance API keys for multi-tenant scenarios:

```python
model = {
    'provider': 'openai',
    'model': 'gpt-4o',
    'api_key': 'sk-...'  # Optional, overrides env var
}
```

## Consequences

### Positive

1. **Provider Independence**: Swap providers without client changes
2. **Multi-Tenant Support**: Different API keys per agent instance
3. **Clean Separation**: Each translator handles one provider's quirks
4. **Testability**: Translators can be unit tested in isolation

### Negative

1. **Code Volume**: ~1600 lines in translators.py
2. **Maintenance**: Provider API changes require translator updates
3. **Feature Gaps**: Some provider features may not have Vel equivalents

## Implementation Notes

### Adding a New Provider

1. Create `vel/providers/{name}.py` implementing `BaseProvider`
2. Add translator class to `vel/providers/translators.py`
3. Add message format handler to `vel/providers/message_translator.py`
4. Register in `vel/providers/__init__.py`

### Provider Registry

Soft-loads providers (no error if dependencies missing):

```python
try:
    self._providers['openai'] = OpenAIProvider()
except (ImportError, ValueError):
    pass  # Graceful degradation
```

## References

- `vel/providers/base.py` - BaseProvider interface
- `vel/providers/openai.py` - OpenAI implementation
- `vel/providers/anthropic.py` - Anthropic implementation
- `vel/providers/google.py` - Gemini implementation
- `vel/providers/translators.py` - Stream translators
