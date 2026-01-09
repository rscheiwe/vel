---
paths:
  - "vel/providers/**/*.py"
description: "Guidelines for implementing LLM providers"
---

# Provider Development Guidelines

## BaseProvider Interface

All providers must implement:

```python
from vel.providers.base import BaseProvider

class MyProvider(BaseProvider):
    async def stream(
        self,
        messages: List[Dict],
        model: str,
        tools: Optional[List[ToolSpec]] = None,
        **kwargs
    ) -> AsyncGenerator[StreamEvent, None]:
        """Stream LLM response as events."""
        ...

    async def generate(
        self,
        messages: List[Dict],
        model: str,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate complete response (non-streaming)."""
        ...
```

## Per-Instance API Keys

Support API key override for multi-tenant use:

```python
def __init__(self, api_key: Optional[str] = None):
    self.api_key = api_key or os.getenv('MY_PROVIDER_API_KEY')
    if not self.api_key:
        raise ValueError("API key required")
```

## Stream Translation

Create a translator in `translators.py`:

```python
class MyProviderStreamTranslator:
    def __init__(self):
        self._buffer = ""
        self._tool_calls = {}

    def translate(self, chunk: Any) -> List[StreamEvent]:
        """Convert provider chunk to Vel events."""
        events = []
        # Parse chunk, emit appropriate events
        return events
```

## Event Mapping

Map provider events to Vercel AI SDK V5:

| Provider Event | Vel Event |
|---------------|-----------|
| Text chunk | `text-delta` |
| Tool call start | `tool-input-start` |
| Tool call complete | `tool-input-available` |
| Finish | `finish-message` |
| Error | `error` |

## Error Handling

- Catch provider-specific errors and wrap in Vel exceptions
- Include original error details for debugging
- Emit `error` event on stream failures

```python
try:
    async for chunk in native_stream:
        yield self.translator.translate(chunk)
except ProviderRateLimitError as e:
    yield ErrorEvent(message=str(e), code='rate_limit')
    raise
```

## Soft Loading

Register in `providers/__init__.py` with graceful fallback:

```python
try:
    from .my_provider import MyProvider
    self._providers['my_provider'] = MyProvider()
except (ImportError, ValueError):
    pass  # Missing deps or API key - silent skip
```

## Testing

Add tests in `tests/test_providers/`:

```python
@pytest.mark.asyncio
async def test_stream_text():
    provider = MyProvider(api_key='test-key')
    events = []
    async for event in provider.stream([...], model='test'):
        events.append(event)
    assert any(e['type'] == 'text-delta' for e in events)
```

## Reference Implementations

- `vel/providers/openai.py` - OpenAI Chat Completions
- `vel/providers/anthropic.py` - Anthropic Claude
- `vel/providers/google.py` - Google Gemini
