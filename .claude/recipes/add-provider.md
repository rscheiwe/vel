# Add Provider Recipe

**Goal:** Add support for a new LLM provider to Vel
**Prerequisites:** Provider API documentation, API key for testing
**Estimated Time:** 2-4 hours

---

## Steps

### Step 1: Create Provider File

Create `vel/providers/{provider_name}.py`:

```python
"""
{ProviderName} provider implementation.
"""
import os
from typing import Any, AsyncGenerator, Dict, List, Optional

from vel.providers.base import BaseProvider
from vel.events import StreamEvent


class {ProviderName}Provider(BaseProvider):
    """
    {ProviderName} API provider.

    Supports:
    - Streaming text generation
    - Tool/function calling
    - [Other features]
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv('{PROVIDER_NAME}_API_KEY')
        if not self.api_key:
            raise ValueError("{PROVIDER_NAME}_API_KEY required")
        # Initialize client
        self._client = ...

    async def stream(
        self,
        messages: List[Dict[str, Any]],
        model: str,
        tools: Optional[List[Dict]] = None,
        **kwargs
    ) -> AsyncGenerator[StreamEvent, None]:
        """Stream response from {ProviderName}."""
        # Convert messages to provider format
        provider_messages = self._convert_messages(messages)

        # Create streaming request
        response = await self._client.chat.completions.create(
            model=model,
            messages=provider_messages,
            tools=tools,
            stream=True,
            **kwargs
        )

        # Translate and yield events
        translator = {ProviderName}StreamTranslator()
        async for chunk in response:
            for event in translator.translate(chunk):
                yield event

    async def generate(
        self,
        messages: List[Dict[str, Any]],
        model: str,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate complete response (non-streaming)."""
        # Implementation
        ...

    def _convert_messages(self, messages: List[Dict]) -> List[Dict]:
        """Convert Vel messages to provider format."""
        # Implementation
        ...
```

### Step 2: Create Stream Translator

Add to `vel/providers/translators.py`:

```python
class {ProviderName}StreamTranslator:
    """Translates {ProviderName} stream chunks to Vel events."""

    def __init__(self):
        self._text_started = False
        self._tool_calls: Dict[int, Dict] = {}

    def translate(self, chunk: Any) -> List[StreamEvent]:
        """Convert provider chunk to Vel events."""
        events = []

        # Handle text content
        if hasattr(chunk, 'text'):
            if not self._text_started:
                events.append({'type': 'text-start'})
                self._text_started = True
            events.append({'type': 'text-delta', 'delta': chunk.text})

        # Handle tool calls
        if hasattr(chunk, 'tool_calls'):
            for tc in chunk.tool_calls:
                # Buffer partial tool calls
                # Emit tool-input-available when complete
                ...

        # Handle finish
        if hasattr(chunk, 'finish_reason') and chunk.finish_reason:
            if self._text_started:
                events.append({'type': 'text-end'})
            events.append({
                'type': 'finish-message',
                'finish_reason': chunk.finish_reason
            })

        return events
```

### Step 3: Add Message Translation

Update `vel/providers/message_translator.py`:

```python
def translate_to_{provider_name}(messages: List[Dict]) -> List[Dict]:
    """Convert Vel messages to {ProviderName} format."""
    result = []
    for msg in messages:
        translated = {
            'role': msg['role'],
            'content': msg.get('content', '')
        }
        # Handle tool calls
        if 'tool_calls' in msg:
            translated['tool_calls'] = [...]
        # Handle tool results
        if msg['role'] == 'tool':
            translated = {...}
        result.append(translated)
    return result
```

### Step 4: Register Provider

Update `vel/providers/__init__.py`:

```python
try:
    from .{provider_name} import {ProviderName}Provider
    self._providers['{provider_name}'] = {ProviderName}Provider()
except (ImportError, ValueError):
    pass  # Graceful fallback if deps missing or no API key
```

### Step 5: Add Tests

Create `tests/test_providers/test_{provider_name}.py`:

```python
import pytest
from vel.providers.{provider_name} import {ProviderName}Provider

@pytest.fixture
def provider():
    return {ProviderName}Provider(api_key='test-key')

def test_provider_init():
    provider = {ProviderName}Provider(api_key='test-key')
    assert provider.api_key == 'test-key'

@pytest.mark.asyncio
async def test_stream_text(provider, mock_response):
    events = []
    async for event in provider.stream([...], model='test'):
        events.append(event)

    assert any(e['type'] == 'text-delta' for e in events)
    assert events[-1]['type'] == 'finish-message'

@pytest.mark.asyncio
async def test_stream_tool_calls(provider, mock_tool_response):
    events = []
    async for event in provider.stream([...], model='test', tools=[...]):
        events.append(event)

    assert any(e['type'] == 'tool-input-available' for e in events)
```

### Step 6: Update Documentation

1. Add to `CLAUDE.md` Providers section
2. Update `docs/providers/` if exists
3. Add usage example to `examples/`

---

## Validation

Run these checks before submitting:

```bash
# Unit tests pass
pytest tests/test_providers/test_{provider_name}.py -v

# Type hints valid
mypy vel/providers/{provider_name}.py

# Integration test (requires API key)
python -c "
from vel import Agent
agent = Agent(model={'provider': '{provider_name}', 'model': 'model-id'})
# Quick test
"
```

---

## Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| Import error | Missing dependency | Add to requirements.txt |
| API key error | Env var not set | Set `{PROVIDER_NAME}_API_KEY` |
| Event sequence wrong | Translator state issue | Check state machine logic |
| Tool calls not working | Schema mismatch | Verify tool format in provider docs |

---

## Reference

- `.claude/rules/provider-development.md` - Coding guidelines
- `docs/adr/003-provider-translation.md` - Architecture decision
- `vel/providers/openai.py` - Reference implementation
