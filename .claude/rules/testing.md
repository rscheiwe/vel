---
paths:
  - "tests/**/*.py"
description: "Testing conventions for Vel"
---

# Testing Guidelines

## Test Structure

```
tests/
├── test_agent.py           # Core agent tests
├── test_events.py          # Event serialization
├── test_memory.py          # FactStore tests
├── test_memory_context.py  # ContextManager integration
├── test_trajectory_store.py
├── test_judge.py
├── test_strategy_extractor.py
├── test_memory_consolidator.py
├── test_rlm.py
├── test_thinking.py
├── test_tools.py
└── test_providers/
    └── test_openai.py
```

## Async Tests

Use `@pytest.mark.asyncio` for async functions:

```python
import pytest

@pytest.mark.asyncio
async def test_stream_events():
    agent = Agent(...)
    events = []
    async for event in agent.run_stream({'message': 'test'}):
        events.append(event)
    assert events[-1]['type'] == 'finish'
```

## Fixtures

### Database Fixtures

Use temporary databases for isolation:

```python
@pytest.fixture
def db_path(tmp_path):
    return str(tmp_path / "test.db")

@pytest.fixture
def fact_store(db_path):
    return FactStore(db_path)
```

### Mock Embeddings

Use hash-based embeddings for determinism:

```python
@pytest.fixture
def mock_embeddings():
    def embed(text: str) -> List[float]:
        h = hashlib.sha256(text.encode()).digest()
        return [b / 255.0 for b in h[:128]]
    return embed
```

### Mock Providers

Create mock providers for unit tests:

```python
@pytest.fixture
def mock_provider():
    class MockProvider(BaseProvider):
        async def stream(self, messages, model, **kwargs):
            yield TextDeltaEvent(delta="Hello")
            yield FinishMessageEvent(finish_reason="stop")
    return MockProvider()
```

## Assertions

### Event Assertions

```python
def assert_has_event(events: List[Dict], event_type: str) -> Dict:
    """Find and return event of given type."""
    for e in events:
        if e['type'] == event_type:
            return e
    pytest.fail(f"No {event_type} event found")
```

### Tool Call Assertions

```python
def assert_tool_called(events: List[Dict], tool_name: str) -> Dict:
    """Assert tool was called and return the event."""
    for e in events:
        if e['type'] == 'tool-input-available' and e['tool_name'] == tool_name:
            return e
    pytest.fail(f"Tool {tool_name} was not called")
```

## Test Categories

### Unit Tests

Test individual functions in isolation:

```python
def test_reduce_text_event():
    state = State(step=0, messages=[])
    new_state, effects = reduce(state, {'type': 'text-delta', 'delta': 'hi'})
    assert len(effects) == 1
    assert effects[0]['type'] == 'emit'
```

### Integration Tests

Test component interactions:

```python
@pytest.mark.asyncio
async def test_agent_with_tools():
    agent = Agent(
        model={'provider': 'mock', 'model': 'test'},
        tools=[ToolSpec.from_function(mock_tool)]
    )
    result = await agent.run({'message': 'call the tool'})
    assert 'tool_result' in result
```

### E2E Tests

Test full workflows (use sparingly, they're slow):

```python
@pytest.mark.slow
@pytest.mark.asyncio
async def test_full_conversation():
    agent = Agent(...)
    # Multi-turn conversation test
```

## Running Tests

```bash
# All tests
pytest tests/

# Specific file
pytest tests/test_memory.py -v

# Skip slow tests
pytest tests/ -m "not slow"

# With coverage
pytest tests/ --cov=vel --cov-report=html
```

## Common Patterns

### Testing Streaming

```python
@pytest.mark.asyncio
async def test_streaming():
    events = []
    async for event in agent.run_stream(input):
        events.append(event)

    # Check event sequence
    types = [e['type'] for e in events]
    assert 'text-start' in types
    assert 'text-delta' in types
    assert types[-1] == 'finish'
```

### Testing Error Handling

```python
@pytest.mark.asyncio
async def test_guardrail_rejection():
    with pytest.raises(GuardrailError) as exc_info:
        await agent.run({'message': 'blocked content'})
    assert 'PII detected' in str(exc_info.value)
```
