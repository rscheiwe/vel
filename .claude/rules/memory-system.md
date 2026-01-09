---
paths:
  - "vel/memory/**/*.py"
  - "vel/core/context.py"
description: "Memory system implementation guidelines"
---

# Memory System Guidelines

## Core Principle: Opt-In

Memory is **disabled by default**. Never assume memory is available:

```python
# Good - check before use
if ctx.memory_enabled('facts'):
    ctx.fact_put(namespace, key, value)

# Bad - assumes memory is configured
ctx.fact_put(namespace, key, value)  # May raise
```

## FactStore Patterns

### Namespace Convention

Use hierarchical namespaces:

```python
# User-specific facts
ctx.fact_put('user:alice', 'theme', 'dark')
ctx.fact_put('user:alice', 'timezone', 'US/Pacific')

# Session-specific facts
ctx.fact_put(f'session:{session_id}', 'last_tool', 'search')

# Global facts
ctx.fact_put('global', 'api_version', '2.0')
```

### Value Serialization

Values are JSON-serialized. Use simple types:

```python
# Good - JSON-serializable
ctx.fact_put('user:x', 'prefs', {'theme': 'dark', 'lang': 'en'})

# Bad - not serializable
ctx.fact_put('user:x', 'callback', lambda x: x)
```

## ReasoningBank Patterns

### Confidence Updates

Bayesian-style multiplicative updates:

| Outcome | Multiplier | Example |
|---------|------------|---------|
| Success | 1.20x | 0.50 -> 0.60 |
| Failure | 0.85x | 0.50 -> 0.425 |

### Strategy Quality

Write generalizable strategies:

```python
# Good - generalizable
ctx.rb_add(
    strategy="Validate API response schema before processing",
    success=True
)

# Bad - too specific
ctx.rb_add(
    strategy="Call get_weather with city='NYC'",
    success=True
)
```

### Anti-Patterns

Track what NOT to do:

```python
ctx.rb_add(
    strategy="Avoid calling deprecated v1 endpoint",
    anti_pattern=True
)
```

## Background Workers

### Worker Pattern

Use `AsyncWorker` for non-blocking operations:

```python
class EvaluationWorker(AsyncWorker):
    async def run(self):
        while True:
            trajectories = await self.store.get_unevaluated()
            for t in trajectories:
                result = await self.judge.evaluate(t)
                await self.store.update_evaluation(t.id, result)
            await asyncio.sleep(self.interval)
```

### Concurrency Control

Use semaphores for external API calls:

```python
self._semaphore = asyncio.Semaphore(max_concurrent)

async def evaluate(self, trajectory):
    async with self._semaphore:
        return await self._judge_llm(trajectory)
```

## Database Operations

### SQLite Best Practices

- Use WAL mode for concurrent reads
- Batch writes when possible
- Index frequently queried columns

```python
conn.execute("PRAGMA journal_mode=WAL")
conn.execute("CREATE INDEX IF NOT EXISTS idx_ns ON facts(namespace)")
```

### Transaction Safety

Current limitation: No explicit transaction management. For critical operations, consider external locking.

## Testing

Use hash-based mock embeddings for determinism:

```python
def mock_embeddings(text: str) -> List[float]:
    h = hashlib.sha256(text.encode()).digest()
    return [b / 255.0 for b in h[:128]]  # 128-dim vector
```

## Reference

- `vel/memory/fact_store.py` - FactStore implementation
- `vel/memory/strategy_reasoningbank.py` - ReasoningBank
- `vel/memory/auto_learning.py` - AutoLearningManager
- ADR-004 for architecture decisions
