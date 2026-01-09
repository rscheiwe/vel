---
paths:
  - "vel/**/*.py"
description: "Python coding standards for Vel core"
---

# Python Standards for Vel

## Type Hints

- **Required** for all public API functions and methods
- Use `Optional[T]` for nullable parameters
- Use `Union[A, B]` sparingly; prefer protocols or base classes
- Return type hints are mandatory

```python
# Good
async def stream(
    self,
    messages: List[Dict[str, Any]],
    model: str,
    tools: Optional[List[ToolSpec]] = None
) -> AsyncGenerator[StreamEvent, None]:
    ...

# Bad - missing type hints
def process(data):
    ...
```

## Async Patterns

- **Async for I/O**: All network calls, database operations, file I/O
- **Sync for pure logic**: Reducers, validators, formatters
- Use `async for` with generators, not `list()`

```python
# Good
async for event in provider.stream(...):
    yield event

# Bad - blocks event loop
events = list(provider.stream(...))
```

## Error Handling

- Use domain-specific exceptions from `vel/core/`
- Catch specific exceptions, not bare `except:`
- Log errors with context before re-raising

```python
# Good
from vel.core import GuardrailError

try:
    result = await guardrail.check(content)
except GuardrailError as e:
    logger.error(f"Guardrail failed: {e}", extra={'content_length': len(content)})
    raise

# Bad
try:
    ...
except:
    pass
```

## Imports

- Standard library first, then third-party, then local
- Use absolute imports for cross-module references
- Avoid circular imports by importing types in `if TYPE_CHECKING:`

```python
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vel.agent import Agent
```

## Docstrings

- Required for public classes and functions
- Use triple quotes with summary line
- Include Args, Returns, Raises sections for complex functions

```python
def reduce(state: State, event: Dict) -> Tuple[State, List[Effect]]:
    """
    Apply event to state, producing new state and effects.

    Args:
        state: Current execution state
        event: Event to process

    Returns:
        Tuple of (new_state, list_of_effects)
    """
```

## Reference

Follow patterns in `vel/agent.py` as the canonical example.
