---
layout: default
title: "Factor 12: Stateless Reducer"
parent: 12-Factor Alignment
nav_order: 12
---

# Factor 12: Make Agent a Stateless Reducer

**Principle:** Design agents as stateless reducers for predictable, reproducible behavior.

## How Vel Implements This

Vel implements the reducer pattern for non-streaming execution:

```python
# The reducer is a pure function: (State, Event) -> (State, Effects)
from agents.core import State, reduce

# Initial state
state = State(run_id='run-123')

# Event sequence
events = [
    {'kind': 'start'},
    {'kind': 'llm_step', 'step': {'tool': 'get_weather', 'args': {'city': 'NYC'}}},
    {'kind': 'tool_result', 'result': {'temp_f': 65, 'condition': 'cloudy'}},
    {'kind': 'llm_step', 'step': {'done': True, 'answer': 'The weather is cloudy, 65°F'}},
]

# Deterministic state transitions
for event in events:
    state, effects = reduce(state, event)

    # State is immutable, effects are generated
    for effect in effects:
        if effect.kind == 'call_llm':
            # Execute LLM call
            pass
        elif effect.kind == 'call_tool':
            # Execute tool
            pass
        elif effect.kind == 'halt':
            # Agent completed
            break

# Same inputs always produce same outputs
# Easy to test, debug, and reason about
```

## Stateless Execution

```python
# Agent execution is stateless with external state storage
agent = Agent(
    id='my-agent',
    model={'provider': 'openai', 'model': 'gpt-4o'},
    session_storage='database'  # State stored externally
)

# Each call is independent
# State loaded from database
# New state saved after completion
# Agent itself is stateless
```

## Benefits

- ✓ Predictable behavior: same input → same output
- ✓ Easy to test with pure functions
- ✓ Reproducible debugging
- ✓ State stored externally (database)
- ✓ Horizontally scalable (agents are stateless)

**See:** [Architecture - Reducer](../README.md#architecture)
