# ADR-002: Stateless Reducer Pattern

**Status:** Accepted
**Date:** 2025-01-08
**Decision Makers:** Vel Core Team

---

## Context

Agent orchestration requires managing complex state transitions:
- Multi-step tool execution loops
- Message history accumulation
- Error handling and recovery
- Streaming event emission

Traditional approaches (mutable state, class hierarchies) make testing difficult and side effects unpredictable.

## Decision

Adopt a **pure reducer pattern** inspired by functional programming:

```python
def reduce(state: State, event: Dict) -> Tuple[State, List[Effect]]
```

### Core Principle

`(State, Event) -> (State, Effects)`

- **State**: Immutable snapshot of agent execution
- **Event**: Something that happened (user input, LLM response, tool result)
- **Effects**: Immutable commands describing side effects

### Effect Types

```python
Effect = Literal[
    'emit',       # Yield stream event
    'call_tool',  # Execute tool
    'call_llm',   # Call language model
    'checkpoint', # Save state
    'halt'        # Stop execution
]
```

### Implementation

The reducer in `vel/core/reducer.py` demonstrates the pattern. Most orchestration logic lives in `vel/agent.py` for pragmatic reasons, but follows reducer principles:

1. State transitions are explicit
2. Side effects are commands, not immediate actions
3. Each step is independently testable

## Consequences

### Positive

1. **12-Factor Alignment**: Stateless design matches 12-Factor Agent principles
2. **Testability**: Pure functions with predictable inputs/outputs
3. **Debugging**: State transitions are traceable and reproducible
4. **Replay**: Execution can be replayed from any checkpoint

### Negative

1. **Learning Curve**: Developers unfamiliar with reducers need onboarding
2. **Verbosity**: Some operations require more boilerplate
3. **Pragmatic Tradeoffs**: Full purity sacrificed for practical implementation

## Alternatives Considered

1. **Mutable State Machine**: Rejected due to testing complexity
2. **Actor Model**: Rejected as over-engineering for single-agent use case
3. **Full Event Sourcing**: Rejected due to storage overhead

## References

- [12-Factor Agents](https://github.com/humanlayer/12-factor-agents)
- `vel/core/reducer.py` - Reducer implementation
- `vel/agent.py` - Agent orchestrator
