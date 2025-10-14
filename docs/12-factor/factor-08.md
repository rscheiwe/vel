---
layout: default
title: "Factor 8: Own Your Control Flow"
parent: 12-Factor Alignment
nav_order: 8
---

# Factor 8: Own Your Control Flow

**Principle:** Maintain explicit control over agent decision-making to prevent unpredictable behavior.

## How Vel Implements This

Vel provides explicit control flow through policies and the reducer pattern:

```python
# Policy-based control
agent = Agent(
    id='my-agent',
    model={'provider': 'openai', 'model': 'gpt-4o'},
    tools=['deploy', 'rollback'],
    policies={
        'max_steps': 5,  # Limit tool calls per run
        'retry': {'attempts': 2}  # Retry failed tools
    }
)

# Explicit state transitions (non-streaming mode)
# The reducer pattern gives full control over state transitions
from vel.core import State, reduce

state = State(run_id='run-123')
event = {'kind': 'llm_step', 'step': {'tool': 'deploy', 'args': {...}}}

# You control the state transition
new_state, effects = reduce(state, event)

for effect in effects:
    if effect.kind == 'call_tool':
        # You decide whether to execute the tool
        if should_execute(effect.payload['tool']):
            result = await execute_tool(effect.payload['tool'], effect.payload['args'])
        else:
            result = {'error': 'Tool execution blocked by policy'}

# Custom control flow in streaming mode
async for event in agent.run_stream({'message': 'Deploy backend'}):
    if event['type'] == 'tool-input-available':
        tool_name = event['toolName']

        # Your custom control flow
        if tool_name == 'deploy' and not is_business_hours():
            # Block deployment
            print("Deployment blocked: outside business hours")
            break
```

## Benefits

- ✓ Policy-based limits prevent runaway execution
- ✓ Explicit state transitions via reducer (non-streaming)
- ✓ Full visibility into agent decisions
- ✓ Easy to add guards and validations

**See:** [API Reference - Agent Policies](../api-reference#constructor)
