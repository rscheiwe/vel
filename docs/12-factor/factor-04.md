---
layout: default
title: "Factor 4: Tools are Structured Outputs"
parent: 12-Factor Alignment
nav_order: 4
---

# Factor 4: Tools are Structured Outputs

**Principle:** Tools are JSON outputs that trigger deterministic code. Clean separation between LLM decisions and code execution.

## How Vel Implements This

Vel treats tool calls as structured outputs with validation:

```python
# Tools are structured outputs with JSON schemas
tool = ToolSpec(
    name='create_deployment',
    input_schema={
        'type': 'object',
        'properties': {
            'environment': {'type': 'string', 'enum': ['dev', 'staging', 'prod']},
            'version': {'type': 'string'}
        },
        'required': ['environment', 'version']
    },
    output_schema={
        'type': 'object',
        'properties': {
            'deployment_id': {'type': 'string'},
            'status': {'type': 'string'}
        },
        'required': ['deployment_id', 'status']
    },
    handler=create_deployment_handler
)

# LLM decides "what", your code controls "how"
async def create_deployment_handler(input: dict, ctx: dict) -> dict:
    # Your deterministic code
    environment = input['environment']
    version = input['version']

    # Complex business logic
    if environment == 'prod':
        # Send to approval queue
        approval_id = await request_approval(environment, version)
        return {'deployment_id': approval_id, 'status': 'pending_approval'}
    else:
        # Direct deployment
        deployment_id = await deploy(environment, version)
        return {'deployment_id': deployment_id, 'status': 'deployed'}
```

## Benefits

- ✓ LLM outputs structured JSON
- ✓ Your code executes deterministically
- ✓ Validation at input and output boundaries
- ✓ Tool execution not limited to atomic functions

**See:** [Tools Documentation](../tools)
