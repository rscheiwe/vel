---
layout: default
title: "Factor 5: Unify Execution and Business State"
parent: 12-Factor Alignment
nav_order: 5
---

# Factor 5: Unify Execution State and Business State

**Principle:** Integrate AI execution state with business logic for coherent systems.

## How Vel Implements This

Vel stores execution state alongside business state:

```python
# Execution state stored in Postgres
agent = Agent(
    id='deployment-agent',
    model={'provider': 'openai', 'model': 'gpt-4o'},
    session_storage='database'  # Persistent execution state
)

# Tool handlers can access and update business state
async def deployment_handler(input: dict, ctx: dict) -> dict:
    run_id = ctx['run_id']
    session_id = ctx['session_id']

    # Execute deployment
    deployment = await create_deployment(input)

    # Update business database
    await db.deployments.insert({
        'id': deployment['deployment_id'],
        'run_id': run_id,  # Link to agent execution
        'session_id': session_id,
        'status': deployment['status'],
        'created_at': datetime.now()
    })

    return deployment
```

## Benefits

- ✓ Execution history persisted in database
- ✓ Tool context includes run_id and session_id
- ✓ Easy correlation between agent actions and business outcomes
- ✓ Audit trail for compliance

**See:** [Session Management - Database Storage](../sessions#database-storage)
