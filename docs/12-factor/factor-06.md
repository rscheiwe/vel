---
layout: default
title: "Factor 6: Launch/Pause/Resume"
parent: 12-Factor Alignment
nav_order: 6
---

# Factor 6: Launch/Pause/Resume with Simple APIs

**Principle:** Provide flexible control over agent workflows with simple, composable APIs.

## How Vel Implements This

Vel provides dual execution modes with async control:

```python
import asyncio
from vel import Agent

agent = Agent(
    id='my-agent',
    model={'provider': 'openai', 'model': 'gpt-4o'},
    session_storage='database'  # Persist state for resume
)

# Launch
async def launch_agent(user_input: str, session_id: str):
    """Start agent execution"""
    async for event in agent.run_stream(
        {'message': user_input},
        session_id=session_id
    ):
        if event['type'] == 'tool-input-available':
            # Pause for approval if needed
            if event['toolName'] == 'deploy_to_prod':
                await request_approval(event, session_id)
                raise PauseExecution(event)

        yield event

# Resume
async def resume_agent(approval: dict, session_id: str):
    """Resume after approval"""
    # Load previous context from database
    context = agent.ctxmgr.get_session_context(session_id)

    # Add approval to context
    context.append({
        'role': 'user',
        'content': f"Approval granted: {approval}"
    })
    agent.ctxmgr.set_session_context(session_id, context)

    # Continue execution
    async for event in agent.run_stream(
        {'message': 'Continue with approved deployment'},
        session_id=session_id
    ):
        yield event

# Cancellation support
task = asyncio.create_task(launch_agent(input, session_id))
# ... later
task.cancel()  # Graceful cancellation
```

## Benefits

- ✓ Streaming enables real-time pause/resume
- ✓ Database persistence enables resume after restart
- ✓ AsyncIO cancellation support
- ✓ Session context preserved across interruptions

**See:** [Getting Started - Streaming Mode](../getting-started#streaming-mode)
