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
async def launch_agent(user_input: str, session_id: str, cancel_token: asyncio.Event | None = None):
    """Start agent execution"""
    async for event in agent.run_stream(
        {'message': user_input},
        session_id=session_id,
        cancel_token=cancel_token,
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
cancel_token = asyncio.Event()
task = asyncio.create_task(launch_agent(input, session_id, cancel_token=cancel_token))
# ... later
cancel_token.set()  # Vel closes open blocks, emits abort, then finish
```

For detached harness runs, use `RunManager.cancel(run_id)` instead of cancelling
the asyncio task yourself. `RunManager.cancel()` sets the cooperative token,
waits for the run to close the stream, records status `cancelled`, wakes stream
subscribers, and settles the checkpoint so recovery will not restart it.

Cancelled streams remain well formed: open text/reasoning blocks are closed,
in-flight tools receive `tool-output-error`, any open step receives
`finish-step`, then Vel emits `abort` followed by the terminal `finish`.

## Benefits

- ✓ Streaming enables real-time pause/resume
- ✓ Database persistence enables resume after restart
- ✓ Cooperative cancellation with well-formed terminal streams
- ✓ Session context preserved across interruptions

**See:** [Getting Started - Streaming Mode](../getting-started#streaming-mode)
