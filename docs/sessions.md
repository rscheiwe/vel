---
layout: default
title: Session Management
nav_order: 3
---

# Session Management

Complete guide to managing multi-turn conversations in Vel.

## Overview

Sessions enable the agent to remember context across multiple calls, creating natural multi-turn conversations. Without sessions, each call is independent.

## Basic Usage

```python
from vel import Agent

agent = Agent(
    id='my-agent',
    model={'provider': 'openai', 'model': 'gpt-4o'},
    session_storage='memory'  # Default
)

session_id = 'user-123'

# Turn 1
answer1 = await agent.run(
    {'message': 'My name is Alice'},
    session_id=session_id
)

# Turn 2 - remembers Alice
answer2 = await agent.run(
    {'message': 'What is my name?'},
    session_id=session_id
)
```

## Session Storage Modes

### In-Memory Storage (Default)

**Fast**, but sessions are lost when the process restarts.

```python
agent = Agent(
    id='my-agent',
    model={'provider': 'openai', 'model': 'gpt-4o'},
    session_storage='memory'  # Default
)
```

**Characteristics:**
- ✓ Fast (no database overhead)
- ✓ Simple setup (no configuration needed)
- ✗ Not persistent (lost on restart)
- ✗ Not shared across processes

**Best for:** Development, short-lived sessions, single-process deployments

### Database Storage

**Persistent**, survives restarts and can be shared across processes.

```python
agent = Agent(
    id='my-agent',
    model={'provider': 'openai', 'model': 'gpt-4o'},
    session_storage='database'
)
```

**Requires:** `POSTGRES_DSN` environment variable

**Characteristics:**
- ✓ Persistent (survives restarts)
- ✓ Shared across processes
- ✗ Slower (database I/O)
- ✗ Requires Postgres setup

**Best for:** Production, long-lived sessions, multi-process deployments

## Context Manager Modes

Control how much history is retained in a session.

### Full Memory (Default)

Remembers entire conversation history.

```python
from vel import ContextManager

agent = Agent(
    id='my-agent',
    model={'provider': 'openai', 'model': 'gpt-4o'},
    context_manager=ContextManager()  # or omit (default)
)
```

**Use case:** Most conversations, customer support, chatbots

### Stateless (No Memory)

Each call is independent, no history retained.

```python
from vel import StatelessContextManager

agent = Agent(
    id='my-agent',
    model={'provider': 'openai', 'model': 'gpt-4o'},
    context_manager=StatelessContextManager()
)
```

**Use case:** Batch processing, independent queries, stateless APIs

### Limited History

Only keeps last N messages (sliding window).

```python
from vel import ContextManager

agent = Agent(
    id='my-agent',
    model={'provider': 'openai', 'model': 'gpt-4o'},
    context_manager=ContextManager(max_history=10)  # Last 10 messages
)
```

**Use case:** Long conversations, cost optimization, token limit management

## Hybrid Approach

Combine session storage and context manager modes:

```python
# Database + Limited History
# Persistent storage, but only keeps last 20 messages
agent = Agent(
    id='my-agent',
    model={'provider': 'openai', 'model': 'gpt-4o'},
    session_storage='database',
    context_manager=ContextManager(max_history=20)
)
```

## Advanced Usage

### Custom Context Manager

Implement custom memory logic:

```python
from vel import ContextManager

class RAGContextManager(ContextManager):
    """Context manager with RAG support"""

    def messages_for_llm(self, run_id: str, session_id: Optional[str] = None):
        # Get standard messages
        messages = super().messages_for_llm(run_id, session_id)

        # Add RAG-retrieved context
        if session_id:
            retrieved_docs = self.retrieve_relevant_docs(session_id)
            messages.insert(0, {
                'role': 'system',
                'content': f"Relevant context: {retrieved_docs}"
            })

        return messages

    def retrieve_relevant_docs(self, session_id: str):
        # Your RAG logic here
        return "..."

agent = Agent(
    id='my-agent',
    model={'provider': 'openai', 'model': 'gpt-4o'},
    context_manager=RAGContextManager()
)
```

### Session Management Operations

```python
# Get session context
context = agent.ctxmgr.get_session_context('user-123')

# Clear a session
agent.ctxmgr.clear_session('user-123')

# Set session context (e.g., loaded from external source)
messages = [
    {'role': 'user', 'content': 'Hello'},
    {'role': 'assistant', 'content': 'Hi there!'}
]
agent.ctxmgr.set_session_context('user-123', messages)
```

## Streaming with Sessions

Sessions work with both `run()` and `run_stream()`:

```python
session_id = 'user-123'

# Turn 1: streaming
async for event in agent.run_stream(
    {'message': 'My name is Alice'},
    session_id=session_id
):
    print(event)

# Turn 2: non-streaming (context preserved)
answer = await agent.run(
    {'message': 'What is my name?'},
    session_id=session_id
)
```

## Best Practices

### Session ID Strategy

```python
# User-based sessions
session_id = f"user-{user_id}"

# Conversation-based sessions
session_id = f"conv-{conversation_id}"

# Time-based sessions
session_id = f"session-{timestamp}"
```

### Session Cleanup

For in-memory storage, implement cleanup:

```python
# Periodic cleanup
def cleanup_old_sessions():
    # Get sessions older than 1 hour
    old_sessions = get_inactive_sessions(max_age_hours=1)
    for session_id in old_sessions:
        agent.ctxmgr.clear_session(session_id)
```

For database storage, use TTL:

```sql
-- Cleanup sessions older than 7 days
DELETE FROM vel_sessions
WHERE updated_at < NOW() - INTERVAL '7 days';
```

### Memory Management

```python
# Optimize for long conversations
agent = Agent(
    id='my-agent',
    model={'provider': 'openai', 'model': 'gpt-4o'},
    context_manager=ContextManager(
        max_history=50,      # Limit context window
        summarize=False       # Future: enable summarization
    ),
    session_storage='database'
)
```

## Database Schema

The `vel_sessions` table structure:

```sql
CREATE TABLE vel_sessions (
    id TEXT PRIMARY KEY,              -- Session ID
    context JSONB NOT NULL,            -- Message history
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    expires_at TIMESTAMPTZ             -- Optional TTL
);
```

## Troubleshooting

### Sessions Not Persisting

**Problem:** Session context lost between calls.

**Solutions:**
1. Ensure you're using the same `session_id` across calls
2. Check that `session_storage='database'` if you need persistence
3. Verify `POSTGRES_DSN` is configured correctly

### Memory Usage Growing

**Problem:** Memory usage increases over time with many sessions.

**Solutions:**
1. Use `max_history` to limit context window
2. Implement session cleanup
3. Use `session_storage='database'` to offload to Postgres

### Slow Response Times

**Problem:** Responses slow with database storage.

**Solutions:**
1. Use `session_storage='memory'` if persistence not needed
2. Add database indexes on `id` and `updated_at`
3. Use connection pooling in Postgres

## Examples

See `examples/context_modes.py` for complete demonstrations.
