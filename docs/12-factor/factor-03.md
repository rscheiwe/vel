---
layout: default
title: "Factor 3: Own Your Context Window"
parent: 12-Factor Alignment
nav_order: 3
---

# Factor 3: Own Your Context Window

**Principle:** Everything is context engineering. Structure information to maximize LLM understanding.

## How Vel Implements This

Vel gives you complete control over context window management:

```python
# Limit context window to control token usage
agent = Agent(
    id='my-agent',
    model={'provider': 'openai', 'model': 'gpt-4o'},
    context_manager=ContextManager(max_history=20)  # Sliding window
)

# Stateless context (no history)
agent = Agent(
    id='my-agent',
    model={'provider': 'openai', 'model': 'gpt-4o'},
    context_manager=StatelessContextManager()  # Each call independent
)

# Custom context structuring
class RAGContextManager(ContextManager):
    def messages_for_llm(self, run_id: str, session_id: Optional[str] = None):
        messages = super().messages_for_llm(run_id, session_id)

        # Compress old messages into summary
        if len(messages) > 10:
            old_messages = messages[:8]
            summary = self.summarize(old_messages)
            messages = [
                {'role': 'system', 'content': f"Previous conversation: {summary}"}
            ] + messages[8:]

        return messages
```

## Context Compaction

Use built-in compaction strategies:

```python
from agents import ContextCompactor

# Sliding window
compacted = ContextCompactor.sliding_window(
    messages,
    max_messages=10,
    preserve_system=True
)

# Summarize old messages
compacted = ContextCompactor.summarize_old_messages(
    messages,
    threshold=10
)

# Truncate long messages
compacted = ContextCompactor.truncate_long_messages(
    messages,
    max_length=500
)
```

## Benefits

- ✓ Configurable memory: full/stateless/limited
- ✓ Custom context managers for advanced use cases
- ✓ Direct access to message history
- ✓ Token optimization through context management

**See:** [Session Management - Context Manager Modes](../sessions#context-manager-modes)
