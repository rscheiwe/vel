---
layout: default
title: Three Memory Systems
parent: Memory System
nav_order: 1
---

# Understanding Vel's Three Memory Systems

Vel has three distinct memory systems that serve different purposes. Understanding when to use each system is critical for building effective agents.

## The Three Systems

| System | Purpose | Managed By | Persistence | When to Use |
|--------|---------|------------|-------------|-------------|
| **Message History** | Conversation turns | ContextManager (automatic) | Transient or Persistent | Always (automatic) |
| **Fact Store** | Structured long-term data | You (manual) | Always persistent (SQLite) | User preferences, domain knowledge |
| **Session Persistence** | Infrastructure for message history | Agent configuration | Memory or Database | When you need history across restarts |

## 1. Message History

**What it is:** The conversation turns between user and assistant.

**Managed by:** `ContextManager` - automatically tracks all messages

**Configuration:**
```python
# Full message history (default)
ctx = ContextManager()

# Limited window (last 20 messages)
ctx = ContextManager(max_history=20)

# No history (stateless)
from vel.core import StatelessContextManager
ctx = StatelessContextManager()
```

**How it works:**
- Agent automatically appends user messages and assistant responses
- Retrieved automatically before each LLM call
- Used to provide conversation context to the LLM

**Key point:** You don't manually add to message history. The Agent runtime does this automatically.

---

## 2. Fact Store

**What it is:** A namespaced key-value store for long-term structured data.

**Managed by:** You - explicitly store and retrieve facts

**Configuration:**
```python
from vel.core import ContextManager, MemoryConfig

# Enable fact store
mem = MemoryConfig(mode="facts", db_path=".vel/vel.db")
ctx = ContextManager()
ctx.set_memory_config(mem)

# Store facts manually
ctx.fact_put("user:alice", "theme", "dark")
ctx.fact_put("user:alice", "expertise", "intermediate")
ctx.fact_put("project:myapp", "tech_stack", ["FastAPI", "React", "PostgreSQL"])

# Retrieve facts manually
theme = ctx.fact_get("user:alice", "theme")
all_facts = ctx.fact_list("user:alice")
```

**Use cases:**
- **User preferences:** Theme, language, communication style
- **Project metadata:** Current project, technologies, deadlines
- **Domain knowledge:** Company facts, API endpoints, business rules
- **Application state:** Feature flags, configuration values

**Key point:** Facts are structured data YOU decide to store. They persist across conversations and even across agent restarts.

**Lifecycle:**
```python
# Session 1 (Monday)
ctx.fact_put("user:alice", "current_project", "inventory-api")

# Session 2 (Tuesday - facts still available)
project = ctx.fact_get("user:alice", "current_project")  # "inventory-api"
```

---

## 3. Session Persistence

**What it is:** WHERE message history gets saved (infrastructure layer).

**Managed by:** Agent configuration parameter

**Configuration:**
```python
from vel import Agent

# Transient: Message history in-memory only (default)
agent = Agent(
    id='my-agent',
    model={'provider': 'openai', 'model': 'gpt-4o'},
    session_persistence='transient'
)

# Persistent: Message history saved to database
agent = Agent(
    id='my-agent',
    model={'provider': 'openai', 'model': 'gpt-4o'},
    session_persistence='persistent'  # Requires PostgreSQL
)
```

**Options:**
- `'transient'` (default): Message history stored in-memory only
  - Fast
  - Lost when process restarts
  - Good for development, short sessions

- `'persistent'`: Message history saved to PostgreSQL
  - Survives restarts
  - Shared across multiple instances
  - Required for production long-running conversations

**Key point:** This controls the infrastructure layer. It doesn't change WHAT gets stored (message history), only WHERE it's stored.

---

## Comparison: Message History vs Fact Store

A common source of confusion is mixing up message history and facts. Here's the key difference:

### Message History (Automatic)

```python
# User sends message
agent.run({"message": "I prefer dark mode"}, session_id="alice")

# Message history now contains:
# [
#   {"role": "user", "content": "I prefer dark mode"},
#   {"role": "assistant", "content": "I'll remember that..."}
# ]

# This happens AUTOMATICALLY - you don't manually add these
```

**Characteristics:**
- Unstructured (raw conversation text)
- Temporary (limited by max_history or session lifecycle)
- Automatic (Agent manages it)
- Purpose: Provide conversation context to LLM

### Fact Store (Manual)

```python
# YOU decide to extract and store structured data
if "prefer dark mode" in user_message:
    ctx.fact_put("user:alice", "theme", "dark")

# Later (even days later, in a new session)
theme = ctx.fact_get("user:alice", "theme")  # "dark"
```

**Characteristics:**
- Structured (key-value pairs)
- Permanent (persists indefinitely)
- Manual (you decide what to store)
- Purpose: Store curated metadata for future use

---

## When to Use Each System

### Use Message History when:
- ✅ You want normal conversation context ("What did I just ask?")
- ✅ You need recent multi-turn context
- ✅ You want automatic management

### Use Fact Store when:
- ✅ You need long-term structured data
- ✅ You want data to persist across sessions
- ✅ You're storing user preferences, project metadata, or domain knowledge
- ✅ You want to programmatically query stored data

### Use Session Persistence when:
- ✅ You need conversations to survive server restarts
- ✅ You have long-running conversations
- ✅ You need multiple servers to share conversation history

---

## Example: Complete Usage

```python
from vel import Agent
from vel.core import ContextManager, MemoryConfig

# 1. Configure message history (sliding window)
ctx = ContextManager(max_history=20)

# 2. Enable fact store
mem = MemoryConfig(mode="facts", db_path=".vel/vel.db")
ctx.set_memory_config(mem)

# 3. Configure session persistence
agent = Agent(
    id='my-agent',
    model={'provider': 'openai', 'model': 'gpt-4o'},
    context_manager=ctx,
    session_persistence='persistent'  # Save message history to DB
)

# 4. Load and use facts before agent run
user_id = "user:alice"
theme = ctx.fact_get(user_id, "theme") or "light"
expertise = ctx.fact_get(user_id, "expertise") or "beginner"

# Build context-aware prompt
prompt = f"User prefers {theme} theme and is {expertise} level. Their question: {{user_question}}"

# 5. Run agent (message history tracked automatically)
response = agent.run({
    "message": prompt.format(user_question="How do I deploy FastAPI?")
}, session_id="alice")

# 6. Extract and store new facts
if "advanced" in response:
    ctx.fact_put(user_id, "expertise", "advanced")
```

---

## Summary Table

| Question | Answer |
|----------|--------|
| How do I store conversation turns? | You don't - Agent does this automatically (message history) |
| How do I store user preferences? | Explicitly with `ctx.fact_put()` (fact store) |
| How do I make conversations survive restarts? | Use `session_persistence='persistent'` |
| How do I limit conversation memory? | Use `ContextManager(max_history=N)` |
| How do I store project metadata? | Explicitly with `ctx.fact_put()` (fact store) |
| Where does fact store data live? | SQLite database (configured via MemoryConfig) |
| Where does message history live? | In-memory or PostgreSQL (configured via session_persistence) |

---

## Next Steps

- [Fact Store Examples](../../examples/memory_examples/fact_store_demo.py) - See fact store in action
- [Memory Architecture](memory-architecture.md) - Technical implementation details
- [ReasoningBank](reasoningbank-strategy.md) - Strategy-level memory (4th type!)
