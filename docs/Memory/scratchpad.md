---
layout: default
title: Scratchpad
parent: Memory System
nav_order: 6
---

# Scratchpad: Ephemeral Working Memory

The Scratchpad provides **ephemeral, in-memory working memory** for agents during multi-step tool execution. Unlike Fact Store (long-term) or ReasoningBank (learned strategies), the Scratchpad lives only for the duration of a single agent run.

---

## Overview

| Aspect | Description |
|--------|-------------|
| **Lifetime** | Single agent run (ephemeral) |
| **Storage** | In-memory (no database) |
| **Purpose** | Working memory during multi-step tasks |
| **Dependencies** | None (pure Python stdlib) |
| **Context Continuity** | Automatic summary injection between runs |

---

## Quick Start

```python
from vel import Agent
from vel.tools.scratchpad import ScratchpadConfig

# Simplest usage
agent = Agent(
    id="my-agent",
    model={"provider": "openai", "model": "gpt-4o"},
    scratchpad=True,  # That's it!
)

# Agent now has scratchpad tools automatically injected
result = await agent.run({"message": "Research quantum computing companies"})
```

When `scratchpad=True`, the agent receives these tools:

| Tool | Purpose |
|------|---------|
| `write_to_scratchpad` | Save any content with a key |
| `read_from_scratchpad` | View all or specific entries |
| `save_plan` | Store execution plan (displayed first) |
| `record_finding` | Auto-numbered research findings |
| `record_observation` | Log tool output observations |
| `search_scratchpad` | Search entries by keyword |
| `checkpoint_scratchpad` | Create state snapshots |
| `list_scratchpad_checkpoints` | List all checkpoints |

---

## Configuration

### Using ScratchpadConfig

```python
from vel.tools.scratchpad import ScratchpadConfig

agent = Agent(
    id="researcher",
    model={"provider": "openai", "model": "gpt-4o"},
    scratchpad=ScratchpadConfig(
        max_entries=50,           # Max entries before eviction (default: 100)
        max_content_length=10000, # Max chars per entry (default: 50000)
        summary_max_chars=800,    # Summary length for next run (default: 500)
        include_search=True,      # Include search tool (default: True)
        include_checkpoint=True,  # Include checkpoint tools (default: True)
    ),
)
```

### Configuration Options

| Option | Default | Range | Description |
|--------|---------|-------|-------------|
| `max_entries` | 100 | 10-500 | Maximum entries before oldest are evicted |
| `max_content_length` | 50000 | 1000-100000 | Max characters per entry |
| `summary_max_chars` | 500 | 100-2000 | Summary length for context injection |
| `include_search` | True | - | Include `search_scratchpad` tool |
| `include_checkpoint` | True | - | Include checkpoint tools |

---

## Lifecycle

```
agent.run("message 1")
       │
       ▼
┌──────────────────┐
│ Create Scratchpad│
│ Inject prev summary (if exists)
└────────┬─────────┘
         │
         ▼
┌──────────────────────────────────────┐
│         Agent Execution               │
│  • LLM calls scratchpad tools        │
│  • save_plan("1. Search 2. Analyze") │
│  • record_finding("OAuth required")  │
│  • read_from_scratchpad()            │
└────────┬─────────────────────────────┘
         │
         ▼
┌──────────────────┐
│  Generate Summary │──► Stored in Agent._scratchpad_summary
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Discard Scratchpad│
└────────┬─────────┘
         │
         ▼
    Return result


agent.run("message 2")  ← Previous summary automatically injected
```

---

## Multi-Run Context Continuity

The key feature of the Scratchpad is **automatic summary injection** between runs:

```python
agent = Agent(
    model={"provider": "openai", "model": "gpt-4o"},
    scratchpad=True,
)

# Run 1: Initial research
result1 = await agent.run({"message": "Research quantum computing companies"})
# Agent uses scratchpad tools, summary captured at end

# Run 2: Follow-up (previous summary injected automatically)
result2 = await agent.run({"message": "Compare their funding rounds"})
# System prompt includes: [Previous Run Context] ...

# Run 3: Continue building
result3 = await agent.run({"message": "Which one should I invest in?"})
# Summary from run 2 injected (replaces run 1 summary)

# Start fresh conversation
agent.clear_scratchpad_context()
result4 = await agent.run({"message": "What's the weather?"})
# No previous context
```

---

## Entry Types

The Scratchpad uses semantic entry types for organization:

| Type | Icon | Purpose | Eviction |
|------|------|---------|----------|
| `PLAN` | 📍 | Execution plans | Protected |
| `FINDING` | 📊 | Research discoveries | Evictable |
| `OBSERVATION` | 👁 | Tool output notes | Evictable |
| `REASONING` | 💭 | Chain-of-thought | Evictable |
| `ERROR` | ❌ | Error logs | Protected |
| `CHECKPOINT` | 💾 | State snapshots | Protected |
| `NOTE` | 📝 | General notes | Evictable |
| `CONTEXT` | 🔍 | Context markers | Evictable |

**Protected** entries are never evicted when the scratchpad reaches capacity.

---

## Formatted Output

When the agent reads the scratchpad, it sees a formatted view:

```
📋 SCRATCHPAD CONTENTS
━━━━━━━━━━━━━━━━━━━━━━

📍 CURRENT PLAN:
  1. Search for API documentation
  2. Extract authentication patterns
  3. Implement OAuth flow

📊 FINDINGS (3):
  #1 [from: api_docs] OAuth 2.0 is required
  #2 [from: github] Rate limit is 5000/hour
  #3 [from: testing] Refresh tokens expire in 7 days

💭 REASONING:
  [step 1] User needs API integration
  [step 2] Authentication is the first barrier

📝 NOTES:
  • user_preference: Prefers Python examples
```

---

## Standalone Usage

You can use the Scratchpad class directly without an Agent:

```python
from vel.tools.scratchpad import Scratchpad, EntryType

sp = Scratchpad()

# Write operations
sp.set_plan("1. Research\n2. Analyze\n3. Recommend")
sp.add_finding("Market growing 25% YoY", source="report")
sp.add_finding("Competitor has 40% share", source="analysis")
sp.add_observation("API returned 150 results", tool_name="search")
sp.add_reasoning("Growth rate suggests opportunity")
sp.log_error("Rate limit hit", context="API call")

# Read all
print(sp.read())

# Read specific entry
print(sp.read("_plan"))

# Search
results = sp.search("market", entry_types=[EntryType.FINDING])

# Generate summary
summary = sp.get_summary(max_chars=500)

# Checkpoint
sp.checkpoint("before_analysis")

# Statistics
stats = sp.get_stats()
print(f"Total entries: {stats.total_entries}")
print(f"By type: {stats.entries_by_type}")

# Serialization
data = sp.to_dict()
restored = Scratchpad.from_dict(data)
```

---

## Summary Generation

The summary prioritizes information in this order:

1. **Errors** (highest priority, never truncated)
2. **Plan** (truncated to ~100 chars)
3. **Recent Findings** (last 5, with source attribution)
4. **Status** (entry count)

Example summary output:

```
[Previous Run Context]
Plan: 1. Research competitors 2. Analyze market 3. Make recommendation...
Findings (3):
  - Market growing 25% YoY [from: report]
  - Competitor has 40% share [from: analysis]
  - New regulations Q3 [from: legal]
Status: 8 entries total
```

---

## Thread Safety

All Scratchpad operations are thread-safe using `threading.RLock()`. This handles concurrent tool calls within a single agent run.

---

## Comparison with Other Memory Systems

| Feature | Scratchpad | Fact Store | ReasoningBank |
|---------|------------|------------|---------------|
| **Lifetime** | Single run | Persistent | Persistent |
| **Storage** | In-memory | SQLite | SQLite + Embeddings |
| **Purpose** | Working memory | Long-term facts | Learned strategies |
| **LLM Access** | Via tools | Runtime-injected | Runtime-injected |
| **Use Case** | Multi-step tasks | User preferences | Self-improvement |

---

## Best Practices

### 1. Guide the Agent

Add scratchpad usage guidance to your system prompt:

```python
prompt = PromptTemplate(
    system="""You have access to a scratchpad for working memory.

Best practices:
1. Always start complex tasks with save_plan()
2. Record findings immediately after each tool call
3. Review your scratchpad before synthesizing answers
"""
)

agent = Agent(
    model={"provider": "openai", "model": "gpt-4o"},
    prompt=prompt,
    scratchpad=True,
)
```

### 2. Clear Context Appropriately

```python
# Clear when topic changes
agent.clear_scratchpad_context()

# Clear when user explicitly starts new conversation
if is_new_conversation:
    agent.clear_scratchpad_context()
```

### 3. Use Compact Config for Simple Tasks

```python
# For simple tasks, reduce overhead
agent = Agent(
    model={"provider": "openai", "model": "gpt-4o"},
    scratchpad=ScratchpadConfig(
        max_entries=25,
        include_checkpoint=False,  # Fewer tools
    ),
)
```

---

## Example: Research Agent

```python
import asyncio
from vel import Agent
from vel.tools import ToolSpec
from vel.tools.scratchpad import ScratchpadConfig

def search_web(query: str) -> str:
    """Search the web for information."""
    # Your search implementation
    return f"Results for: {query}"

def read_document(url: str) -> str:
    """Read a document from URL."""
    # Your document reading implementation
    return f"Content from: {url}"

async def main():
    agent = Agent(
        id="research-agent",
        model={"provider": "openai", "model": "gpt-4o"},
        tools=[
            ToolSpec.from_function(search_web),
            ToolSpec.from_function(read_document),
        ],
        scratchpad=ScratchpadConfig(
            summary_max_chars=800,  # More context between runs
        ),
    )

    # Multi-step research with context continuity
    await agent.run({"message": "Research the history of AI"})
    await agent.run({"message": "Focus on the transformer architecture"})
    await agent.run({"message": "Summarize the key breakthroughs"})

asyncio.run(main())
```

---

## See Also

- [Memory Overview](memory-overview.md) - All memory systems
- [Fact Store](three-memory-systems.md) - Long-term key-value storage
- [ReasoningBank](reasoningbank-strategy.md) - Strategy memory
- [Auto-Learning Guide](auto-learning-guide.md) - Self-evolving agents
