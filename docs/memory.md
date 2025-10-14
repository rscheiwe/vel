---
layout: default
title: Memory System
nav_order: 7
has_children: true
---

# Memory System

Vel's optional memory system provides runtime-owned memory that works without LLM tool calls.

## Vel's Three Memory Systems

Vel has three distinct memory systems that work together:

1. **Message History** - Conversation turns (automatic, managed by ContextManager)
2. **Fact Store** - Long-term structured facts (manual, via MemoryConfig)
3. **Session Persistence** - Where message history is saved (infrastructure layer)

This document focuses on #2 (Fact Store) and ReasoningBank.

## Overview

The memory system includes two complementary components:

- **Fact Store** - Namespaced key-value store for long-term structured data
- **ReasoningBank** - Strategic memory with embeddings for reasoning patterns

## Key Features

- **Runtime-owned** - No LLM tool calls required
- **Opt-in** - Disabled by default, enable via `MemoryConfig`
- **Pre/post execution** - Retrieval before generation, updates after completion
- **Bounded latency** - Synchronous reads (<50ms), async writes
- **Backwards compatible** - Existing agents work unchanged

## Quick Start

```python
from vel.core import ContextManager, MemoryConfig

# Configure memory
mem = MemoryConfig(
    mode="all",                 # "none" | "facts" | "reasoning" | "all"
    db_path=".vel/vel.db",
    rb_top_k=5,
    embeddings_fn=encode_fn     # Required for ReasoningBank
)

# Enable on context manager
ctx = ContextManager()
ctx.set_memory_config(mem)

# Use fact store
ctx.fact_put("user:alice", "theme", "dark")
theme = ctx.fact_get("user:alice", "theme")

# Use ReasoningBank
signature = {"intent": "planning", "domain": "api", "risk": "low"}
advice = ctx.prepare_for_run(signature)  # Get strategy advice
# ... run agent ...
ctx.finalize_outcome(run_success=True)   # Update confidence
```

## Configuration via Environment

```bash
VEL_MEMORY_MODE=all           # none | facts | reasoning | all
VEL_MEMORY_DB=.vel/vel.db
VEL_RB_TOP_K=5
```

## Next Steps

- [Memory Overview](Memory/memory-overview.md) - Detailed introduction
- [Memory Architecture](Memory/memory-architecture.md) - System design
- [ReasoningBank Strategy](Memory/reasoningbank-strategy.md) - Strategic memory
- [Memory Embeddings](Memory/memory-embeddings.md) - Embedding options
- [Enabling ReasoningBank](Memory/enabling-reasoningbank.md) - Step-by-step guide to setup and population
- [ReasoningBank Phase 2 Roadmap](Memory/reasoningbank-phase2-roadmap.md) - Future: Automatic strategy learning
