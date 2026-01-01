---
layout: default
title: Memory Overview
parent: Memory System
nav_order: 1
---

# Vel Memory System (Optional Runtime Boosts)

Vel's memory system provides **optional** memory components for different use cases:

| Component | Lifetime | Purpose |
|-----------|----------|---------|
| **Scratchpad** | Single run | Ephemeral working memory for multi-step tasks |
| **Fact Store** | Persistent | Namespaced key-value store for long-term structured data |
| **ReasoningBank** | Persistent | Strategy memory for distilled "do/avoid" heuristics |

All memory modules are **off by default**.

---

## What's New: Phase 2 Auto-Learning ✅

ReasoningBank now supports **automatic self-evolution**. Agents can learn from experience without human intervention:

| Component | Description |
|-----------|-------------|
| **TrajectoryStore** | Records agent execution traces |
| **LLM-as-Judge** | Automatic success/failure evaluation |
| **StrategyExtractor** | Distills strategies from successful runs |
| **MemoryConsolidator** | Merges similar strategies, prevents bloat |

**Getting Started:** See the [Auto-Learning Guide](auto-learning-guide.md) for a step-by-step walkthrough, or run the [demo](https://github.com/rscheiwe/vel/blob/main/examples/memory_examples/auto_learning_demo.py) to see it in action.

---

## 1. Why Runtime-Owned Memory?

Traditional memory tool usage requires the LLM to decide when to read or write, which can:

- Add latency due to additional tool calls
- Keep the generation stream open longer than necessary
- Introduce uncertainty about when to persist updates

Vel's design makes **the runtime** — not the LLM — responsible for memory. All reads happen **before** generation, and all writes happen **after** the stream closes.

---

## 2. Memory Modes

| Mode         | Description                                    |
| ------------ | ---------------------------------------------- |
| `none`       | Default. No memory.                            |
| `facts`      | Enables fact store only (namespaced KV).       |
| `reasoning`  | Enables ReasoningBank strategy memory only.    |
| `all`        | Enables both fact store and ReasoningBank.     |

---

## 3. Quick Example

```python
from vel.core.context import MemoryConfig
from vel.prompts.context_manager import vel_context

def encode(texts):
    import numpy as np, hashlib
    out = []
    for t in texts:
        h = hashlib.sha256(t.encode()).digest()
        v = np.frombuffer(h, dtype=np.uint8).astype(np.float32)[:128]
        v = (v - v.mean()) / (v.std() + 1e-8)
        out.append(v)
    return np.vstack(out)

mem = MemoryConfig(
    mode="all",                # "none" | "facts" | "reasoning" | "all"
    db_path=".vel/vel.db",
    rb_top_k=5,
    embeddings_fn=encode
)

signature = {"intent": "planning", "domain": "fastapi", "risk": "low"}

with vel_context(run_cfg, signature, memory_cfg=mem):
    # run your agent as usual
    pass
```

---

## 4. Fact Store

### What It Is

A lightweight, SQLite-backed **namespaced key-value store** for long-term structured data.
Useful for user preferences, project metadata, and domain knowledge.

### API

```python
ctx.fact_put("user:richard", "theme", "dark")
ctx.fact_get("user:richard", "theme")  # "dark"
ctx.fact_list("user:richard")
```

### When To Use It

- Retaining session or user preferences.
- Storing summaries or short-term conclusions.
- Holding key facts between runs.

---

## 5. ReasoningBank Memory

### What It Is

A **strategy memory** that stores distilled heuristics and “anti-patterns” discovered from prior runs.
Each item is a short sentence like:

> “Summarize the user’s goal before making a plan.”

The ReasoningBank can then recommend top strategies for similar contexts based on embeddings.

### Key Concepts

| Field           | Description                                                       |
| --------------- | ----------------------------------------------------------------- |
| `strategy_text` | One-sentence heuristic or tactic.                                 |
| `anti_patterns` | Optional list of "things to avoid."                               |
| `confidence`    | Numeric score (0.0-1.0) that adjusts based on success/failure.    |
| `signature`     | JSON signature describing the situation (intent, domain, etc.).   |

### Retrieval

At run start:

```python
advice = ctx.prepare_for_run(signature)
if advice:
    system_prompt += "\n\n" + advice
```

### Update

At run end:

```python
ctx.finalize_outcome(run_success=True)
```

(Internally this updates confidence and anti-patterns asynchronously.)

---

## 6. Scratchpad (Working Memory)

The **Scratchpad** is ephemeral working memory for multi-step agent execution. Unlike Fact Store and ReasoningBank (which persist), the Scratchpad lives only for a single agent run.

### Quick Start

```python
from vel import Agent

agent = Agent(
    id="researcher",
    model={"provider": "openai", "model": "gpt-4o"},
    scratchpad=True,  # Enable with defaults
)

# Agent automatically gets scratchpad tools:
# save_plan, record_finding, record_observation, read_from_scratchpad, etc.
result = await agent.run({"message": "Research quantum computing"})
```

### Key Features

- **Automatic summary injection**: Summary from run N is injected into run N+1
- **Structured entry types**: Plan, Finding, Observation, Reasoning, Error
- **Thread-safe**: Handles concurrent tool calls
- **Zero dependencies**: Pure Python stdlib

See the [Scratchpad Guide](scratchpad.md) for full documentation.

---

## 7. When To Use Each

| Use Case                                              | Recommended Memory |
| ----------------------------------------------------- | ------------------ |
| Multi-step tool execution within a run                | Scratchpad         |
| Research tasks with findings accumulation             | Scratchpad         |
| User preferences                                      | Fact Store         |
| Project metadata                                      | Fact Store         |
| Domain knowledge                                      | Fact Store         |
| Behavioral heuristics ("always clarify intent first") | ReasoningBank      |
| Avoiding repeated mistakes                            | ReasoningBank      |
| Both contextual and strategic adaptation              | Fact Store + ReasoningBank |

---

## 8. Configuration via Environment Variables

```bash
VEL_MEMORY_MODE=all           # none | facts | reasoning | all
VEL_MEMORY_DB=.vel/vel.db
VEL_RB_TOP_K=5
```

`embeddings_fn` must still be provided in code (cannot be set via environment variable).

---

## 9. Performance Notes

- Retrieval is bounded and synchronous (typically <40ms).
- Post-run updates happen in a background thread.
- If the queue is full, updates are dropped to protect latency.
- Top-K retrieval is configurable per run (default 5).

---

## 10. Safety Guidelines

- ReasoningBank advice should be treated as **system-only**, not user-facing text.
- Limit advice injection to 3–5 items to avoid prompt bloat.
- Periodically prune or decay confidence to retire stale heuristics.

---

## 11. Summary

Vel’s memory extensions give your agent:

- **Predictable latency**
- **Runtime control**
- **Optional strategic recall**
