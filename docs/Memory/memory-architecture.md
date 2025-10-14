---
layout: default
title: Memory Architecture
parent: Memory System
nav_order: 2
---

# Vel Memory Architecture

Vel’s memory system is modular and **runtime-owned**, meaning it operates entirely inside the agent runtime rather than through LLM tool calls.  
It allows agents to recall facts, reasoning strategies, and user-specific context **without increasing latency** or changing the prompt architecture.

---

## 🧩 Overview

Vel defines three complementary memory components:

| Component        | Type       | Scope             | Purpose                                  |
| ---------------- | ---------- | ----------------- | ---------------------------------------- |
| `ContextManager` | short-term | per run / session | Maintains current conversation state     |
| `Fact Store` | mid-term   | persistent        | Stores facts, summaries, and user data   |
| `ReasoningBank`  | long-term  | persistent        | Stores learned strategies and heuristics |

These can be mixed and matched via the `MemoryConfig` object.

---

## 🧠 Conceptual Layers

```


      ┌────────────────────────────┐
      │       LLM Runtime          │
      │    (Agent Execution)       │
      └────────────┬───────────────┘
                   │

┌─────────────────┴─────────────────┐
│ Context Layer │
│ (Conversation & In-Run Context) │
│ → ContextManager │
└─────────────────┬─────────────────┘
│
┌──────────────┴──────────────┐
│ Memory Layer │
│ Fact Store + ReasoningBank
│ (Optional Runtime Boosts) │
└──────────────┬──────────────┘
│
┌───────────┴────────────┐
│ SQLite / Vector DB │
│ (Memory persistence) │
└────────────────────────┘

```

---

## ⚙️ Runtime Flow

### 1. **Pre-Run Initialization**

When a new agent run starts:

1. The `ContextManager` loads any session or run-specific history.
2. If enabled:
   - `Fact Store` retrieves relevant key-value items (facts, preferences, summaries).
   - `ReasoningBank` retrieves top-K similar strategies based on task signature embeddings.
3. The combined context is injected into the **system prompt** before model generation.

---

### 2. **Streaming and Execution**

- The LLM streams tokens as usual.
- The runtime decides when to stop streaming (no open memory tools or callbacks required).
- The agent executes tools and gathers results via the normal `ContextManager` interface.

> 🧭 Key difference from “tool-based” memory systems:  
> The LLM never calls a `memory_read` or `memory_write` tool — all memory operations happen **before** and **after** model inference.

---

### 3. **Post-Run Finalization**

After the response is completed:

1. The `ContextManager` appends messages to session memory.
2. If enabled:
   - `Fact Store` may store summaries, outcomes, or key facts.
   - `ReasoningBank` evaluates whether the strategy was successful and updates its vector database accordingly.
3. Updates are handled asynchronously in background threads to avoid blocking latency.

---

## 🧩 Component Details

### 🟦 1. ContextManager

- Maintains transient conversation state (per-run or per-session).
- Handles inputs, outputs, and message trimming (`max_history`).
- Stateless variants can disable all retention.

```python
ctx = ContextManager(max_history=5)
ctx.set_input(run_id, {"message": "Plan a FastAPI project"})
ctx.append_assistant_message(run_id, "Let's start with project scaffolding.")
```

---

### 🟨 2. Fact Store

- Namespaced key-value store for long-term structured data
- Ideal for user preferences, project metadata, and domain knowledge
- Implemented with SQLite under the hood

```python
ctx.fact_put("user:richard", "theme", "dark")
theme = ctx.fact_get("user:richard", "theme")
```

The fact store is **structured** — it stores curated metadata you explicitly save.

---

### 🟥 3. ReasoningBank

- Strategic memory system that recalls _how to think_, not _what to think_.
- Each entry stores a distilled heuristic plus vector embeddings for the context.

```python
advice = ctx.reasoningbank.get_advice(signature)
print(advice)
# -> ["Clarify user intent before planning.", "Avoid replanning mid-stream."]
```

It’s powered by embeddings for conceptual similarity:

- **vector_strategy**: embedding of heuristic text
- **vector_signature**: embedding of contextual metadata

---

## 🔁 Control Flow Diagram

```
┌─────────────────────────────┐
│   Agent Runtime (Vel)       │
└────────────┬────────────────┘
             │
             ▼
    ┌──────────────────┐
    │ ContextManager   │
    │ Load session/run │
    └────────┬─────────┘
             │
     ┌───────┴───────────────┐
     │  MemoryConfig Switch  │
     │  mode = facts/both │
     └───────┬───────────────┘
             │
   ┌─────────┴─────────┐
   │  Fact Store   │
   │  (facts & kv)     │
   ├────────────────────┤
   │  ReasoningBank     │
   │  (strategy recall) │
   └─────────┬─────────┘
             │
             ▼
    ┌──────────────────┐
    │ SQLite / VectorDB │
    └──────────────────┘
```

---

## ⚡ Runtime Behavior

| Stage     | Component      | Operation                        | Timing      |
| --------- | -------------- | -------------------------------- | ----------- |
| Run start | ReasoningBank  | Retrieve top-K strategies        | synchronous |
| Run start | Fact Store | Retrieve contextual facts        | synchronous |
| Run loop  | ContextManager | Manage messages, tools           | synchronous |
| Run end   | ReasoningBank  | Update confidence, anti-patterns | async       |
| Run end   | Fact Store | Store new facts or summaries     | async       |

---

## 🔐 Design Goals

1. **Zero LLM dependency**

   - No memory tools or external calls.

2. **Predictable latency**

   - Pre-load and post-update, never mid-stream.

3. **Composable configuration**

   - Enable memory per run or globally.

4. **Embeddings optional**

   - Works even with simple string similarity if desired.

5. **Persistent but lightweight**

   - Uses SQLite for simplicity; can scale to vector DBs.

---

## ⚙️ Typical Configuration

```python
from vel.core.context import MemoryConfig

mem = MemoryConfig(
    mode="all",             # "none" | "facts" | "reasoning" | "all"
    db_path=".vel/vel.db",
    rb_top_k=5,
    embeddings_fn=encode     # custom or model-based
)
```

The `MemoryConfig` object wires together all components dynamically at runtime.

---

## 🧩 Summary

| Component      | Memory Type    | Purpose                     | Retention  | Example                   |
| -------------- | -------------- | --------------------------- | ---------- | ------------------------- |
| ContextManager | Working memory | Holds conversation state    | transient  | "This run"                |
| Fact Store | Declarative    | Stores facts, summaries     | persistent | "User prefers dark theme" |
| ReasoningBank  | Procedural     | Stores reasoning heuristics | persistent | "Clarify before planning" |

Together, they form a **cognitive architecture**:

- **ContextManager** = short-term “working memory”
- **Fact Store** = mid-term “fact memory”
- **ReasoningBank** = long-term “strategy memory”

> 🧠 In short: Vel agents don’t just _remember what they said_ — they _learn how to think_.
