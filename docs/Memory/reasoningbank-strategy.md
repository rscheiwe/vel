---
layout: default
title: ReasoningBank Strategy
parent: Memory System
nav_order: 3
---

# ReasoningBank: Strategy Memory in Vel

ReasoningBank is Vel’s **strategic memory layer**, designed to help agents improve reasoning over time by recalling _how_ they’ve solved similar problems in the past.  
It stores distilled heuristics and anti-patterns, indexed by **embeddings**, to make retrieval semantic rather than literal.

---

## 🧩 Concept Overview

ReasoningBank treats each successful (or failed) task as a **learning event**.

Each run yields:

- **Strategy text** – A short heuristic such as  
  _“Summarize user intent before planning.”_
- **Anti-patterns** – Optional “things to avoid,” e.g.  
  _“Do not replan mid-stream.”_
- **Signature** – Structured metadata describing the task context, such as  
  `{"intent": "planning", "domain": "fastapi", "risk": "low"}`
- **Outcome** – Whether the strategy succeeded.

These items are stored in a small local database (SQLite by default) and indexed by vector embeddings.

---

## 🔍 Why Embeddings Matter

Embeddings give ReasoningBank a **sense of similarity** between tasks.  
They make recall _semantic_ rather than _exact_.

### Without embeddings

Retrieval would rely on literal key matches —  
e.g., `"intent": "planning"` must appear exactly the same for a strategy to be found.

That means:

- `"build API"` and `"create endpoint"` would not match.
- `"analyze report"` and `"summarize document"` would appear unrelated.

### With embeddings

Embeddings represent each text (strategy or signature) as a **vector** in high-dimensional space.  
Similar meanings produce nearby vectors.

This lets ReasoningBank retrieve strategies that _feel conceptually related_ even when the words differ.

> Example:  
> A strategy learned from “Summarize a document” can help “Condense a report” because their embeddings are close.

---

## ⚙️ How It Works Internally

### 1. Insertion (Learning)

When a run completes successfully:

1. The runtime creates embeddings for both the **strategy text** and the **task signature**.
2. It stores:
   - `strategy_text`, `anti_patterns`, `signature`, `confidence`
   - `vector_strategy`, `vector_signature`
3. The confidence score increases slightly with each success.

### 2. Retrieval (Advice)

When a new task begins:

1. The runtime encodes the current task signature using the same embedding function.
2. It computes **cosine similarity** between the new vector and stored signature vectors.
3. The top-K most similar strategies are retrieved and injected as _“Strategy Advice”_ at the start of the LLM prompt.

This happens synchronously and usually completes in under 50 ms.

### 3. Updating (Outcome)

After the run:

- The system adjusts confidence scores asynchronously.
- Failed runs lower confidence or record anti-pattern notes.

All of this happens without LLM tool calls — memory is fully runtime-owned.

---

## 🧠 The Role of `embeddings_fn`

Vel doesn’t impose a specific embedding model — you provide one.

It just needs a callable:

```python
embeddings_fn: Callable[[List[str]], np.ndarray]
```

that takes a list of texts and returns a NumPy array of vectors.

### Example 1 — Minimal Hash-Based Embeddings

A deterministic option for local use (no dependencies):

```python
def encode(texts):
    import numpy as np, hashlib
    out = []
    for t in texts:
        h = hashlib.sha256(t.encode()).digest()
        v = np.frombuffer(h, dtype=np.uint8).astype(np.float32)
        v = (v - v.mean()) / (v.std() + 1e-8)
        out.append(v[:256])
    return np.vstack(out)
```

### Example 2 — SentenceTransformer Embeddings

For semantic understanding:

```python
from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer("all-MiniLM-L6-v2")

def encode(texts):
    return np.array(model.encode(texts, normalize_embeddings=True), dtype=np.float32)
```

---

## 📊 Similarity Metric

ReasoningBank uses **cosine similarity** to rank stored strategies:

[
\text{similarity}(A, B) = \frac{A \cdot B}{||A|| , ||B||}
]

Top-K items (default `k=5`) are returned as recommendations.
They can optionally be prefixed to your LLM’s system prompt:

```
Strategy Advice:
1. Clarify the user’s intent before executing.
2. Avoid re-evaluating after completion.
```

---

## 🧱 Database Schema (Actual Implementation)

**rb_strategies table:**

| Column          | Type      | Description                         |
| --------------- | --------- | ----------------------------------- |
| `id`            | INTEGER   | Primary key                         |
| `signature_json`| TEXT      | Context metadata (JSON)             |
| `strategy_text` | TEXT      | The heuristic itself                |
| `anti_patterns` | TEXT      | JSON list of "avoid" statements     |
| `evidence_refs` | TEXT      | JSON list of run IDs                |
| `confidence`    | REAL      | Strength of belief (0.0-1.0)        |
| `created_at`    | REAL      | Unix timestamp                      |
| `updated_at`    | REAL      | Unix timestamp                      |

**rb_embeddings table:**

| Column        | Type    | Description                                    |
| ------------- | ------- | ---------------------------------------------- |
| `strategy_id` | INTEGER | Foreign key to rb_strategies(id)               |
| `embedding`   | BLOB    | Combined embedding (signature + strategy text) |
| `dim`         | INTEGER | Embedding dimensionality                       |

Note: Unlike the paper's two separate embeddings, Vel combines signature and strategy text into one embedding vector for efficiency.

All vectors are stored as serialized NumPy arrays (`float32`).

---

## 🚀 Performance and Scaling

| Operation | Description                                         | Typical latency  |
| --------- | --------------------------------------------------- | ---------------- |
| Retrieval | Embedding + cosine similarity across ≤1k strategies | 20–50 ms         |
| Update    | Background async write (confidence + anti-patterns) | <1 ms per record |

For larger-scale deployments, ReasoningBank can easily migrate to:

- FAISS
- Qdrant
- SQLite + vector extension

The interface remains the same.

---

## 🧩 When to Use ReasoningBank

| Scenario                                      | Benefit                              |
| --------------------------------------------- | ------------------------------------ |
| Repeated reasoning tasks (analysis, planning) | Reuses heuristics for similar tasks  |
| Multi-session agents                          | Learns to “think” better over time   |
| Specialized domains                           | Adapts domain-specific strategies    |
| Rapid prototyping                             | Improves behavior without retraining |

---

## ⚠️ Best Practices

- Limit `rb_top_k` to 3–5 to keep prompts lean.
- Manually prune low-confidence strategies periodically (no automatic decay implemented).
- Never inject strategy advice directly into user-visible text (system-only).
- Keep your embedding function stable — changing models resets similarity baselines.
- Provide clear success/failure signals for accurate confidence updates.

---

## 🧩 Summary

**ReasoningBank** provides _strategic_ rather than _episodic_ memory.
By embedding past strategies and task signatures, it lets your agent recall **what reasoning worked before** in **conceptually similar** contexts.

This gives your agent:

- Semantic recall without explicit retraining.
- Runtime-only memory (no extra tool calls).
- Continual self-improvement through feedback loops.

> Think of ReasoningBank as _long-term procedural memory for reasoning itself_ —
> it doesn’t store facts, it stores _how to think_.
