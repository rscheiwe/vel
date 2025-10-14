---
layout: default
title: Memory Embeddings
parent: Memory System
nav_order: 4
---

# Embeddings in Vel Memory

Embeddings are at the heart of **ReasoningBank** and other memory modules in Vel.  
They transform text into **numerical vectors** that encode meaning, enabling similarity-based retrieval rather than exact string matching.

---

## 🧩 What Are Embeddings?

An _embedding_ is a dense vector representation of text — a list of floating-point numbers that captures semantic relationships between words or phrases.

> Example:  
> “summarize document” → `[0.12, -0.37, 0.41, ...]`  
> “condense report” → `[0.13, -0.36, 0.39, ...]`

These vectors are **close together** in embedding space, even though the words differ.

---

## 🧠 Why Vel Uses Embeddings

In Vel, embeddings power **semantic generalization** for memory retrieval.  
They enable modules like ReasoningBank to find strategies that _feel conceptually similar_ to the current task context.

| Module                     | Purpose of Embeddings                                       |
| -------------------------- | ----------------------------------------------------------- |
| `ReasoningBank`            | Measure similarity between current task and past strategies |
| (future) `EpisodicMemory`  | Match relevant prior facts or sessions                      |
| (planned) `KnowledgeCache` | Map concept graphs for long-term recall                     |

Embeddings give agents a way to “remember meaning” rather than “remember text.”

---

## ⚙️ The Embedding Function Interface

Vel’s memory system uses a simple, pluggable function signature:

```python
embeddings_fn: Callable[[List[str]], np.ndarray]
```

It must:

- Accept a list of text strings.
- Return a 2D NumPy array of `float32` vectors.
- Maintain consistent dimensionality across runs.

### Example Contract

```python
def my_embedder(texts: List[str]) -> np.ndarray:
    # Returns array shape (len(texts), D)
    ...
```

You pass this function to `MemoryConfig(embeddings_fn=my_embedder)`.

---

## 🚀 Embedding Options

Below are recommended strategies depending on your environment and scale.

### 1. Deterministic Hash Embeddings (Offline / No Dependencies)

Fast, reproducible, and dependency-free.
Use for prototyping or offline operation.

```python
def hash_embeddings(texts):
    import numpy as np, hashlib
    out = []
    for t in texts:
        h = hashlib.sha256(t.encode()).digest()
        v = np.frombuffer(h, dtype=np.uint8).astype(np.float32)
        v = (v - v.mean()) / (v.std() + 1e-8)
        out.append(v[:256])
    return np.vstack(out)
```

✅ Pros:

- No network calls
- Deterministic
- Works anywhere

⚠️ Cons:

- Not semantically meaningful — good for testing only.

---

### 2. SentenceTransformer (Recommended for Local Semantic Use)

Uses open-source transformer models to produce high-quality embeddings.

```python
from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer("all-MiniLM-L6-v2")

def encode(texts):
    return np.array(model.encode(texts, normalize_embeddings=True), dtype=np.float32)
```

✅ Pros:

- Semantic generalization
- Runs locally on CPU/GPU
- Consistent across environments

⚠️ Cons:

- Requires model download
- Slightly slower (~10–20ms per embedding)

---

### 3. OpenAI Embeddings (Cloud-Based)

For production workloads needing high recall accuracy.

```python
from openai import OpenAI
import numpy as np

client = OpenAI()

def encode(texts):
    response = client.embeddings.create(
        input=texts,
        model="text-embedding-3-large"
    )
    return np.array([d.embedding for d in response.data], dtype=np.float32)
```

✅ Pros:

- Best semantic quality
- Works across languages
- Consistent dimensionality

⚠️ Cons:

- Requires API key and internet access
- Adds network latency (~200ms)

---

## 📏 Dimensionality & Normalization

### Dimensionality

Different models produce embeddings of different lengths (e.g., 256, 384, 1024, 1536).
Vel does **not** enforce a dimension — it just stores vectors as-is.
Choose one and stay consistent.

### Normalization

Always L2-normalize vectors before computing similarity:
[
\hat{v} = \frac{v}{||v||}
]

Most libraries (like SentenceTransformers) handle this automatically via `normalize_embeddings=True`.

If you write your own embedder, normalize manually:

```python
v = v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-8)
```

---

## 🔍 Similarity Computation

Vel’s memory retrieval uses **cosine similarity**:

[
\text{sim}(A, B) = \frac{A \cdot B}{||A|| , ||B||}
]

It measures _angle_, not _magnitude_, so embeddings with similar meaning but different energy still match well.

---

## 🧩 Embeddings in ReasoningBank

In ReasoningBank, two embeddings are used:

| Vector                 | Purpose                                                     |
| ---------------------- | ----------------------------------------------------------- |
| **`vector_strategy`**  | Encodes the text of the learned heuristic itself.           |
| **`vector_signature`** | Encodes the structured metadata describing the run context. |

When retrieving:

1. The current task signature is embedded.
2. Cosine similarity is computed against all stored signature vectors.
3. The top-K most similar strategies are returned.

---

## ⚖️ Performance Considerations

| Operation                | Typical Time                   | Notes                          |
| ------------------------ | ------------------------------ | ------------------------------ |
| Embedding a short text   | 10–30 ms (SentenceTransformer) | Local CPU inference            |
| Cosine similarity search | <40 ms (≤1k items)             | Numpy vector ops               |
| DB read/write            | <5 ms                          | SQLite with serialized vectors |

If you anticipate >5k entries, migrate to **FAISS** or **Qdrant** for vector indexing.

---

## 🧠 Practical Tips

- Always normalize embeddings before storage.
- Use consistent models across learning and retrieval.
- Avoid mixing different embedding sources in the same DB.
- Periodically test similarity accuracy by inspecting top matches.

---

## 🔒 Reproducibility & Determinism

For reproducible behavior across environments:

- Fix random seeds in your embedding model.
- Keep model version pinned (e.g., `"all-MiniLM-L6-v2"`).
- Avoid dynamic dimensionality changes.

Hash-based embeddings are fully deterministic but non-semantic; semantic models require version control.

---

## 🧾 Example: Comparing Strategies

```python
texts = [
    "Summarize the user goal before planning",
    "Clarify the main objective before creating steps"
]

E = encode(texts)
similarity = (E[0] @ E[1]) / (np.linalg.norm(E[0]) * np.linalg.norm(E[1]))
print(similarity)  # ~0.9 with semantic embeddings
```

---

## 🧩 Summary

Embeddings are how Vel’s ReasoningBank **understands similarity**:

- They turn strategy and context into geometry.
- They enable reuse of reasoning patterns across diverse situations.
- They make “memory” semantic and adaptive.

When chosen well, embeddings let Vel agents **learn to reason better** — not just remember words.

> In short: embeddings make the agent’s memory _intelligent_ rather than _literal_.
