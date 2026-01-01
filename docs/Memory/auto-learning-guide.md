---
layout: default
title: Auto-Learning Guide
parent: Memory System
nav_order: 7
---

# Auto-Learning Step-by-Step Guide

This guide walks you through setting up ReasoningBank's auto-learning feature. Your agents will automatically learn from experience—extracting strategies from successful runs and avoiding patterns from failures.

---

## What You'll Build

After completing this guide, your agents will:

1. **Record** every run (what happened, what tools were used, how it ended)
2. **Evaluate** each run automatically (did it succeed or fail?)
3. **Extract** reusable strategies from successful runs
4. **Improve** over time by boosting good strategies and pruning bad ones

---

## Prerequisites

- Vel installed (`pip install vel`)
- An OpenAI API key (for the auto-evaluator)
- About 15 minutes

---

## Step 1: Set Up Your Embedding Function

The system needs a way to find similar strategies. You provide a function that converts text to numbers (embeddings).

**For testing (no dependencies):**

```python
import hashlib
import numpy as np

def encode_embeddings(texts):
    """Hash-based embeddings - good for testing."""
    out = []
    for t in texts:
        h = hashlib.sha256(t.encode()).digest()
        v = np.frombuffer(h, dtype=np.uint8).astype(np.float32)[:256]
        v = (v - v.mean()) / (v.std() + 1e-8)
        out.append(v)
    return np.vstack(out)
```

**For production (better similarity matching):**

```python
from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer("all-MiniLM-L6-v2")

def encode_embeddings(texts):
    """Semantic embeddings - better for production."""
    return np.array(model.encode(texts, normalize_embeddings=True), dtype=np.float32)
```

---

## Step 2: Configure Memory with Auto-Learning

Tell Vel you want auto-learning enabled:

```python
from vel.core import ContextManager, MemoryConfig

mem = MemoryConfig(
    mode="reasoning",              # Enable ReasoningBank
    db_path=".vel/memory.db",      # Where to store everything
    embeddings_fn=encode_embeddings,
    rb_top_k=5,                    # How many strategies to retrieve per run

    # Phase 2: Auto-learning options
    enable_auto_learning=True,     # Turn on automatic learning
    enable_trajectories=True       # Record agent runs
)

ctx = ContextManager()
ctx.set_memory_config(mem)
```

---

## Step 3: Set Up the Auto-Learning Manager

The manager runs background workers that evaluate and extract strategies:

```python
from vel.memory import (
    AutoLearningManager,
    AutoLearningConfig
)

config = AutoLearningConfig(
    enabled=True,
    llm_provider="openai",
    llm_model="gpt-4o-mini",       # Cheap and fast (~$0.0003 per evaluation)

    # How often to run each worker
    evaluation_interval_seconds=60,      # Check for new runs every minute
    extraction_interval_seconds=120,     # Extract strategies every 2 minutes
    consolidation_interval_seconds=21600 # Clean up every 6 hours
)

# Get the stores from context manager
adapters = ctx._adapters

manager = AutoLearningManager(
    config=config,
    trajectory_store=adapters["trajectories"],
    reasoning_bank_store=adapters["rb_store"],
    reasoning_bank=adapters["rb"]
)
```

---

## Step 4: Start the Background Workers

```python
import asyncio

async def main():
    # Start the learning system
    await manager.start()
    print("Auto-learning is running!")

    # ... your agent runs here ...

    # When you're done
    await manager.stop()

asyncio.run(main())
```

---

## Step 5: Run Your Agent (Learning Happens Automatically)

Now just run your agent normally. The system handles the rest:

```python
from vel import Agent

agent = Agent(
    id="my-agent",
    model={"provider": "openai", "model": "gpt-4o"},
    tools=["search", "calculate"]
)

# Before the run: get advice from past experience
signature = {"intent": "research", "domain": "science"}
advice = ctx.prepare_for_run(signature)

if advice:
    print(f"Tips from past runs:\n{advice}")

# Run the agent
result = await agent.run({"message": "What causes lightning?"})

# After the run: tell the system how it went
ctx.finalize_outcome(run_success=True)
```

**What happens behind the scenes:**

1. The run is recorded in TrajectoryStore
2. Background worker evaluates: "Did this run succeed?"
3. If yes → extract a strategy: "Research questions benefit from checking multiple sources"
4. Strategy is stored with 60% initial confidence
5. Next similar run retrieves this strategy as advice

---

## Step 6: (Optional) Add Seed Strategies

Start with proven strategies instead of learning from scratch:

```python
from vel.memory import EXAMPLE_SEED_STRATEGIES, populate_seed_strategies

# Add 11 research/critical-thinking strategies
count = populate_seed_strategies(adapters["rb_store"])
print(f"Added {count} seed strategies")
```

Example seed strategies include:
- "Gather evidence from at least three independent sources before synthesizing conclusions"
- "Generate multiple competing hypotheses, then design tests that could falsify each"
- "Reproduce the issue with minimal steps first, then bisect to isolate the cause"

---

## How Confidence Evolves

Strategies get stronger or weaker based on outcomes:

| Outcome | What Happens | Example |
|---------|--------------|---------|
| Success | Confidence × 1.20 | 0.60 → 0.72 |
| Failure | Confidence × 0.85 | 0.60 → 0.51 |

Confidence is capped between 5% and 95%. Strategies below 20% are automatically pruned.

**Example evolution:**

```
Strategy: "Break complex tasks into smaller steps"

Run 1: Success  → 0.50 → 0.60
Run 2: Success  → 0.60 → 0.72
Run 3: Failure  → 0.72 → 0.61
Run 4: Success  → 0.61 → 0.73
Run 5: Success  → 0.73 → 0.88
Run 6: Success  → 0.88 → 0.95 (capped)
```

---

## Complete Example

Here's everything together:

```python
import asyncio
import hashlib
import numpy as np
from vel import Agent
from vel.core import ContextManager, MemoryConfig
from vel.memory import AutoLearningManager, AutoLearningConfig

# Step 1: Embedding function
def encode_embeddings(texts):
    out = []
    for t in texts:
        h = hashlib.sha256(t.encode()).digest()
        v = np.frombuffer(h, dtype=np.uint8).astype(np.float32)[:256]
        v = (v - v.mean()) / (v.std() + 1e-8)
        out.append(v)
    return np.vstack(out)

# Step 2: Configure memory
mem = MemoryConfig(
    mode="reasoning",
    db_path=".vel/memory.db",
    embeddings_fn=encode_embeddings,
    enable_auto_learning=True,
    enable_trajectories=True
)

ctx = ContextManager()
ctx.set_memory_config(mem)

# Step 3: Set up auto-learning
config = AutoLearningConfig(
    enabled=True,
    llm_provider="openai",
    llm_model="gpt-4o-mini"
)

manager = AutoLearningManager(
    config=config,
    trajectory_store=ctx._adapters["trajectories"],
    reasoning_bank_store=ctx._adapters["rb_store"],
    reasoning_bank=ctx._adapters["rb"]
)

async def main():
    # Step 4: Start learning
    await manager.start()

    # Step 5: Run your agent
    agent = Agent(
        id="learning-agent",
        model={"provider": "openai", "model": "gpt-4o"}
    )

    signature = {"intent": "planning", "domain": "api"}

    # Get advice from past runs
    advice = ctx.prepare_for_run(signature)
    if advice:
        print(f"Advice: {advice}")

    # Run
    result = await agent.run({"message": "Design a REST API for a todo app"})

    # Report outcome
    ctx.finalize_outcome(run_success=True)

    # Let workers process
    await asyncio.sleep(5)

    # Stop
    await manager.stop()

asyncio.run(main())
```

---

## Monitoring and Debugging

### Check what's in the database

```python
# See all strategies
rb_store = ctx._adapters["rb_store"]
strategies = rb_store.retrieve({"intent": "planning"}, k=20, min_conf=0.0)

for s in strategies:
    print(f"[{s.confidence:.2f}] {s.strategy_text}")
```

### Check pending trajectories

```python
traj_store = ctx._adapters["trajectories"]
stats = traj_store.get_statistics()

print(f"Total runs: {stats['total']}")
print(f"Pending evaluation: {stats['pending_evaluation']}")
print(f"Pending extraction: {stats['pending_extraction']}")
```

### Run consolidation manually

```python
from vel.memory import MemoryConsolidator

consolidator = MemoryConsolidator(
    reasoning_bank_store=rb_store,
    max_strategies=1000,
    min_confidence=0.20
)

result = consolidator.consolidate()
print(f"Merged: {result.strategies_merged}")
print(f"Pruned: {result.strategies_pruned}")
```

---

## Common Questions

### How much does auto-learning cost?

With gpt-4o-mini:
- Evaluation: ~$0.0003 per run
- Extraction: ~$0.0005 per successful run
- **100 runs/day ≈ $0.08/day**

### Can I use a different LLM?

Yes, change the config:

```python
config = AutoLearningConfig(
    llm_provider="anthropic",
    llm_model="claude-3-haiku-20240307"
)
```

### How do I turn off auto-learning?

```python
mem = MemoryConfig(
    mode="reasoning",
    enable_auto_learning=False,  # Disable
    enable_trajectories=False
)
```

### What if I want manual control?

See [Enabling ReasoningBank](enabling-reasoningbank.md) for the manual (Phase 1) approach.

---

## Next Steps

- Run the full demo: `python examples/memory_examples/auto_learning_demo.py`
- Read the [Phase 2 Roadmap](reasoningbank-phase2-roadmap.md) for technical details
- Check the [Memory Architecture](memory-architecture.md) for system design

---

## Summary

| Step | What You Do | What Happens |
|------|-------------|--------------|
| 1 | Provide embedding function | System can find similar strategies |
| 2 | Configure with `enable_auto_learning=True` | Recording and learning enabled |
| 3 | Create `AutoLearningManager` | Background workers ready |
| 4 | Call `await manager.start()` | Workers begin processing |
| 5 | Run agents normally | Runs are recorded automatically |
| 6 | Call `finalize_outcome()` | Confidence updates happen |

Your agents now learn from every run. Successful patterns get reinforced, unsuccessful ones fade away.
