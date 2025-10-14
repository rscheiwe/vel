---
layout: default
title: Enabling ReasoningBank
parent: Memory System
nav_order: 5
---

# Enabling ReasoningBank

This guide explains how to enable and populate ReasoningBank in Vel, and clarifies the distinction between **infrastructure** and **content**.

---

## Two Phases: Infrastructure vs. Content

Enabling ReasoningBank happens in **two distinct phases**:

1. **Phase 1: Enable the Infrastructure** — Turn on the system (automatic via config)
2. **Phase 2: Populate with Strategies** — Add content (manual or programmatic)

**Key Insight:** Enabling ReasoningBank gives you an empty database with retrieval/update mechanisms, but you must explicitly add strategy items.

---

## Phase 1: Enable the Infrastructure

### Step 1: Configure Memory

```python
from vel.core import ContextManager, MemoryConfig
import numpy as np

# Define embeddings function (required for ReasoningBank)
def encode_embeddings(texts):
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")
    return np.array(model.encode(texts, normalize_embeddings=True), dtype=np.float32)

# Configure ReasoningBank
mem = MemoryConfig(
    mode="reasoning",           # Enable ReasoningBank
    db_path=".vel/vel.db",          # SQLite database path
    rb_top_k=5,                     # Top-K strategies to retrieve
    embeddings_fn=encode_embeddings # Embedding function
)
```

### Step 2: Enable on Context Manager

```python
ctx = ContextManager()
ctx.set_memory_config(mem)
```

**At this point:**
- ✅ SQLite database created at `.vel/vel.db`
- ✅ Tables `rb_strategies` and `rb_embeddings` initialized
- ✅ Retrieval mechanism ready
- ✅ Confidence scoring enabled
- ❌ **But the database is empty!**

Running `ctx.prepare_for_run(signature)` will return `""` (no strategies).

---

## Phase 2: Populate with Strategies

ReasoningBank is **empty by default**. You must explicitly add strategy items.

### Why This Design?

This aligns with Vel's **"Own Your Memory"** philosophy:

- You decide what strategies are worth storing
- You control when to extract them
- You determine how to evaluate success
- You define what constitutes a "good" strategy

Vel doesn't make these decisions automatically behind the scenes.

---

## Usage Patterns

### Pattern 1: Manual Curation (Simplest)

Hand-craft strategies based on domain knowledge:

```python
from vel.core import ContextManager, MemoryConfig

# ... configure memory (see Phase 1) ...

# Get ReasoningBank adapter
rb = ctx._adapters.get("rb")

if rb:
    # Add strategies manually
    rb.store.upsert_strategy(
        signature={"intent": "planning", "domain": "api"},
        strategy_text="Always clarify the user's goal before creating a plan",
        anti_patterns=["Skip validation", "Assume requirements"],
        evidence_refs=[],
        confidence=0.7
    )

    rb.store.upsert_strategy(
        signature={"intent": "debugging", "domain": "backend"},
        strategy_text="Check logs and error messages before proposing solutions",
        anti_patterns=["Jump to conclusions", "Ignore stack traces"],
        confidence=0.8
    )

    rb.store.upsert_strategy(
        signature={"intent": "refactoring", "domain": "python"},
        strategy_text="Run tests before and after refactoring to ensure correctness",
        anti_patterns=["Refactor without tests", "Change too much at once"],
        confidence=0.75
    )
```

**When to use:** You have domain expertise and want to seed the bank with known good practices.

---

### Pattern 2: Post-Run Analysis (Semi-Automatic)

Analyze runs after completion and programmatically extract strategies:

```python
async def analyze_and_learn(run_id, success, trajectory, ctx):
    """
    Analyze a completed run and optionally add a strategy to ReasoningBank.

    Args:
        run_id: The run identifier
        success: Whether the run was successful
        trajectory: List of messages/steps from the run
        ctx: ContextManager instance
    """
    rb = ctx._adapters.get("rb")
    if not rb:
        return

    # Only learn from successful runs (or use failures for anti-patterns)
    if success:
        # Option A: Extract strategy with simple heuristics
        strategy = extract_strategy_heuristic(trajectory)

        # Option B: Use LLM to distill strategy
        # strategy = await extract_strategy_with_llm(trajectory)

        if strategy:
            rb.store.upsert_strategy(
                signature=extract_signature(trajectory),
                strategy_text=strategy,
                confidence=0.6  # Start conservative
            )

def extract_strategy_heuristic(trajectory):
    """
    Example heuristic: Look for patterns in successful tool usage.
    """
    # Your logic here
    # This is simplified - real implementation would be more sophisticated
    tool_sequence = [msg.get('tool') for msg in trajectory if msg.get('tool')]

    if len(tool_sequence) > 3:
        return f"For complex tasks, break down into {len(tool_sequence)} steps"

    return None

def extract_signature(trajectory):
    """
    Extract task characteristics from trajectory.
    """
    # Analyze the trajectory to determine intent, domain, risk
    return {
        "intent": "planning",  # derived from trajectory
        "domain": "api",       # derived from trajectory
        "risk": "low"          # derived from trajectory
    }
```

**When to use:** You want semi-automatic learning where the system proposes strategies but you control the extraction logic.

---

### Pattern 3: LLM-Based Strategy Distillation (Advanced)

Use an LLM to analyze trajectories and extract strategies (closest to the academic paper):

```python
async def extract_strategy_with_llm(trajectory):
    """
    Use an LLM to distill a strategy from a successful trajectory.

    This mimics the academic ReasoningBank paper's approach.
    """
    from vel import Agent

    # Serialize trajectory
    trajectory_text = format_trajectory(trajectory)

    # Create a strategy extraction agent
    extractor = Agent(
        id='strategy-extractor:v1',
        model={'provider': 'anthropic', 'model': 'claude-sonnet-4'},
        tools=[]
    )

    prompt = f"""
    Analyze the following successful agent trajectory and extract a generalizable
    reasoning strategy. Provide:

    1. A one-sentence strategy (what worked well)
    2. 1-3 anti-patterns (what to avoid)

    Trajectory:
    {trajectory_text}

    Respond in JSON format:
    {{
      "strategy_text": "One sentence strategy",
      "anti_patterns": ["Pattern 1", "Pattern 2"]
    }}
    """

    result = await extractor.run({"message": prompt})

    # Parse LLM response (add error handling in production)
    import json
    strategy_data = json.loads(result)

    return strategy_data

def format_trajectory(trajectory):
    """
    Format trajectory for LLM analysis.
    """
    lines = []
    for i, step in enumerate(trajectory, 1):
        role = step.get('role', 'unknown')
        content = step.get('content', '')
        tool = step.get('tool', '')

        if tool:
            lines.append(f"Step {i} [{role}]: Used tool '{tool}'")
        else:
            lines.append(f"Step {i} [{role}]: {content[:100]}...")

    return "\n".join(lines)
```

**Usage in agent code:**

```python
async def run_with_learning(agent, input_data, ctx):
    """
    Run agent and automatically learn from trajectory.
    """
    # Run agent
    trajectory = []
    success = False

    try:
        async for event in agent.run_stream(input_data):
            trajectory.append(event)
            # ... handle events ...

        success = True  # Determine based on your criteria

    except Exception as e:
        success = False

    # Learn from this run
    if success:
        strategy_data = await extract_strategy_with_llm(trajectory)

        rb = ctx._adapters.get("rb")
        if rb and strategy_data:
            rb.store.upsert_strategy(
                signature=extract_signature(trajectory),
                strategy_text=strategy_data["strategy_text"],
                anti_patterns=strategy_data.get("anti_patterns", []),
                confidence=0.6
            )
```

**When to use:** You want automatic strategy extraction similar to the academic paper, but you still control when it happens.

---

### Pattern 4: Batch Learning (Offline)

Periodically analyze historical runs and extract patterns:

```python
def learn_from_history(db_path=".vel/vel.db"):
    """
    Batch process historical runs to extract strategies.

    Run this as a cron job or manual script.
    """
    from vel.memory import ReasoningBankStore, Embeddings
    from vel.core import MemoryConfig

    # Load all successful runs from your run store
    successful_runs = load_successful_runs_from_db()

    # Group by similarity
    clusters = cluster_similar_runs(successful_runs)

    # Extract common patterns
    mem = MemoryConfig(mode="reasoning", embeddings_fn=encode_fn)
    ctx = ContextManager()
    ctx.set_memory_config(mem)

    rb = ctx._adapters.get("rb")

    for cluster in clusters:
        # Find common patterns in cluster
        pattern = extract_common_pattern(cluster)

        if pattern:
            rb.store.upsert_strategy(
                signature=pattern["signature"],
                strategy_text=pattern["strategy"],
                confidence=calculate_confidence(cluster),
                evidence_refs=[run["id"] for run in cluster]
            )

def cluster_similar_runs(runs):
    """
    Group runs by similarity (use embeddings, intent, domain, etc.)
    """
    # Your clustering logic
    pass

def extract_common_pattern(cluster):
    """
    Find what successful runs in this cluster have in common.
    """
    # Your pattern extraction logic
    pass
```

**When to use:** You have a large corpus of runs and want to extract strategies in bulk offline.

---

## Comparison: Vel vs. Academic Paper

### Academic ReasoningBank (Fully Automatic)

```
1. Agent runs task
2. LLM-as-Judge evaluates: success or failure?
3. LLM distills trajectory → strategy item {title, description, content}
4. Strategy automatically added to bank
5. Next run retrieves similar strategies
6. Cycle repeats (self-evolving)
```

**Everything is automatic** — agent learns from every run without human intervention.

### Vel ReasoningBank (Infrastructure + User Control)

```
1. Agent runs task
2. ✅ Vel updates confidence if strategies were used (automatic)
3. ❌ You evaluate success/failure (provide boolean)
4. ❌ You distill trajectory → strategy (manual or your code)
5. ❌ You add strategy to bank (explicit call)
6. ✅ Next run retrieves similar strategies (automatic)
7. ✅ Confidence updates over time (automatic)
```

**Vel provides the infrastructure** (steps 2, 6, 7) but leaves strategy creation (steps 3-5) to you.

---

## Verification: Check What's in ReasoningBank

```python
# Check if strategies exist
rb = ctx._adapters.get("rb")
if rb:
    signature = {"intent": "planning", "domain": "api"}
    strategies = rb.get_advice(signature, k=10)

    print(f"Found {len(strategies)} strategies:")
    for s in strategies:
        print(f"  - {s.strategy_text} (confidence: {s.confidence:.2f})")
```

---

## Quick Start Checklist

- [ ] Configure `MemoryConfig` with embeddings function
- [ ] Enable on `ContextManager` via `set_memory_config()`
- [ ] **Add strategies** using one of the patterns above
- [ ] Call `prepare_for_run(signature)` before agent execution
- [ ] Call `finalize_outcome(success, fail_notes)` after execution
- [ ] Verify strategies are being retrieved and updated

---

## Common Pitfalls

### ❌ Pitfall 1: Expecting Automatic Learning

**Problem:** Enabling ReasoningBank and expecting it to learn automatically.

**Solution:** You must explicitly add strategies. See usage patterns above.

### ❌ Pitfall 2: No Embeddings Function

**Problem:** Setting `mode="reasoning"` but `embeddings_fn=None`.

**Solution:** Always provide an embeddings function:

```python
# Minimum (hash-based, for testing)
def hash_embed(texts):
    import hashlib
    out = []
    for t in texts:
        h = hashlib.sha256(t.encode()).digest()
        v = np.frombuffer(h, dtype=np.uint8).astype(np.float32)[:256]
        v = (v - v.mean()) / (v.std() + 1e-8)
        out.append(v)
    return np.vstack(out)

# Better (semantic)
def semantic_embed(texts):
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")
    return np.array(model.encode(texts, normalize_embeddings=True), dtype=np.float32)
```

### ❌ Pitfall 3: Forgetting to Call finalize_outcome

**Problem:** Strategies never update confidence scores.

**Solution:** Always call `finalize_outcome` after agent execution:

```python
signature = {"intent": "planning"}
advice = ctx.prepare_for_run(signature)

# ... run agent ...

ctx.finalize_outcome(run_success=True)  # or False with fail_notes
```

### ❌ Pitfall 4: Vague Strategy Text

**Problem:** Strategy text is too specific or vague:
- Too specific: "Use requests.get to fetch data from the API at port 8000"
- Too vague: "Do things correctly"

**Solution:** Keep strategies generalizable but actionable:
- ✅ "Validate API responses before processing data"
- ✅ "Check input parameters before executing operations"
- ✅ "Break complex tasks into 3-5 distinct phases"

---

## Next Steps

- Review [Memory Architecture](memory-architecture.md) for system design
- See [ReasoningBank Strategy](reasoningbank-strategy.md) for implementation details
- Check [Memory Embeddings](memory-embeddings.md) for embedding options
- Run tests: `pytest tests/test_memory.py tests/test_memory_context.py`

---

## Summary

**Enabling ReasoningBank** = Infrastructure setup (automatic via config)
**Populating ReasoningBank** = Adding strategies (manual or your code)

Vel gives you the tools, but you decide what gets learned and when. This "Own Your Memory" approach ensures full control over your agent's learning process.
