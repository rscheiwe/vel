# Setup Auto-Learning Recipe

**Goal:** Configure the full auto-learning pipeline for production
**Prerequisites:** Database setup, embeddings function, judge model access
**Estimated Time:** 1-2 hours

---

## Steps

### Step 1: Configure Database

```python
import os

# Set database path
os.environ['VEL_MEMORY_DB'] = '/path/to/.vel/vel.db'

# Or configure programmatically
from vel.core import ContextManager, MemoryConfig

ctx = ContextManager()
ctx.set_memory_config(MemoryConfig(
    mode='all',
    db_path='.vel/vel.db'
))
```

**Database will be created automatically with required tables.**

---

### Step 2: Configure Embeddings

Choose an embeddings provider:

#### Option A: OpenAI Embeddings

```python
import openai

async def openai_embeddings(text: str) -> list[float]:
    response = await openai.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding

embeddings_fn = openai_embeddings
```

#### Option B: Sentence Transformers (Local)

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

def local_embeddings(text: str) -> list[float]:
    return model.encode(text).tolist()

embeddings_fn = local_embeddings
```

#### Option C: Hash-based (Testing Only)

```python
import hashlib

def hash_embeddings(text: str) -> list[float]:
    h = hashlib.sha256(text.encode()).digest()
    return [b / 255.0 for b in h[:128]]

embeddings_fn = hash_embeddings  # Deterministic but not semantic
```

---

### Step 3: Configure Auto-Learning Manager

```python
from vel.memory import AutoLearningManager, AutoLearningConfig

config = AutoLearningConfig(
    # Database
    db_path='.vel/vel.db',

    # Embeddings
    embeddings_fn=embeddings_fn,

    # Judge configuration
    judge_provider='openai',          # or 'anthropic'
    judge_model='gpt-4o-mini',        # Cost-optimized

    # Worker intervals (seconds)
    evaluation_interval=60,           # How often to evaluate trajectories
    extraction_interval=120,          # How often to extract strategies
    consolidation_interval=3600,      # How often to consolidate (1 hour)

    # Thresholds
    confidence_threshold=0.3,         # Min confidence to keep strategies
    similarity_threshold=0.85,        # For deduplication
    max_strategies=1000,              # Cap total strategies

    # Cost controls
    max_evaluation_batch=10,          # Trajectories per evaluation batch
    max_concurrent_evaluations=3      # Parallel judge calls
)

manager = AutoLearningManager(config)
```

---

### Step 4: Seed Initial Strategies (Optional)

For cold start, seed with known-good strategies:

```python
seed_strategies = [
    {
        'strategy': 'Validate API response schema before processing',
        'domain': 'error_handling',
        'confidence': 0.5
    },
    {
        'strategy': 'Use pagination for large result sets',
        'domain': 'performance',
        'confidence': 0.5
    },
    {
        'strategy': 'Log tool inputs and outputs for debugging',
        'domain': 'observability',
        'confidence': 0.5
    }
]

await manager.seed_strategies(seed_strategies)
```

---

### Step 5: Start Background Workers

```python
# Start all workers
await manager.start()

# Or start individually
await manager.start_evaluation_worker()
await manager.start_extraction_worker()
await manager.start_consolidation_worker()
```

**Workers run in background, processing trajectories as they arrive.**

---

### Step 6: Connect to Agent

```python
from vel import Agent

agent = Agent(
    id='learning-agent',
    model={'provider': 'openai', 'model': 'gpt-4o'},
    context_manager=ctx,  # Uses memory config from Step 1
    tools=[...],
    policies={
        'record_trajectories': True  # Enable trajectory recording
    }
)

# Trajectories are automatically recorded after each run
result = await agent.run({'message': 'Do something...'})
```

---

### Step 7: Monitor Learning

```python
# Check learning stats
stats = await manager.get_stats()
print(f"Trajectories: {stats['total_trajectories']}")
print(f"Evaluated: {stats['evaluated_trajectories']}")
print(f"Strategies: {stats['total_strategies']}")
print(f"Avg Confidence: {stats['avg_confidence']:.2f}")

# View top strategies
strategies = await manager.get_top_strategies(limit=10)
for s in strategies:
    print(f"[{s['confidence']:.2f}] {s['strategy']}")
```

---

### Step 8: Production Deployment

```python
# Graceful shutdown
async def shutdown():
    await manager.stop()
    # Workers finish current batch before stopping

# Health check endpoint
async def health_check():
    return {
        'learning_enabled': manager.is_running(),
        'workers': {
            'evaluation': manager.evaluation_worker.is_running(),
            'extraction': manager.extraction_worker.is_running(),
            'consolidation': manager.consolidation_worker.is_running()
        }
    }
```

---

## Validation

### Verify Database Setup

```bash
sqlite3 .vel/vel.db ".tables"
# Should show: facts, strategies, trajectories, trajectory_tool_calls
```

### Verify Workers Running

```python
assert manager.is_running()
assert manager.evaluation_worker.is_running()
```

### Verify Trajectory Recording

```python
# After running agent
from vel.memory import TrajectoryStore
store = TrajectoryStore('.vel/vel.db')
recent = await store.get_recent(limit=5)
assert len(recent) > 0
```

### Verify Strategy Extraction

```python
# After successful runs
strategies = await manager.get_strategies()
assert len(strategies) > 0
```

---

## Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| No trajectories recorded | `record_trajectories` not set | Add to agent policies |
| Judge always fails | Model access issue | Check API key, try different model |
| Strategies not extracting | No successful trajectories | Run more successful tasks |
| High costs | Too many evaluations | Increase `evaluation_interval` |
| Memory growing | Too many strategies | Lower `max_strategies` |
| Strategies not improving | Low confidence threshold | Raise `confidence_threshold` |

---

## Cost Optimization

| Setting | Lower Cost | Higher Quality |
|---------|------------|----------------|
| `judge_model` | gpt-4o-mini | gpt-4o |
| `evaluation_interval` | 300 (5 min) | 30 (30 sec) |
| `max_evaluation_batch` | 5 | 20 |
| `similarity_threshold` | 0.90 | 0.80 |

**Estimated costs (gpt-4o-mini):**
- ~$0.001 per trajectory evaluation
- ~$0.002 per strategy extraction
- ~100 trajectories/day = ~$3/month

---

## Reference

- `vel/memory/auto_learning.py` - Implementation
- `docs/adr/004-memory-architecture.md` - Architecture decision
- `docs/memory-overview.md` - Memory system overview
- `.claude/rules/memory-system.md` - Development guidelines
