# ADR-004: Three-Tier Memory Architecture

**Status:** Accepted
**Date:** 2025-01-08
**Decision Makers:** Vel Core Team

---

## Context

AI agents need different types of memory:
- **Short-term**: Current conversation context
- **Long-term facts**: User preferences, domain knowledge
- **Strategic**: Learned behaviors, successful patterns

A single memory system cannot optimally serve all these needs.

## Decision

Implement a **three-tier memory architecture**, all opt-in:

### Tier 1: Message History (Always Active)

Conversation turns managed by `ContextManager`:

```python
ctx.append_message(run_id, role='user', content='...')
ctx.get_context(run_id)  # Returns message history
```

Supports transient (in-memory) or persistent (database) storage.

### Tier 2: FactStore (Opt-In)

Namespaced key-value store for long-term structured facts:

```python
ctx.fact_put('user:alice', 'theme', 'dark')
ctx.fact_get('user:alice', 'theme')  # Returns 'dark'
ctx.fact_list('user:alice')  # Returns all facts for namespace
```

Backed by SQLite. Suitable for:
- User preferences
- Domain knowledge
- Configuration state

### Tier 3: ReasoningBank (Opt-In)

Strategy memory with embedding-based retrieval:

```python
ctx.rb_add(strategy='Check API status before retrying', success=True)
strategies = ctx.rb_retrieve('API call failed', top_k=5)
```

Features:
- **Embeddings**: Pluggable encoder (sentence-transformers, OpenAI, hash)
- **Confidence Scoring**: Bayesian-style updates (1.20x success, 0.85x failure)
- **Anti-Patterns**: Track what NOT to do

### Auto-Learning Pipeline (Phase 2)

Background workers for automatic learning:

| Component | Purpose |
|-----------|---------|
| `TrajectoryStore` | Records execution traces |
| `LLMJudge` | Automatic success/failure evaluation |
| `StrategyExtractor` | Distills strategies from successes |
| `MemoryConsolidator` | Deduplication and pruning |

## Consequences

### Positive

1. **Opt-In Complexity**: Simple agents don't pay memory overhead
2. **Separation of Concerns**: Each tier optimized for its use case
3. **Bayesian Confidence**: Strategies improve with consistent success
4. **Background Processing**: Learning doesn't block execution

### Negative

1. **Configuration Complexity**: Multiple memory modes to understand
2. **Storage Requirements**: SQLite database, embeddings storage
3. **Cold Start**: New agents have no learned strategies

## Configuration

```python
from vel.core import ContextManager, MemoryConfig

ctx = ContextManager()
ctx.set_memory_config(MemoryConfig(
    mode='all',           # 'none' | 'facts' | 'reasoning' | 'all'
    db_path='.vel/vel.db'
))
```

Environment variables:
```bash
VEL_MEMORY_MODE=all
VEL_MEMORY_DB=.vel/vel.db
```

## References

- `vel/core/context.py` - ContextManager
- `vel/memory/fact_store.py` - FactStore
- `vel/memory/strategy_reasoningbank.py` - ReasoningBank
- `vel/memory/auto_learning.py` - AutoLearningManager
- `docs/memory-overview.md` - User documentation
