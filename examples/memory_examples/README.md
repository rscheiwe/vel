# Memory Examples

This directory contains end-to-end demonstrations of Vel's memory system.

## Vel's Three Memory Systems

Vel has three distinct memory systems:
1. **Message History** - Conversation turns (automatic, managed by ContextManager)
2. **Fact Store** - Long-term structured facts (manual, shown in Example 1)
3. **Session Persistence** - Where message history is saved (infrastructure layer)

## Examples

### 1. Fact Store (`fact_store_demo.py`)

Demonstrates the namespaced key-value store for long-term structured data.

**Use Cases:**
- User preferences (theme, language, expertise level)
- Project metadata (current project, technologies used)
- Domain knowledge (company facts, API endpoints)
- Application state (feature flags, configuration)

**Features:**
- Store user preferences and facts
- Retrieve context before agent runs
- Use context to personalize agent responses
- Save interaction results back to fact store
- Demonstrate persistence across sessions

**Run:**
```bash
# Set API key
export ANTHROPIC_API_KEY=sk-ant-...
# or
export OPENAI_API_KEY=sk-...

# Run demo
python examples/memory_examples/fact_store_demo.py
```

**What to expect:**
- Streaming agent output with context-aware responses
- Facts stored and retrieved from SQLite
- Facts persist across sessions

---

### 2. ReasoningBank Memory (`reasoningbank_integration.py`)

Demonstrates strategy-level memory for learning reasoning patterns.

**Features:**
- Configure ReasoningBank with embeddings
- Seed initial strategies manually (Phase 1)
- Retrieve top-K relevant strategies via similarity
- Inject strategy advice into agent prompts
- Update confidence scores based on outcomes
- Show strategy evolution over time

**Run:**
```bash
# Set API key
export ANTHROPIC_API_KEY=sk-ant-...
# or
export OPENAI_API_KEY=sk-...

# Run demo
python examples/memory_examples/reasoningbank_integration.py
```

**What to expect:**
- Streaming agent output guided by learned strategies
- Strategy advice injected into system prompt
- Confidence scores update after each run
- Demonstration of different task signatures

---

## Database Location

Both examples use `.vel/demo.db` for persistent storage.

To reset memory between runs:
```bash
rm -rf .vel/demo.db
```

---

## Phase 1 vs Phase 2

Both examples demonstrate **Phase 1** functionality:
- Manual strategy creation (you control what's learned)
- Automatic retrieval and confidence updates
- Infrastructure for memory management

**Phase 2** (future) will add:
- Automatic strategy extraction from trajectories
- LLM-as-Judge for success evaluation
- Self-evolving agents that learn without manual curation

See `docs/Memory/reasoningbank-phase2-roadmap.md` for Phase 2 implementation plans.

---

## Embedding Options

The ReasoningBank example uses **hash-based embeddings** for simplicity (no external dependencies).

For production, use semantic embeddings:

### Option 1: SentenceTransformers (Recommended)
```python
from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer("all-MiniLM-L6-v2")

def encode(texts):
    return np.array(model.encode(texts, normalize_embeddings=True), dtype=np.float32)
```

### Option 2: OpenAI Embeddings
```python
from openai import OpenAI
import numpy as np

client = OpenAI()

def encode(texts):
    response = client.embeddings.create(
        input=texts,
        model="text-embedding-3-small"
    )
    return np.array([d.embedding for d in response.data], dtype=np.float32)
```

---

## Documentation

For more information, see:
- [Memory Overview](../../docs/memory.md)
- [Enabling ReasoningBank](../../docs/Memory/enabling-reasoningbank.md)
- [Memory Architecture](../../docs/Memory/memory-architecture.md)
- [Phase 2 Roadmap](../../docs/Memory/reasoningbank-phase2-roadmap.md)

---

## Troubleshooting

**ImportError: No module named 'agents'**
```bash
# Install Vel in development mode
pip install -e .
```

**No API key set**
```bash
# Set Anthropic key
export ANTHROPIC_API_KEY=sk-ant-...

# Or OpenAI key
export OPENAI_API_KEY=sk-...
```

**Memory not persisting**
- Check that `.vel/demo.db` exists after first run
- Ensure write permissions for `.vel/` directory
- SQLite database is created automatically on first use

**No strategies retrieved (ReasoningBank)**
- First run creates empty database
- Strategies are seeded automatically on first run
- Run the example twice to see persistence

---

## Next Steps

After running these examples:

1. **Customize for your use case:**
   - Modify namespaces for your application
   - Adjust signatures to match your task taxonomy
   - Create domain-specific strategies

2. **Integrate into your agents:**
   - Use `MemoryConfig` in your agent initialization
   - Call `prepare_for_run()` before each execution
   - Call `finalize_outcome()` after completion

3. **Monitor and improve:**
   - Track confidence scores over time
   - Prune low-confidence strategies
   - Add new strategies based on patterns you observe

4. **Consider Phase 2:**
   - Review the Phase 2 roadmap
   - Evaluate automatic learning needs
   - Plan implementation timeline
