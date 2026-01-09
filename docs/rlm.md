---
layout: default
title: RLM (Recursive Language Model)
nav_order: 12
---

# RLM (Recursive Language Model)

RLM is a middleware that enables Vel agents to handle very long contexts through recursive reasoning, bounded iteration, scratchpad notes, and specialized context probing.

## What is RLM?

RLM acts as a **subprocess or middleware** between your user's message and the normal chat-completions call. Instead of sending the entire context directly to the LLM, RLM runs an intelligent multi-step reasoning session that probes the context iteratively.

### High-Level Architecture

```
User → Agent.run(input, context_refs)
         ↓ (RLM enabled check)
       RlmController.run()
         ↓ (iterative loop)
       ┌─────────────────────────────────────────┐
       │ RLM Control Loop                        │
       │ ─────────────────────────────────────── │
       │ 1. Load context → ContextStore          │
       │ 2. Initialize scratchpad + budget       │
       │ 3. Iterate until FINAL() or exhausted:  │
       │    - provider.generate() ← LLM API call │
       │    - Execute tools (context_probe, etc) │
       │    - Update scratchpad with notes       │
       │    - Check for FINAL() signal           │
       │ 4. Optional: Writer synthesis           │
       └─────────────────────────────────────────┘
         ↓
       Returns answer to Agent.run()
         ↓
       User receives final answer
```

### From Your App's Perspective

**There is NO separate API for RLM** - you always use the same methods:

```python
# Same API whether RLM is enabled or not
answer = await agent.run(input, context_refs=...)
# or
async for event in agent.run_stream(input, context_refs=...):
    ...
```

**Internal routing is transparent:**
1. `Agent.run()` checks if RLM should activate (`rlm.enabled=True` AND `context_refs` provided)
2. If yes → Routes to `RlmController.run()` (middleware subprocess)
3. RlmController runs multi-step reasoning loop with context probing
4. Returns final answer back to `Agent.run()`
5. User receives answer - no difference from standard flow

**In short: RLM = middleware that sits inside `Agent.run()`, not a separate entry point. The user experience is identical whether RLM is enabled or not.**

### Key Difference from Standard Agent Execution

| Standard Agent | RLM Agent |
|----------------|-----------|
| Load all context in prompt | Probe context via tools |
| Single LLM call → answer | Multiple LLM calls → reasoning loop → answer |
| Limited by context window | Handles 5MB+ contexts |
| No intermediate notes | Scratchpad accumulates notes |
| Direct answer | Iterative refinement → synthesis |

## Overview

Traditional LLMs have context window limitations. RLM solves this by:
- **Probing context** via tools rather than loading everything at once
- **Iterative reasoning** with a scratchpad for notes
- **Recursive decomposition** for complex questions
- **Budget enforcement** for cost and performance control

Based on [Alex Zhang's RLM approach](https://alexzhang13.github.io/blog/2025/rlm/).

## Quick Start

```python
from vel import Agent

# Create agent with RLM enabled
agent = Agent(
    id='rlm-agent:v1',
    model={'provider': 'openai', 'model': 'gpt-4o-mini'},
    rlm={
        'enabled': True,
        'depth': 1,  # Allow one level of recursion
        'budgets': {
            'max_steps_root': 12,
            'max_tokens_total': 120000,
            'max_cost_usd': 0.50
        }
    }
)

# Use with large context
answer = await agent.run(
    input={'message': 'What are the key features?'},
    context_refs="Very large document content..."
)
```

## Configuration

### RlmConfig

```python
rlm_config = {
    'enabled': True,           # Enable RLM
    'depth': 1,                # Recursion depth (0-2)
    'control_model': {         # Fast model for iteration
        'provider': 'openai',
        'model': 'gpt-4o-mini'
    },
    'writer_model': {          # Strong model for synthesis
        'provider': 'openai',
        'model': 'gpt-4o'
    },
    'notes_cap': 200,          # Max notes in scratchpad
    'notes_window': 40,        # Recent notes to show
    'budgets': {
        'max_steps_root': 12,      # Max tool calls (root)
        'max_steps_child': 8,       # Max tool calls (child)
        'max_tokens_total': 120000, # Total token limit
        'max_cost_usd': 0.50        # Cost limit
    },
    'tools': {
        'allow_exec': False,       # Enable python_exec (security risk)
        'probe_max_bytes': 4096    # Max bytes per probe
    },
    'stream_events': True      # Emit RLM stream events
}
```

### Context References

Pass context via `context_refs` parameter:

```python
# Raw text
context_refs = "Long document text..."

# File path
context_refs = "/path/to/document.txt"

# Multiple files
context_refs = [
    "/path/to/doc1.txt",
    "/path/to/doc2.txt"
]

# Structured references
context_refs = [
    {'type': 'text', 'source': 'Raw text content'},
    {'type': 'file', 'source': '/path/to/file.txt'},
]
```

## How It Works

### 1. Control Loop

RLM runs an iterative loop with the following detailed execution flow:

```
┌─────────────────────────────────────────────────────────────┐
│ RLM Controller Initialization                               │
├─────────────────────────────────────────────────────────────┤
│ 1. Load context_refs → ContextStore (chunked)              │
│ 2. Initialize Scratchpad (empty notes)                     │
│ 3. Initialize Budget (steps, tokens, cost limits)          │
│ 4. Build system prompt with RLM rules + tool schemas       │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ Iterative Reasoning Loop (Step N)                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────────────────────────────────┐              │
│  │ 1. Check Budget                          │              │
│  │    - Steps exhausted? → best-effort exit │              │
│  │    - Tokens/cost OK? → continue          │              │
│  └──────────────────────────────────────────┘              │
│                  ↓                                          │
│  ┌──────────────────────────────────────────┐              │
│  │ 2. Call LLM (control model)             │              │
│  │    Input:                                │              │
│  │    - System: RLM planner prompt          │              │
│  │    - User: Original question             │              │
│  │    - System: Scratchpad (last N notes)   │              │
│  │    - Tools: [context_probe, rlm_call]    │              │
│  └──────────────────────────────────────────┘              │
│                  ↓                                          │
│  ┌──────────────────────────────────────────┐              │
│  │ 3. Parse LLM Response                    │              │
│  │    - Contains tool calls?                │              │
│  │      YES → Execute tools                 │              │
│  │      NO  → Check for FINAL()             │              │
│  └──────────────────────────────────────────┘              │
│                  ↓                                          │
│  ┌──────────────────────────────────────────┐              │
│  │ 4a. Execute Tool (if tool calls present) │              │
│  │     - context_probe(search/read/summ)    │              │
│  │     - rlm_call(sub-query, child context) │              │
│  │     - python_exec(code) [if enabled]     │              │
│  │     → Returns: {preview, meta, truncated}│              │
│  └──────────────────────────────────────────┘              │
│                  ↓                                          │
│  ┌──────────────────────────────────────────┐              │
│  │ 4b. Update Scratchpad                    │              │
│  │     - Extract key info from tool result  │              │
│  │     - Add Note(text, source_hint)        │              │
│  │     - Deduplicate by ID                  │              │
│  │     - Cap if exceeds max_notes           │              │
│  └──────────────────────────────────────────┘              │
│                  ↓                                          │
│  ┌──────────────────────────────────────────┐              │
│  │ 5. Check for FINAL() Signal              │              │
│  │    - Regex: FINAL("answer")              │              │
│  │    - Or: FINAL_VAR(variable_name)        │              │
│  │    Found? → Exit loop                    │              │
│  │    Not found? → Loop back to step 1      │              │
│  └──────────────────────────────────────────┘              │
│                                                             │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ Optional: Writer Synthesis                                  │
├─────────────────────────────────────────────────────────────┤
│ If writer_model configured:                                 │
│   1. Format scratchpad as bullet points                     │
│   2. Call writer model with:                                │
│      - System: Writer synthesis prompt                      │
│      - User: Question + scratchpad notes                    │
│   3. Writer produces final answer with citations            │
│   4. Replace raw answer with synthesized version            │
└─────────────────────────────────────────────────────────────┘
                          ↓
                  Return Final Answer
```

**Key Points:**
- Each iteration calls the LLM with updated scratchpad context
- Tools probe the large context in small, bounded chunks
- Budget enforcement prevents runaway costs
- FINAL() signal is the termination condition
- Writer synthesis (optional) produces polished output

### 2. Tools

RLM provides three tools for interacting with large contexts:

#### context_probe - Safe Context Probing

The primary tool for accessing large contexts safely:

**Operations:**
- `search` - Keyword/regex search over context
- `read` - Read specific chunk by ID
- `summarize` - Summarize chunk or text

**Example:**
```python
# LLM calls this internally
context_probe(kind='search', query='pricing', max_results=10)
→ Returns: {preview: "Found 3 results...", meta: {...}}
```

**Safety:** ✅ Always enabled, no code execution

#### rlm_call - Recursive Sub-queries

Spawn depth-bounded child RLM calls for decomposition:

**Usage:**
```python
# LLM calls this to delegate sub-questions
rlm_call(
    query="What are the system requirements?",
    context_slice="chunk_5"
)
→ Returns: {answer: "...", notes: [...]}
```

**Depth control:** Only available when `depth > 0`

#### python_exec - REPL-Style Code Execution ⚠️

**Inspired by the original RLM concept** - treats context as a variable in a REPL environment.

**What it does:**
```python
# LLM can write and execute Python code
python_exec(code="""
import re
prices = re.findall(r'\$(\d+)', CONTEXT)
avg = sum(int(p) for p in prices) / len(prices)
print(f"Average: ${avg}")
""")
→ Returns: "Average: $49.5"
```

**When to use:**
- ✅ Complex data processing (parse JSON/CSV/XML)
- ✅ Statistical analysis
- ✅ Advanced regex/text transformations
- ✅ Structured data extraction

**Security considerations:**

⚠️ **Disabled by default** (`allow_exec: False`)

Only enable when:
- You trust the input sources
- Running in isolated environment
- Need processing beyond search/read
- Context contains structured data

**Sandboxing:**
```python
namespace = {
    'CONTEXT': context,  # Pre-bound
    '__builtins__': {
        # Minimal safe builtins
        'len', 'str', 'int', 'float', 'list', 'dict', 'print'
        # NO: open, exec, eval, import, file I/O
    }
}
# + 500ms timeout, 4KB output cap, no network
```

**Note:** Basic sandboxing. Production use requires `RestrictedPython` or similar.

**Comparison:**

| Feature | context_probe | python_exec |
|---------|---------------|-------------|
| **Safety** | ✅ Safe | ⚠️ Risky |
| **Capability** | Search/read/summarize | Unlimited computation |
| **Default** | ✅ Enabled | ❌ Disabled |
| **Use case** | Text search | Data analysis |
| **Sandboxing** | Not needed | Required |

**Example flow with python_exec:**

```
User: "What's the average price mentioned?"
  ↓
LLM: "I need to extract and calculate prices"
  ↓
LLM calls: python_exec(code="""
import re
prices = re.findall(r'\$(\d+)', CONTEXT)
avg = sum(int(p) for p in prices) / len(prices)
print(f"Average: ${avg}")
""")
  ↓
Returns: "Average: $49.5"
  ↓
Scratchpad note: "Average price: $49.5"
  ↓
LLM: FINAL("The average price is $49.50")
```

**Bottom line:** Most use cases work fine with just `context_probe`. Use `python_exec` only when you need complex analysis and can accept the security tradeoffs.

### 3. Scratchpad

Accumulates atomic notes during execution:
- Deduplication by ID
- Capping at max_notes
- Source hints for citations

### 4. FINAL() Signal

The LLM emits `FINAL("answer")` when ready:
```
FINAL("The answer is X based on Y.")
```

Or reference a variable:
```
FINAL_VAR(my_answer)
```

### 5. Writer Synthesis (Optional)

If `writer_model` is configured:
1. Control model collects notes
2. Writer model synthesizes final answer
3. Includes citations from scratchpad

## Streaming

RLM emits custom stream events:

```python
async for event in agent.run_stream(
    input={'message': 'Question?'},
    context_refs=large_doc
):
    event_type = event.get('type')

    if event_type == 'data-rlm-start':
        # RLM execution started
        pass

    elif event_type == 'data-rlm-step-start':
        # Reasoning step started
        step = event['data']['step']
        budget = event['data']['budget']

    elif event_type == 'data-rlm-probe':
        # Tool being executed
        tool = event['data']['tool']
        args = event['data']['args']

    elif event_type == 'data-rlm-note':
        # Note added to scratchpad
        text = event['data']['text']

    elif event_type == 'data-rlm-final':
        # FINAL() detected
        answer = event['data']['answer']

    elif event_type == 'data-rlm-complete':
        # Execution complete
        meta = event['data']['meta']
```

## Budget Enforcement

RLM enforces hard limits:

- **Steps**: Max tool calls (12 root, 8 child default)
- **Tokens**: Total tokens across all LLM calls
- **Cost**: Estimated cost in USD

When exhausted, returns best-effort answer from scratchpad.

## Examples

### Basic Usage

```python
agent = Agent(
    id='doc-qa:v1',
    model={'provider': 'openai', 'model': 'gpt-4o-mini'},
    rlm={'enabled': True, 'depth': 1}
)

answer = await agent.run(
    input={'message': 'Summarize the key findings.'},
    context_refs='path/to/research-paper.pdf'
)
```

### With Custom Models

```python
agent = Agent(
    id='analyst:v1',
    model={'provider': 'openai', 'model': 'gpt-4o-mini'},
    rlm={
        'enabled': True,
        'depth': 1,
        'control_model': {'provider': 'openai', 'model': 'gpt-4o-mini'},
        'writer_model': {'provider': 'openai', 'model': 'gpt-4o'}
    }
)
```

### Runtime Override

```python
# Agent-level config
agent = Agent(id='agent:v1', model={...}, rlm={'enabled': False})

# Enable RLM for specific call
answer = await agent.run(
    input={'message': 'Question?'},
    context_refs=large_doc,
    rlm={'enabled': True, 'depth': 1}  # Override
)
```

## Advanced Usage

### Recursive Decomposition

Set `depth > 0` to allow recursive calls:

```python
rlm={'enabled': True, 'depth': 2}  # Two levels of recursion
```

The LLM can spawn child queries:
```
rlm_call(query="What are the system requirements?", context_slice="chunk_5")
```

### Budget Tuning

Adjust budgets based on your needs:

```python
rlm={
    'budgets': {
        'max_steps_root': 20,      # More iterations
        'max_steps_child': 5,       # Fewer for children
        'max_tokens_total': 200000, # Higher token limit
        'max_cost_usd': 1.00        # Higher cost limit
    }
}
```

### Custom Context Probing

Extend `ContextStore` for custom backends:

```python
from vel.rlm import ContextStore

class VectorSearchStore(ContextStore):
    def search(self, query, max_results=10):
        # Use vector search instead of regex
        embeddings = self.embed(query)
        results = self.vector_db.search(embeddings, k=max_results)
        return results
```

## Limitations

- **Provider support**: Works with all providers (OpenAI, Gemini, Anthropic)
- **Tool calling required**: Provider must support function calling
- **No automatic distillation**: Unlike academic ReasoningBank, requires manual strategy management
- **python_exec security**: Sandboxed but risky - disabled by default

## Best Practices

### Do

- **Start with depth=1** - Most tasks don't need deep recursion
- **Tune budgets** - Monitor costs and adjust limits
- **Use control/writer split** - Cheap model for iteration, strong for synthesis
- **Chunk appropriately** - Default 4KB chunks work well
- **Monitor events** - Use streaming to understand execution
- **Disable python_exec** - Unless you trust the input

### Don't

- Enable `allow_exec` without sandboxing
- Set unlimited budgets
- Use expensive models for control
- Ignore budget exhaustion reasons

### Optimization Tips

1. **Chunking**: Pre-chunk documents for faster probing
2. **Caching**: Cache probe results for repeated queries
3. **Summarization**: Summarize chunks before detailed analysis
4. **Early Exit**: Tune FINAL() detection for faster completion

### Cost Estimation

RLM uses conservative mid-range pricing for budget tracking:
- Prompt: $1.00 / 1M tokens
- Completion: $3.00 / 1M tokens

Actual costs vary by provider and model. Monitor `result.metadata['cost']` in production

## Troubleshooting

### RLM not activating

- Check `rlm.enabled = True`
- Ensure `context_refs` is provided
- Verify provider supports tool calling

### Budget exhausted

- Increase `max_steps` or `max_tokens`
- Reduce context size
- Lower `notes_window` to reduce context

### No FINAL() signal

- Check system prompt is correct
- Ensure budget allows enough steps
- Review LLM responses for issues

## See Also

- [RLM PRD](../features/RLM_SUPPORT.md) - Product requirements
- [Alex Zhang's RLM Blog](https://alexzhang13.github.io/blog/2025/rlm/) - Original concept
- [Examples](../examples/rlm_basic.py) - Code examples
- [Tests](../tests/test_rlm.py) - Unit tests
