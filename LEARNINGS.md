# Vel Self-Evolution Log

This codebase learns from experience. Each mistake becomes a permanent lesson.

> **Last Updated:** 2025-01-08

---

## Contents

- [How This Works](#how-this-works)
- [Meta-Rules for Writing Entries](#meta-rules-for-writing-entries)
- [Topic Index](#topic-index)
- [Learnings](#learnings)
- [Anti-Patterns](#anti-patterns)
- [Quick Reference Tables](#quick-reference-tables)

---

## How This Works

When you make a mistake or discover a gap, trigger self-improvement with:

> "Reflect on this mistake. Abstract the general pattern. Update the appropriate file and log in LEARNINGS.md."

This orchestrates a complete learning cycle:
1. **Reflect** - Analyze root cause while context is fresh
2. **Abstract** - Extract general principle from specific instance
3. **Generalize** - Create reusable decision framework
4. **Document** - Update skill files + log here

### File Routing

| Learning Type | Target File |
|--------------|-------------|
| Core architecture decision | `.claude/adr/` |
| Provider-specific behavior | `LEARNINGS.md` (here) |
| Python coding pattern | `.claude/rules/python-standards.md` |
| Testing pattern | `.claude/rules/testing.md` |
| Tool development | `.claude/rules/tools-development.md` |
| Memory system | `.claude/rules/memory-system.md` |
| Provider development | `.claude/rules/provider-development.md` |
| Multi-step workflow | `.claude/recipes/` |

---

## Meta-Rules for Writing Entries

These rules ensure quality compounds as this file grows:

1. **Use absolute directives** - Start with NEVER/ALWAYS when appropriate
2. **Lead with rationale** - Explain the problem before the solution (1-3 bullets max)
3. **Include concrete examples** - Actual code showing correct vs incorrect
4. **Minimize bloat** - One clear point per entry
5. **Link to files updated** - Show where the fix was applied
6. **Date entries** - Track when learning was captured

### Entry Template

```markdown
### YYYY-MM-DD: [Short Title]

**Problem:** [What went wrong]
**Root Cause:** [Why it happened]
**Pattern:** [General principle - use NEVER/ALWAYS]

**Fix Applied:**
- [File] - [Change made]

**Example:**
```python
# CORRECT
...

# WRONG
...
```
```

---

## Topic Index

Jump to related learnings by topic. Content lives in the entries below—this is just navigation.

| Topic | Entries |
|-------|---------|
| **Providers** | OpenAI Gotchas, Anthropic Gotchas, Gemini Gotchas |
| **Streaming** | Event Ordering, Text Streaming, Error Handling |
| **Memory** | FactStore, ReasoningBank, AutoLearning |
| **Testing** | Mock Embeddings, Async Isolation, Provider Mocking |
| **Performance** | Context Window, Memory Operations, Latency |
| **Debugging** | Tool Issues, Stream Hangs, Memory Persistence |
| **Architecture** | Translator Complexity, Reducer Design, Memory Opt-In |

---

## Learnings

### Provider-Specific Gotchas

#### OpenAI

| Issue | Pattern | Solution |
|-------|---------|----------|
| Tool calls split across chunks | ALWAYS buffer tool deltas | Emit `tool-input-available` only when complete |
| Rate limiting on streaming | NEVER retry mid-stream | Implement exponential backoff |
| `finish_reason` missing on some chunks | ONLY check on final chunk | Guard with `if chunk.choices[0].finish_reason` |

#### Anthropic

| Issue | Pattern | Solution |
|-------|---------|----------|
| Content blocks indexed differently | ALWAYS track `content_block_index` | Separate from event order |
| Thinking blocks have different structure | ALWAYS check block type first | Check for `thinking` type before processing as text |
| Tool use requires explicit stop reason | Handle `tool_use` as valid completion | Not an error condition |

#### Gemini

| Issue | Pattern | Solution |
|-------|---------|----------|
| No native tool streaming | ALWAYS buffer entire tool call | Emit as single event |
| Safety ratings can block responses | ALWAYS handle `SAFETY` finish reason | Graceful degradation |
| Multimodal content structure differs | ALWAYS normalize `parts` array | Convert to text content |

---

### Memory System

#### FactStore

| Issue | Pattern | Solution |
|-------|---------|----------|
| Namespace collision | ALWAYS use hierarchical namespaces | `user:alice`, `session:123` |
| JSON serialization fails | NEVER store non-serializable types | No functions, classes, circular refs |
| Concurrent write races | Use SQLite WAL mode | Helps but doesn't prevent all races |

#### ReasoningBank

| Issue | Pattern | Solution |
|-------|---------|----------|
| Strategy quality varies | ALWAYS prefer generalizable strategies | Specific strategies underperform |
| Unused strategies decay | Schedule consolidation in production | Prevents strategy rot |
| Embedding dimension mismatch | ALWAYS validate dimensions on insert | All embeddings must match |

#### AutoLearning

| Issue | Pattern | Solution |
|-------|---------|----------|
| Cold start performance | Seed strategies before production | Significantly improves initial runs |
| Evaluation costs | Use cheap models for judging | gpt-4o-mini, claude-3-haiku |
| Extraction timing | Extract AFTER success | Not during execution |

---

### Streaming Patterns

#### Event Ordering (Tool Calls)

```python
# CORRECT sequence
1. tool-input-start      # tool name known
2. tool-input-delta      # streaming input JSON
3. tool-input-available  # input complete, execution starts
4. tool-output-available # execution complete

# WRONG - emitting out of order
yield tool-input-available  # Before deltas complete!
```

#### Text Streaming

| Rule | Rationale |
|------|-----------|
| ALWAYS emit `text-start` before first `text-delta` | Client needs to know text block began |
| ALWAYS emit `text-end` when block completes | Client needs to know text block ended |
| Multiple text blocks are valid | Interleaved with tool calls |

#### Error Handling

| Rule | Rationale |
|------|-----------|
| ALWAYS emit `error` event before raising | Client gets notification |
| ALWAYS include error code | Client can categorize |
| NEVER swallow errors in translation | Debugging becomes impossible |

---

### Testing Patterns

#### Mock Embeddings

```python
# CORRECT - deterministic, reproducible
def mock_embed(text: str) -> List[float]:
    h = hashlib.sha256(text.encode()).digest()
    return [b / 255.0 for b in h[:128]]

# WRONG - non-deterministic
def mock_embed(text: str) -> List[float]:
    return [random.random() for _ in range(128)]
```

#### Async Test Isolation

| Rule | Rationale |
|------|-----------|
| ALWAYS use `tmp_path` fixture for databases | Isolation between tests |
| ALWAYS clean up async resources in `finally` | Prevent resource leaks |
| ALWAYS use `asyncio.wait_for()` for timeouts | Prevent hanging tests |

#### Provider Mocking

| Level | When to Use |
|-------|-------------|
| HTTP level | Integration tests |
| Provider level | Unit tests |
| NEVER mock translator | Must test with real provider responses |

---

### Performance Observations

#### Context Window Management

| Observation | Recommendation |
|-------------|----------------|
| 70% utilization is optimal | Quality degrades beyond 75% |
| Truncation loses information | Summarize old messages instead |
| Tool outputs are large | Compress or summarize before returning |

#### Memory Operations

| Observation | Recommendation |
|-------------|----------------|
| Embedding computation is expensive | Batch when possible |
| Database I/O blocks | Use async for all DB operations |
| Consolidation is expensive | Run during low-traffic periods |

#### Streaming Latency

| Observation | Recommendation |
|-------------|----------------|
| First token latency matters most | Optimize for TTFT (Time To First Token) |
| Small deltas add overhead | Buffer deltas < 5 chars |
| `text-start` provides immediate feedback | Emit as soon as text block begins |

---

### Common Debugging Strategies

#### "Tool not called" Issues

```
1. Check tool schema generation (missing type hints?)
2. Verify tool in agent's tool list
3. Check guardrails not blocking
4. Examine system prompt for conflicting instructions
```

#### "Stream hangs" Issues

```
1. Check for unhandled exceptions in provider
2. Verify async iteration (not blocking on list())
3. Check for missing yield statements
4. Examine translator state machine
```

#### "Memory not persisting" Issues

```
1. Verify memory mode enabled (VEL_MEMORY_MODE)
2. Check database path exists and writable
3. Confirm session_id consistency
4. Check for transaction rollbacks
```

---

## Anti-Patterns

Patterns that seem reasonable but cause problems.

### Provider Development

| Anti-Pattern | Why It Fails | Do Instead |
|--------------|--------------|------------|
| Sharing translator state between streams | Race conditions | Create new translator per stream |
| Catching all exceptions silently | Hides bugs | Catch specific, log, re-raise |
| Guessing provider format from docs | Docs are often outdated | Test with real API responses |

### Memory System

| Anti-Pattern | Why It Fails | Do Instead |
|--------------|--------------|------------|
| Storing large objects in FactStore | Slows queries | Store references, fetch on demand |
| High-confidence seed strategies | Blocks learning | Start seeds at 0.5 confidence |
| Synchronous embedding computation | Blocks event loop | Use async throughout |

### Testing

| Anti-Pattern | Why It Fails | Do Instead |
|--------------|--------------|------------|
| Testing with live APIs | Flaky, slow, expensive | Mock at appropriate level |
| Sharing database between tests | Test pollution | Use `tmp_path` fixture |
| Asserting exact event counts | Brittle to changes | Assert event types present |

---

## Quick Reference Tables

### Streaming Event Types

| Event | When Emitted | Required Fields |
|-------|--------------|-----------------|
| `text-start` | Text block begins | - |
| `text-delta` | Text content available | `delta` |
| `text-end` | Text block completes | - |
| `tool-input-start` | Tool call begins | `tool_name` |
| `tool-input-delta` | Tool input streaming | `delta` |
| `tool-input-available` | Tool input complete | `tool_name`, `input` |
| `tool-output-available` | Tool execution done | `output` |
| `finish-message` | Response complete | `finish_reason` |
| `error` | Error occurred | `message`, `code` |

### Architecture Insights

| Question | Answer | Reference |
|----------|--------|-----------|
| Why is `translators.py` so large? | Each provider has unique streaming semantics | ADR-003 |
| Why is the reducer minimal? | Pragmatic balance of pattern and simplicity | ADR-002 |
| Why is memory opt-in? | Avoids complexity for simple agents | ADR-004 |

---

## Adding New Learnings

When you discover a new pattern or gotcha:

1. **Use the trigger prompt** above to reflect
2. **Choose the right file** from the routing table
3. **Follow the entry template**
4. **Update the topic index** if adding a new topic
5. **Add to anti-patterns** if it's a common mistake
