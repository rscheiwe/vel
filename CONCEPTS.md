# Vel Concepts & Terminology

Business definitions and domain concepts for the Vel agent runtime. No code here—see implementation files for details.

> **Last Updated:** 2025-01-08

---

## Contents

- [Core Concepts](#core-concepts)
- [Execution Model](#execution-model)
- [Memory Concepts](#memory-concepts)
- [Streaming Concepts](#streaming-concepts)
- [Provider Concepts](#provider-concepts)
- [Tool Concepts](#tool-concepts)
- [Concept Relationships](#concept-relationships)

---

## Core Concepts

### Agent

The main orchestrator that coordinates LLM calls, tool execution, and state management.

| Property | Description |
|----------|-------------|
| **Identity** | `id` - Unique identifier for the agent |
| **Model** | `provider` + `model` - Which LLM to use |
| **Tools** | List of `ToolSpec` - Capabilities available |
| **Policies** | Configuration for behavior (max_steps, tool_use_behavior) |

**NOT an Agent:**
- Raw LLM API calls (no orchestration)
- Single-turn chat completions (no state)
- Function calling without iteration (no loop)

### Run

A single execution of an agent from input to output.

| Property | Description |
|----------|-------------|
| **run_id** | Unique identifier for this execution |
| **session_id** | Groups related runs (for persistence) |
| **input** | User message + optional context |
| **output** | Final response (text or structured) |

### Step

One iteration of the agent loop (LLM call → tool execution → state update).

| Property | Description |
|----------|-------------|
| **step_number** | Sequential counter (1, 2, 3...) |
| **messages** | Context sent to LLM |
| **response** | LLM output (text + tool calls) |
| **tool_results** | Outputs from executed tools |

### State

Immutable snapshot of agent execution at any point.

| Property | Description |
|----------|-------------|
| **messages** | Conversation history |
| **step** | Current step number |
| **tool_calls** | Pending tool invocations |
| **finished** | Whether execution is complete |

---

## Execution Model

### Reducer Pattern

Pure function: `(State, Event) -> (State, Effects)`

| Concept | Description |
|---------|-------------|
| **State** | Immutable execution snapshot |
| **Event** | Something that happened (user input, LLM response, tool result) |
| **Effect** | Command describing side effect (emit, call_tool, call_llm, halt) |

**Why this pattern:**
- Testability (pure functions)
- Reproducibility (same input → same output)
- Debugging (state transitions are explicit)

### Effects

Commands produced by the reducer, executed by the runtime.

| Effect | Description |
|--------|-------------|
| `emit` | Yield event to stream |
| `call_tool` | Execute a tool |
| `call_llm` | Call the language model |
| `checkpoint` | Save current state |
| `halt` | Stop execution |

### Tool Use Behavior

What happens after tool execution.

| Behavior | Description |
|----------|-------------|
| `RUN_LLM_AGAIN` | Default - call LLM with tool results |
| `STOP_AFTER_TOOL` | Return tool output as final answer |
| `STOP_AT_TOOLS` | Stop only for specific tools |
| `CUSTOM_HANDLER` | User-defined decision logic |

---

## Memory Concepts

### Message History

Conversation turns within a run. Always active.

| Property | Description |
|----------|-------------|
| **Grain** | Per-message |
| **Lifetime** | Single run (transient) or session (persistent) |
| **Storage** | In-memory or database |

### FactStore

Long-term key-value storage. Opt-in.

| Property | Description |
|----------|-------------|
| **Grain** | Per-key within namespace |
| **Lifetime** | Persistent across runs |
| **Storage** | SQLite database |
| **Use Cases** | User preferences, domain knowledge, configuration |

**Namespace Convention:**
- `user:{id}` - User-specific facts
- `session:{id}` - Session-specific facts
- `global` - Application-wide facts

### ReasoningBank

Strategy-level memory with embeddings. Opt-in.

| Property | Description |
|----------|-------------|
| **Grain** | Per-strategy |
| **Lifetime** | Persistent, with confidence decay |
| **Storage** | SQLite + embeddings |
| **Use Cases** | Learned behaviors, successful patterns |

**Confidence Model:**
- Success multiplier: 1.20x
- Failure multiplier: 0.85x
- Initial confidence: 0.5
- Decay: 0.02 per 30 days unused

### Trajectory

Complete execution trace of a run.

| Property | Description |
|----------|-------------|
| **Grain** | Per-run |
| **Contents** | Messages, tool calls, timing, success/failure |
| **Use Cases** | Learning, debugging, analytics |

---

## Streaming Concepts

### Stream Protocol

Vercel AI SDK V5 compatible event format.

| Concept | Description |
|---------|-------------|
| **Event** | Atomic unit of stream (has `type` + payload) |
| **Delta** | Incremental content (text chunk, JSON fragment) |
| **Finish** | End-of-stream marker with reason |

### Event Types

| Category | Events |
|----------|--------|
| **Text** | `text-start`, `text-delta`, `text-end` |
| **Reasoning** | `reasoning-start`, `reasoning-delta`, `reasoning-end` |
| **Tools** | `tool-input-start`, `tool-input-delta`, `tool-input-available`, `tool-output-available` |
| **Control** | `start`, `start-step`, `finish-step`, `finish-message`, `finish` |
| **Error** | `error` |

### Translator

Converts provider-native events to Vel events.

| Property | Description |
|----------|-------------|
| **Input** | Provider-specific chunk format |
| **Output** | Vel stream events |
| **State** | Tracks partial tool calls, text blocks |

---

## Provider Concepts

### Provider

Adapter for a specific LLM API (OpenAI, Anthropic, Gemini).

| Method | Description |
|--------|-------------|
| `stream()` | Async generator of events |
| `generate()` | Non-streaming complete response |

### Model Configuration

How to specify which LLM to use.

```python
model = {
    'provider': 'openai',      # Required: provider name
    'model': 'gpt-4o',         # Required: model ID
    'api_key': 'sk-...'        # Optional: override env var
}
```

### Message Translation

Converting Vel's internal format to provider-specific format.

| Concern | Description |
|---------|-------------|
| **Roles** | Mapping user/assistant/system/tool |
| **Content** | Text, tool calls, tool results |
| **Metadata** | Thinking blocks, usage stats |

---

## Tool Concepts

### ToolSpec

Definition of a tool's interface and implementation.

| Property | Description |
|----------|-------------|
| **name** | Unique identifier |
| **input_schema** | JSON Schema for parameters |
| **output_schema** | JSON Schema for return value |
| **handler** | Function to execute |

### Schema Generation

Automatic JSON Schema from Python type hints.

| Type Hint | JSON Schema |
|-----------|-------------|
| `str` | `{"type": "string"}` |
| `int` | `{"type": "integer"}` |
| `Optional[T]` | T schema + nullable |
| `List[T]` | `{"type": "array", "items": T}` |
| `Literal["a", "b"]` | `{"enum": ["a", "b"]}` |
| `BaseModel` | Full object schema |

### Tool Context

Runtime information passed to tool handlers.

| Property | Description |
|----------|-------------|
| `run_id` | Current run identifier |
| `session_id` | Session identifier |
| `agent_id` | Agent identifier |
| `step` | Current step number |

**Access via:** Parameter named `_context`, `ctx`, or `context` (filtered from schema).

---

## Concept Relationships

### Agent → Run → Step

```
Agent (definition)
  └── Run (execution instance)
       ├── Step 1
       │    ├── LLM Call
       │    ├── Tool Execution
       │    └── State Update
       ├── Step 2
       │    └── ...
       └── Step N
            └── Finish
```

### Memory Hierarchy

```
Message History (always, per-run)
  ↓
FactStore (opt-in, persistent)
  ↓
ReasoningBank (opt-in, with learning)
  ↓
AutoLearning (Phase 2, background)
```

### Event Flow

```
Provider Stream
  ↓ (Translator)
Vel Events
  ↓ (Agent)
Client Stream
```

### State Flow

```
Input → Reducer → (State, Effects) → Runtime → State'
         ↑                               │
         └───────────────────────────────┘
```

---

## Disambiguation

Common confusions and their resolutions.

### Run vs Session

| Concept | Scope | Lifetime |
|---------|-------|----------|
| **Run** | Single input→output | Ephemeral |
| **Session** | Multiple runs | Persistent (if configured) |

### ToolSpec vs Tool Registry

| Approach | Description | Recommendation |
|----------|-------------|----------------|
| **ToolSpec** | Direct object, no registration | Preferred |
| **Registry** | Global lookup by string name | Deprecated |

### Transient vs Persistent

| Mode | Message Storage | Use Case |
|------|-----------------|----------|
| **Transient** | In-memory only | Stateless APIs |
| **Persistent** | Database | Chat applications |

### Reducer vs Agent

| Concept | Role |
|---------|------|
| **Reducer** | Pure state transition logic |
| **Agent** | Orchestrator that runs the loop |

The reducer demonstrates the pattern; the Agent implements it pragmatically.
