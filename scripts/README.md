# Stream Protocol Parity Testing

Test Vel's parity with Vercel AI SDK V5 by comparing trace outputs.

Tests are organized by scenario type:
- **basic_chat/** - Text-only generation (no tools, no reasoning)
- More test types can be added as needed (tools, reasoning, multi-modal, etc.)

## Setup

**Note:** All commands should be run from the repository root directory (`vel/`).

```bash
# Create traces directory
mkdir -p traces

# Install Node dependencies for AI SDK trace scripts
cd scripts && npm install && cd ..

# Set your OpenAI API key (choose one method):

# Method 1: Export (Unix/Linux/macOS)
export OPENAI_API_KEY=sk-...

# Method 2: Inline per-command (Unix/Linux/macOS)
# OPENAI_API_KEY=sk-... node script.js

# Method 3: Windows cmd
# set OPENAI_API_KEY=sk-...

# Method 4: Windows PowerShell
# $env:OPENAI_API_KEY="sk-..."
```

## Quick Start

### Basic Chat (Text-only)

Run the automated test script:

```bash
./scripts/basic_chat/run_test.sh "Write a haiku about nature"
```

This will:
1. Generate Vel trace → `traces/vel.jsonl`
2. Generate AI SDK trace → `traces/ai.jsonl`
3. Compare and report parity status

Expected output:
```
✅ Event type order matches
✅ finish-step.finishReason present in both
✅ finish-step.usage present in both
✅ finish-step.response present in both
✅ finish.finishReason present in both
✅ finish.totalUsage present in both
✅ Parity
```

### Manual Usage

If you prefer to run steps individually:

```bash
# 1. Generate Vel trace
python scripts/basic_chat/vel_trace.py --prompt "Write a haiku" > traces/vel.jsonl

# 2. Generate AI SDK trace
node scripts/basic_chat/ai_sdk_trace.mjs --prompt "Write a haiku" > traces/ai.jsonl

# 3. Compare
python scripts/compare_traces.py traces/vel.jsonl traces/ai.jsonl
```

## Test Scenarios

### Basic Chat - Simple Text Generation

```bash
./scripts/basic_chat/run_test.sh "Hello, how are you?"
```

### Basic Chat - Short Response

```bash
./scripts/basic_chat/run_test.sh "Say hello"
```

### Basic Chat - Creative Writing

```bash
./scripts/basic_chat/run_test.sh "Write a haiku about programming"
```

### Tool Calls (Coming Soon)

Create `scripts/tools/` directory with trace generators that include tool definitions.

### Reasoning Models (Coming Soon)

Create `scripts/reasoning/` directory for o1/o3 model testing with reasoning events.

## Comparison Logic

The `compare_traces.py` script:

1. **Squashes deltas**: Concatenates consecutive `text-delta` and `reasoning-delta` events
2. **Compares event order**: Ensures event types appear in the same sequence
3. **Compares content**: Validates concatenated text/reasoning matches
4. **Compares tool calls**: Validates tool names, inputs, and outputs
5. **Allows optional events**: `finish-message`, `response-metadata`, `source` are optional

## Trace Format

Both scripts output JSONL with AI SDK V5 stream protocol event structure:

```jsonl
{"type": "text-start", "id": "0"}
{"type": "text-delta", "id": "0", "delta": "Hello"}
{"type": "text-delta", "id": "0", "delta": " world"}
{"type": "text-end", "id": "0"}
{"type": "finish-step", "finishReason": "stop", "usage": {...}, "response": {...}}
{"type": "finish", "finishReason": "stop", "totalUsage": {...}}
```

**Key Fields:**
- `type` - Event type (text-start, text-delta, tool-input-available, finish-step, etc.)
- `id` - Block identifier
- `delta` - Delta content (text/reasoning)
- `toolName` - Tool name (for tool events)
- `input` - Tool input (for tool-input-available)
- `output` - Tool output (for tool-output-available)
- `finishReason` - Completion reason (stop, length, tool_calls, error, etc.)
- `usage` - Token usage statistics
- `response` - Response metadata (id, modelId, etc.)

## Debugging Failed Comparisons

If comparison fails:

1. **Check event types:**
   ```bash
   # Extract just event types
   jq -r '.type' < traces/vel.jsonl
   jq -r '.type' < traces/ai.jsonl
   ```

2. **Check concatenated text:**
   ```bash
   # Concatenate all text deltas
   jq -r 'select(.type == "text-delta") | .delta' < traces/vel.jsonl | tr -d '\n'
   ```

3. **Inspect full events:**
   ```bash
   cat traces/vel.jsonl | jq .
   cat traces/ai.jsonl | jq .
   ```

4. **Compare side-by-side:**
   ```bash
   diff -u traces/vel.jsonl traces/ai.jsonl
   ```

## Known Acceptable Differences

The comparison script allows these differences:

1. **Metadata events**: May be emitted at different times (early vs late)
2. **finish-message**: Optional in both traces
3. **Delta granularity**: Text may be chunked differently (compared after concatenation)
4. **Block IDs**: Different ID values are OK as long as structure matches

## Files

```
scripts/
├── basic_chat/              # Basic chat test scenario (text-only)
│   ├── run_test.sh          # Automated test runner
│   ├── vel_trace.py         # Vel trace generator
│   └── ai_sdk_trace.mjs     # AI SDK trace generator
├── compare_traces.py        # Trace comparison tool
├── package.json             # Node.js dependencies for AI SDK scripts
└── README.md                # This file
```

**Planned structure for additional scenarios:**
```
scripts/
├── basic_chat/              # Text-only generation
├── tools/                   # Tool calling tests (coming soon)
├── reasoning/               # o1/o3 reasoning models (coming soon)
└── multimodal/              # Vision/audio tests (coming soon)
```