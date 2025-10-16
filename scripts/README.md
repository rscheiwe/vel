# Stream Protocol Parity Testing

Test Vel's parity with Vercel AI SDK V5 by comparing trace outputs.

## Setup

```bash
# Create traces directory
mkdir -p traces

# Install Node dependencies for AI SDK trace script
npm install @ai-sdk/openai ai yargs

# Ensure OPENAI_API_KEY is set
export OPENAI_API_KEY=sk-...
```

## Usage

### 1) Produce Vel trace

**Chat Completions API:**
```bash
PYTHONPATH=. python scripts/vel_trace.py --prompt "Write a haiku" > traces/vel.jsonl
```

**Responses API (o1/o3 with reasoning):**
```bash
PYTHONPATH=. VEL_RESPONSES=1 python scripts/vel_trace.py --prompt "What is 13^2 + 7?" --model "o1" > traces/vel_responses.jsonl
```

### 2) Produce AI-SDK trace (Node)

**Chat Completions API:**
```bash
node scripts/ai_sdk_trace.mjs --prompt "Write a haiku" > traces/ai.jsonl
```

**Note:** The AI SDK trace script can be configured for Responses API by modifying the model provider in the script.

### 3) Compare

```bash
python scripts/compare_traces.py traces/vel.jsonl traces/ai.jsonl
```

Expected output:
```
✅ Parity
```

## Test Scenarios

### Simple Text Generation

```bash
# Vel
PYTHONPATH=. python scripts/vel_trace.py --prompt "Hello, how are you?" > traces/vel_text.jsonl

# AI SDK
node scripts/ai_sdk_trace.mjs --prompt "Hello, how are you?" > traces/ai_text.jsonl

# Compare
python scripts/compare_traces.py traces/vel_text.jsonl traces/ai_text.jsonl
```

### Tool Calls (requires tool setup)

Add tools to both scripts, then:

```bash
# Vel
PYTHONPATH=. python scripts/vel_trace.py --prompt "What's the weather in SF?" > traces/vel_tools.jsonl

# AI SDK
node scripts/ai_sdk_trace.mjs --prompt "What's the weather in SF?" > traces/ai_tools.jsonl

# Compare
python scripts/compare_traces.py traces/vel_tools.jsonl traces/ai_tools.jsonl
```

### Reasoning Models (o1/o3)

```bash
# Vel
PYTHONPATH=. VEL_RESPONSES=1 python scripts/vel_trace.py --prompt "What is the square root of 169?" --model "o1-mini" > traces/vel_reasoning.jsonl

# AI SDK (modify script for responses model)
# Compare reasoning-start, reasoning-delta, reasoning-end events
```

## Comparison Logic

The `compare_traces.py` script:

1. **Squashes deltas**: Concatenates consecutive `text-delta` and `reasoning-delta` events
2. **Compares event order**: Ensures event types appear in the same sequence
3. **Compares content**: Validates concatenated text/reasoning matches
4. **Compares tool calls**: Validates tool names, inputs, and outputs
5. **Allows optional events**: `finish-message`, `response-metadata`, `source` are optional

## Normalized Trace Format

Both scripts output JSONL with normalized event structure:

```jsonl
{"t": "text-start", "id": "block_1"}
{"t": "text-delta", "id": "block_1", "d": "Hello"}
{"t": "text-delta", "id": "block_1", "d": " world"}
{"t": "text-end", "id": "block_1"}
```

**Field Mapping:**
- `t` - Event type
- `id` - Block identifier
- `d` - Delta content (text/reasoning)
- `tool` - Tool name (for tool events)
- `args` - Tool input (for tool-input-available)
- `out` - Tool output (for tool-output-available)

## Debugging Failed Comparisons

If comparison fails:

1. **Check event types:**
   ```bash
   # Extract just event types
   jq -r '.t' < traces/vel.jsonl
   jq -r '.t' < traces/ai.jsonl
   ```

2. **Check concatenated text:**
   ```bash
   # Concatenate all text deltas
   jq -r 'select(.t == "text-delta") | .d' < traces/vel.jsonl | tr -d '\n'
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

- `scripts/vel_trace.py` - Vel translator trace generator
- `scripts/ai_sdk_trace.mjs` - AI SDK trace generator
- `scripts/compare_traces.py` - Trace comparison tool
- `scripts/README.md` - This file