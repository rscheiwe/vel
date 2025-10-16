# Basic Chat Parity Tests

Text-only generation tests (no tools, no reasoning).

## Usage

```bash
# From repository root
./scripts/basic_chat/run_test.sh "Your prompt here"
```

## What It Does

1. Generates Vel trace using `vel_trace.py`
2. Generates AI SDK V5 trace using `ai_sdk_trace.mjs`
3. Compares traces using `compare_traces.py`
4. Reports parity status

## Files

- `run_test.sh` - Automated test runner
- `vel_trace.py` - Generates trace from Vel Agent
- `ai_sdk_trace.mjs` - Generates trace from AI SDK `streamText()`

## Example

```bash
./scripts/basic_chat/run_test.sh "Write a haiku about nature"
```

Output traces saved to:
- `traces/vel.jsonl`
- `traces/ai.jsonl`
