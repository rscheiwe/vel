#!/bin/bash
# Basic Chat Parity Test Runner
# Usage: ./scripts/basic_chat/run_test.sh "Write a haiku about nature"

set -e  # Exit on error

# Check if prompt provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 \"Your prompt here\""
    echo "Example: $0 \"Write a haiku about nature\""
    exit 1
fi

PROMPT="$1"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
TRACES_DIR="$ROOT_DIR/traces"

# Ensure traces directory exists
mkdir -p "$TRACES_DIR"

echo "========================================"
echo "Basic Chat Parity Test"
echo "========================================"
echo "Prompt: $PROMPT"
echo ""

# Generate Vel trace
echo "📝 Generating Vel trace..."
python "$SCRIPT_DIR/vel_trace.py" --prompt "$PROMPT" > "$TRACES_DIR/vel.jsonl"
echo "✅ Vel trace saved to traces/vel.jsonl"
echo ""

# Generate AI SDK trace
echo "📝 Generating AI SDK trace..."
node "$SCRIPT_DIR/ai_sdk_trace.mjs" --prompt "$PROMPT" > "$TRACES_DIR/ai.jsonl"
echo "✅ AI SDK trace saved to traces/ai.jsonl"
echo ""

# Compare traces
echo "🔍 Comparing traces..."
echo "========================================"
python "$ROOT_DIR/scripts/compare_traces.py" "$TRACES_DIR/vel.jsonl" "$TRACES_DIR/ai.jsonl"
echo "========================================"
echo ""

# Show event counts
echo "📊 Event Summary:"
echo "Vel events:    $(wc -l < "$TRACES_DIR/vel.jsonl" | tr -d ' ')"
echo "AI SDK events: $(wc -l < "$TRACES_DIR/ai.jsonl" | tr -d ' ')"
