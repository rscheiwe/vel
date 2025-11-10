# Tool Use Behavior: `stop_on_first_tool`

## Overview

This feature adds support for halting agent execution after tool calls and returning raw tool output instead of continuing to the LLM for a final answer. This aligns with OpenAI's Agents SDK `tool_use_behavior: "stop_on_first_tool"` functionality.

## Use Cases

- **Structured data extraction** - Get JSON tool output instead of prose
- **Intent routing** - Use LLM for tool selection only, not response generation
- **Latency optimization** - Skip the final LLM call when raw data is sufficient
- **Multi-agent patterns** - One agent routes to tools, another processes results

## API

### Global Configuration

Apply to ALL tools:

```python
agent = Agent(
    id='weather-agent',
    model={'provider': 'openai', 'model': 'gpt-4o'},
    tools=['get_weather', 'send_email'],
    policies={
        'stop_on_first_tool': True  # ALL tools halt execution
    }
)

result = await agent.run({'message': "What's the weather in SF?"})
# Returns: {'city': 'San Francisco', 'temperature': 72, 'condition': 'sunny'}
# Type: Dict[str, Any]
```

### Per-Tool Configuration

Apply to specific tools:

```python
agent = Agent(
    id='assistant-agent',
    model={'provider': 'openai', 'model': 'gpt-4o'},
    tools=['get_weather', 'send_email'],
    policies={
        'tool_behavior': {
            'get_weather': {'stop_on_first_use': True},   # Halts after execution
            'send_email': {'stop_on_first_use': False}    # Continues to LLM
        }
    }
)

# get_weather returns raw output
result1 = await agent.run({'message': "Weather in NYC?"})
# Returns: {'city': 'NYC', 'temperature': 65, ...}
# Type: Dict[str, Any]

# send_email continues to LLM for natural language response
result2 = await agent.run({'message': "Send email to alice@example.com"})
# Returns: "I've sent the email to alice@example.com with subject..."
# Type: str
```

### Priority Order

1. **Per-tool behavior** (highest priority) - `tool_behavior.{tool_name}.stop_on_first_use`
2. **Global setting** - `stop_on_first_tool`
3. **Default** - `False` (normal behavior)

```python
policies={
    'stop_on_first_tool': False,  # Global default
    'tool_behavior': {
        'tool-a': {'stop_on_first_use': True}  # Override for tool-a
    }
}
# tool-a halts, all other tools continue
```

## Behavior

### Non-Streaming (`run()`)

**Normal behavior (default):**
```
1. User message
2. LLM decides to call tool
3. Tool executes → result added to context
4. LLM called again with tool result
5. LLM generates final answer
6. Return final answer (str)
```

**With `stop_on_first_tool: True`:**
```
1. User message
2. LLM decides to call tool
3. Tool executes
4. ✋ HALT - Return raw tool output (Dict[str, Any])
```

### Streaming (`run_stream()`)

**Events emitted:**

```python
# Normal sequence with stop_on_first_tool
{'type': 'start'}
{'type': 'step-start'}
{'type': 'tool-input-start', ...}
{'type': 'tool-input-available', ...}
{'type': 'tool-output-available', ...}  # Raw tool output here
{'type': 'finish-step'}
{'type': 'finish'}
# ✋ STOPS - No additional LLM call
```

The client receives `tool-output-available` event with the raw result, then execution terminates.

## Return Types

### `run()` method

Return type changed to `Union[str, Dict[str, Any]]`:

- **String** - Final answer from LLM (default behavior)
- **Dict** - Raw tool output (when `stop_on_first_tool` enabled)

```python
result = await agent.run({'message': 'query'})

if isinstance(result, str):
    print(f"LLM answer: {result}")
elif isinstance(result, dict):
    print(f"Raw tool output: {result}")
```

### `run_stream()` method

No type changes - yields stream protocol events as normal, but terminates after `tool-output-available`.

## Implementation Details

### New Method: `should_stop_after_tool(tool_name: str) -> bool`

Centralized logic for checking whether to halt after a specific tool:

```python
def should_stop_after_tool(self, tool_name: str) -> bool:
    # 1. Check per-tool behavior (highest priority)
    tool_behaviors = self.policies.get('tool_behavior', {})
    if tool_name in tool_behaviors:
        return tool_behaviors[tool_name].get('stop_on_first_use', False)

    # 2. Fall back to global setting (defaults to False)
    return self.policies.get('stop_on_first_tool', False)
```

### Changes to `run()` - Non-streaming

```python
# After tool execution
result = await self._call_tool(tool_name, args)

# NEW: Check if we should stop
if self.should_stop_after_tool(tool_name):
    return result  # Return raw tool output

# Original behavior: add to context and continue
self.ctxmgr.append_tool_result(run_id, tool_name, result, session_id)
```

### Changes to `run_stream()` - Streaming

```python
# Emit tool output
yield ToolOutputAvailableEvent(...).to_dict()

# NEW: Check if we should stop
if self.should_stop_after_tool(tool_name):
    yield {'type': 'finish-step'}
    yield {'type': 'finish'}
    return  # Don't add to context or continue loop

# Original behavior
self.ctxmgr.append_tool_result(run_id, tool_name, result, session_id)
```

## Backwards Compatibility

✅ **100% backwards compatible**

- Defaults to `False` (existing behavior)
- Existing code without policies works unchanged
- Only activates when explicitly configured
- Type change to `Union[str, Dict[str, Any]]` is additive (runtime behavior unchanged for existing code)

## Examples

See `examples/tool_use_behavior.py` for complete working examples:

1. Global `stop_on_first_tool`
2. Per-tool behavior configuration
3. Streaming with halting
4. Default behavior (no policy)

## Tests

See `tests/test_tool_use_behavior.py`:

- ✅ Global `stop_on_first_tool=True`
- ✅ Global `stop_on_first_tool=False`
- ✅ Default behavior (no policy)
- ✅ Per-tool override
- ✅ Per-tool explicit False overrides global True
- ✅ Unknown tool falls back to global setting
- ✅ Policies structure parsing

All tests pass (7/7) with no regressions in existing tests (166/166).

## OpenAI API Behavior

**Important:** OpenAI's API is stateless. When the model returns `finish_reason="tool_calls"`:

- You can choose to stop (this feature)
- OR continue by making another API call with tool results

The API doesn't "expect" tool results - it only expects them **if you choose to continue** the conversation. This feature implements the "choose to stop" path.

## Future Enhancements

Potential additions:

- `stop_after_n_tools` - Halt after N tool executions (not just first)
- `tool_use_strategy` - "parallel", "sequential", "stop_on_first"
- Per-tool timeout/retry policies
- Tool result transformers

---

**Branch:** `feature/tool-use-behavior`
**Status:** ✅ Ready for review
**Tests:** 7/7 passing, 166/166 existing tests still passing
