# Vel Examples

This directory contains examples demonstrating various Vel features.

## Quick Start Examples

These examples show core functionality with provider interchange and MessageReducer integration:

### Basic Usage
- **`basic_streaming.py`** - Simple streaming text responses
- **`basic_nonstreaming.py`** - Simple non-streaming responses

### Tool Usage
- **`tool_streaming.py`** - Streaming with tool calls
- **`tool_nonstreaming.py`** - Non-streaming with tool calls

### Multi-Step Reasoning
- **`multi_step_streaming.py`** - Multi-step agent with multiple tool calls (streaming)
- **`multi_step_nonstreaming.py`** - Multi-step agent (non-streaming)

### Reasoning/Thinking Models
- **`reasoning_streaming.py`** - OpenAI o1 reasoning or Claude thinking (streaming)
- **`reasoning_nonstreaming.py`** - Reasoning models (non-streaming)

## Running Examples

All examples support provider interchange via the `PROVIDER` configuration variable:

```python
# Change this to test different providers
PROVIDER = 'openai'  # Options: 'openai', 'anthropic', 'gemini'
```

### Set up API keys:

```bash
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
export GOOGLE_API_KEY="..."
```

### Run an example:

```bash
python examples/basic_streaming.py
python examples/tool_streaming.py
python examples/reasoning_streaming.py
```

## Features Demonstrated

Each example demonstrates:
- ✅ **MessageReducer** - Aggregates streaming events into AI SDK-compatible messages
- ✅ **Provider Interchange** - Switch between OpenAI, Anthropic, and Gemini
- ✅ **Streaming vs Non-Streaming** - Compare both execution modes
- ✅ **Real-time Event Handling** - See events as they arrive
- ✅ **Message Structure** - View final AI SDK format for database storage

## Provider-Specific Notes

### OpenAI
- Basic, tool, multi-step: Use standard models (gpt-4o, gpt-4o-mini)
- Reasoning: Use `openai-responses` provider with o1/o3 models
- Note: o1/o3 reasoning content is encrypted

### Anthropic Claude
- Basic, tool, multi-step: Use standard models (claude-sonnet-4)
- Reasoning: Enable extended thinking via `generation_config`:
  ```python
  generation_config={'thinking': {'type': 'enabled', 'budget_tokens': 5000}}
  ```
- Note: Thinking content is visible

### Google Gemini
- Basic, tool, multi-step: Use standard models (gemini-1.5-flash)
- Reasoning: No dedicated reasoning mode currently

## Archive

The `archive/` folder contains older examples for reference. These examples demonstrate additional features but may not follow the current best practices.

## Learn More

- [Vel Documentation](https://rscheiwe.github.io/vel)
- [Stream Protocol](https://rscheiwe.github.io/vel/stream-protocol)
- [MessageReducer Guide](https://rscheiwe.github.io/vel/stream-protocol#message-aggregation)
