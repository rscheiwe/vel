---
layout: default
title: Using Translators Directly
parent: Event Translators
nav_order: 1
---

# Using Translators Directly

This guide explains how to use Vel's translators for custom orchestration while maintaining compatibility with AI SDK frontend components.

## Architecture: Why Two Layers?

Vel separates protocol translation from orchestration based on the **Single Responsibility Principle**:

### Translator: Protocol Adapter Pattern

- **Job:** Convert provider-specific → standard protocol
- **Scope:** Single LLM response stream
- **Stateful:** Only tracks current response (text blocks, tool calls)
- **Reusable:** Can be used in any orchestrator (Vel Agent, Mesh, LangGraph, custom)

### Agent: Orchestrator Pattern

- **Job:** Multi-step execution, tool calling, context management
- **Scope:** Full agentic workflow
- **Stateful:** Sessions, context, run history
- **Opinionated:** Implements specific orchestration pattern

**Why This Matters:** This separation enables **composability**. You can use Vel's translators in any orchestration framework without adopting Vel's Agent pattern. Each layer does one thing well.

## Understanding the Gap

Vel translators convert provider-specific events to standardized Vel protocol events, but they only handle **content-level events** from a single LLM response:

### What Translators Emit ✅

```python
translator = OpenAIAPITranslator()

async for chunk in openai_stream:
    event = translator.translate_chunk(chunk)
    # Emits:
    # - text-start, text-delta, text-end
    # - tool-input-start, tool-input-delta, tool-input-available
    # - response-metadata
    # - finish-message
```

### What Translators DON'T Emit ❌

- `start` - Generation begins
- `start-step` - Step begins (multi-step agents)
- `finish-step` - Step complete with metadata
- `finish` - Full generation complete
- `tool-output-available` - Tool execution results

## Why This Matters for Frontend

AI SDK frontend components (React, Vue, Svelte hooks) expect the **full event stream**:

```typescript
// Frontend expects this:
useChat({
  api: '/api/chat',
  // Needs: start, start-step, text-delta, finish-step, finish
});
```

If you use translator directly without filling gaps, frontend components will:
- ❌ Not detect generation start
- ❌ Not track usage/metadata properly
- ❌ Not show tool execution results
- ❌ Not detect completion correctly

## Solution: Manual Event Orchestration

Here's how to bridge the gap:

### Example 1: Single-Step Text Generation

```python
from vel.providers.translators import OpenAIAPITranslator
from openai import AsyncOpenAI

async def stream_with_orchestration(prompt: str):
    """
    Use translator with manual orchestration.
    Compatible with AI SDK frontend components.
    """
    client = AsyncOpenAI()
    translator = OpenAIAPITranslator()

    # 1. Emit start event
    yield {'type': 'start'}

    # 2. Emit start-step
    yield {'type': 'start-step'}

    # 3. Stream LLM response through translator
    usage = None
    response_id = None
    model_id = None
    finish_reason = 'stop'

    stream = await client.chat.completions.create(
        model='gpt-4o',
        messages=[{'role': 'user', 'content': prompt}],
        stream=True,
        stream_options={'include_usage': True}
    )

    async for chunk in stream:
        chunk_dict = chunk.model_dump()

        # Translate to Vel event
        event = translator.translate_chunk(chunk_dict)

        if event:
            # Track metadata (don't forward)
            if event.type == 'response-metadata':
                response_id = event.id
                model_id = event.model_id
                usage = event.usage
                continue  # Don't emit

            # Track finish reason (don't forward)
            if event.type == 'finish-message':
                finish_reason = event.finish_reason
                continue  # Don't emit

            # Forward content events
            yield event.to_dict()

    # 4. Finalize any pending events
    for pending_event in translator.finalize_tool_calls():
        yield pending_event.to_dict()

    # 5. Emit finish-step with metadata
    yield {
        'type': 'finish-step',
        'finishReason': finish_reason,
        'usage': usage,
        'response': {
            'id': response_id,
            'modelId': model_id
        }
    }

    # 6. Emit finish
    yield {
        'type': 'finish',
        'finishReason': finish_reason,
        'totalUsage': usage
    }

# FastAPI endpoint
@app.post('/api/chat')
async def chat_endpoint(request: ChatRequest):
    return StreamingResponse(
        stream_with_orchestration(request.message),
        media_type='text/event-stream'
    )
```

### Example 2: Multi-Step with Tool Calling

```python
async def multi_step_stream(prompt: str, tools: dict):
    """
    Multi-step orchestration with tool execution.
    Fully compatible with AI SDK frontend.
    """
    client = AsyncOpenAI()
    translator = OpenAIAPITranslator()

    messages = [{'role': 'user', 'content': prompt}]
    max_steps = 5

    # Emit start
    yield {'type': 'start'}

    for step in range(max_steps):
        # Emit start-step
        yield {'type': 'start-step'}

        translator.reset()
        usage = None
        response_metadata = None
        finish_reason = 'stop'
        tool_calls = []

        # Stream LLM response
        stream = await client.chat.completions.create(
            model='gpt-4o',
            messages=messages,
            tools=[
                {'type': 'function', 'function': {'name': name, 'parameters': schema['input']}}
                for name, schema in tools.items()
            ],
            stream=True,
            stream_options={'include_usage': True}
        )

        async for chunk in stream:
            chunk_dict = chunk.model_dump()
            event = translator.translate_chunk(chunk_dict)

            if event:
                # Track metadata internally
                if event.type == 'response-metadata':
                    if not response_metadata:
                        response_metadata = {}
                    response_metadata['id'] = event.id
                    response_metadata['modelId'] = event.model_id
                    usage = event.usage
                    continue

                if event.type == 'finish-message':
                    finish_reason = event.finish_reason
                    continue

                # Track tool calls
                if event.type == 'tool-input-available':
                    tool_calls.append({
                        'id': event.tool_call_id,
                        'name': event.tool_name,
                        'input': event.input
                    })

                # Forward content events
                yield event.to_dict()

        # Finalize translator
        for pending in translator.finalize_tool_calls():
            yield pending.to_dict()

        # If no tool calls, we're done
        if not tool_calls:
            # Emit finish-step
            yield {
                'type': 'finish-step',
                'finishReason': finish_reason,
                'usage': usage,
                'response': response_metadata
            }

            # Emit finish
            yield {
                'type': 'finish',
                'finishReason': finish_reason,
                'totalUsage': usage
            }
            break

        # Execute tools
        for tc in tool_calls:
            # Execute tool (your logic here)
            result = await execute_tool(tc['name'], tc['input'])

            # Emit tool-output-available (CRITICAL - translator doesn't emit this)
            yield {
                'type': 'tool-output-available',
                'toolCallId': tc['id'],
                'output': result
            }

            # Add to message history
            messages.append({
                'role': 'assistant',
                'tool_calls': [{
                    'id': tc['id'],
                    'type': 'function',
                    'function': {
                        'name': tc['name'],
                        'arguments': json.dumps(tc['input'])
                    }
                }]
            })
            messages.append({
                'role': 'tool',
                'tool_call_id': tc['id'],
                'content': json.dumps(result)
            })

        # Emit finish-step after tool execution
        yield {
            'type': 'finish-step',
            'finishReason': finish_reason,
            'usage': usage,
            'response': response_metadata
        }

        # Continue to next step (don't emit finish yet)

    # If we hit max steps, emit final finish
    yield {
        'type': 'finish',
        'finishReason': 'length',
        'totalUsage': usage
    }
```

## Event Checklist

When using translators directly, ensure you emit:

### Required for All Scenarios
- ✅ `start` - At the beginning
- ✅ `start-step` - Before each LLM call
- ✅ `finish-step` - After each LLM response (with usage/response)
- ✅ `finish` - At the very end (with totalUsage)

### Required for Tool Calling
- ✅ `tool-output-available` - After executing each tool
- ✅ Track tool calls from `tool-input-available` events
- ✅ Execute tools yourself (translator doesn't do this)
- ✅ Add tool results to message history

### Internal (Don't Forward)
- 🔒 `response-metadata` - Consume, don't emit (use in finish-step)
- 🔒 `finish-message` - Consume, don't emit (use in finish-step)

## Comparison: Translator vs Agent

### Using Translator (Manual)

**Pros:**
- Full control over orchestration logic
- Can integrate with custom frameworks
- Fine-grained event handling

**Cons:**
- Must implement orchestration yourself
- Must emit start/finish events manually
- Must handle tool execution
- More code to maintain

```python
# ~80 lines of orchestration code
async def custom_stream():
    yield {'type': 'start'}
    yield {'type': 'start-step'}
    # ... translator usage ...
    yield {'type': 'finish-step', ...}
    yield {'type': 'finish', ...}
```

### Using Agent (Automatic)

**Pros:**
- Zero orchestration code needed
- Automatic multi-step handling
- Built-in tool execution
- Session/context management

**Cons:**
- Less control over orchestration flow
- Fixed orchestration pattern

```python
# 5 lines - everything handled
agent = Agent(id='my-agent', model={...}, tools=[...])
async for event in agent.run_stream({'message': prompt}):
    yield event
```

## When to Use Each

### Use Translator When:
- Building integration with external frameworks (Mesh, LangGraph)
- Need custom orchestration logic not supported by Agent
- Want full control over step execution
- Testing/validating protocol translation

### Use Agent When:
- Standard multi-step agentic workflow
- Want automatic tool execution
- Need session persistence
- Want less code to maintain
- Building production applications quickly

## Common Pitfalls

### 1. Missing `tool-output-available`

```python
# ❌ BAD - Frontend won't see tool results
async for event in translator.translate_chunk(chunk):
    if event.type == 'tool-input-available':
        result = execute_tool(event.tool_name, event.input)
        # Missing: yield tool-output-available!
    yield event

# ✅ GOOD - Emit tool output
async for event in translator.translate_chunk(chunk):
    yield event
    if event.type == 'tool-input-available':
        result = await execute_tool(event.tool_name, event.input)
        yield {
            'type': 'tool-output-available',
            'toolCallId': event.tool_call_id,
            'output': result
        }
```

### 2. Forwarding Internal Events

```python
# ❌ BAD - Breaks AI SDK v5 compatibility
async for event in translator.translate_chunk(chunk):
    yield event  # Includes response-metadata, finish-message

# ✅ GOOD - Consume internal events
async for event in translator.translate_chunk(chunk):
    if event.type in ('response-metadata', 'finish-message'):
        # Track internally, don't emit
        continue
    yield event
```

### 3. Missing Usage Metadata

```python
# ❌ BAD - No usage tracking
yield {'type': 'finish-step', 'finishReason': 'stop'}

# ✅ GOOD - Include usage
yield {
    'type': 'finish-step',
    'finishReason': 'stop',
    'usage': usage,  # From response-metadata
    'response': {'id': response_id, 'modelId': model_id}
}
```

## Testing Your Implementation

Verify frontend compatibility:

```typescript
// React component
function TestChat() {
  const { messages, isLoading } = useChat({
    api: '/api/chat',
  });

  // Should work correctly:
  // - isLoading true during generation
  // - messages update with tool calls
  // - isLoading false when complete

  return <div>{/* ... */}</div>;
}
```

## See Also

- [Agent API Reference](api-reference.md#agent) - Using Agent for automatic orchestration
- [Stream Protocol](stream-protocol.md) - Full event specification
- [Providers](providers.md) - Available translators
