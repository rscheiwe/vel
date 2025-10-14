---
layout: default
title: Stream Protocol
nav_order: 7
---

# Stream Protocol

Complete reference for the Vel streaming event protocol, based on the Vercel AI SDK stream protocol.

## Overview

Vel uses a standardized stream protocol for real-time agent responses. This protocol is **provider-agnostic**, meaning the same event structure works across OpenAI, Gemini, Claude, and any future providers.

**Key Benefits:**
- ✓ Consistent events across all providers
- ✓ Compatible with Vercel AI SDK
- ✓ Real-time text and tool call streaming
- ✓ Easy to parse and handle
- ✓ Type-safe event structures

**Reference:** Based on [Vercel AI SDK Stream Protocol](https://ai-sdk.dev/docs/ai-sdk-ui/stream-protocol)

## Event Types

All events have a `type` field identifying the event:

| Event Type | Description |
|------------|-------------|
| `start` | Message generation started |
| `text-start` | Text block started |
| `text-delta` | Text chunk received |
| `text-end` | Text block ended |
| `tool-input-start` | Tool call started |
| `tool-input-delta` | Tool argument chunk (streaming) |
| `tool-input-available` | Tool arguments complete |
| `tool-output-available` | Tool execution result |
| `finish-message` | Message generation complete |
| `error` | Error occurred |
| `reasoning-start` | Reasoning block started (future) |
| `reasoning-delta` | Reasoning chunk (future) |
| `reasoning-end` | Reasoning block ended (future) |

## Event Reference

### start

**When:** Message generation begins

**Fields:**
- `type`: `"start"`
- `messageId` (optional): Unique message identifier

**Example:**
```json
{
  "type": "start",
  "messageId": "msg_abc123"
}
```

---

### text-start

**When:** Text block starts (before first text-delta)

**Fields:**
- `type`: `"text-start"`
- `id`: Text block identifier (UUID)

**Example:**
```json
{
  "type": "text-start",
  "id": "block_123"
}
```

---

### text-delta

**When:** Text chunk arrives

**Fields:**
- `type`: `"text-delta"`
- `id`: Text block identifier
- `delta`: Text chunk (string)

**Example:**
```json
{
  "type": "text-delta",
  "id": "block_123",
  "delta": "Hello"
}
```

**Usage:**
```python
async for event in agent.run_stream({'message': 'Tell me a story'}):
    if event['type'] == 'text-delta':
        print(event['delta'], end='', flush=True)
```

---

### text-end

**When:** Text block completes

**Fields:**
- `type`: `"text-end"`
- `id`: Text block identifier

**Example:**
```json
{
  "type": "text-end",
  "id": "block_123"
}
```

---

### tool-input-start

**When:** Tool call begins (LLM decides to call a tool)

**Fields:**
- `type`: `"tool-input-start"`
- `toolCallId`: Unique tool call identifier
- `toolName`: Name of the tool being called

**Example:**
```json
{
  "type": "tool-input-start",
  "toolCallId": "call_abc123",
  "toolName": "get_weather"
}
```

---

### tool-input-delta

**When:** Tool argument chunk arrives (OpenAI and Claude stream arguments incrementally)

**Fields:**
- `type`: `"tool-input-delta"`
- `toolCallId`: Tool call identifier
- `inputTextDelta`: Argument JSON chunk (string)

**Example:**
```json
{
  "type": "tool-input-delta",
  "toolCallId": "call_abc123",
  "inputTextDelta": "{\"city\":"
}
```

**Note:** Gemini doesn't stream tool arguments incrementally; only emits `tool-input-available` with complete arguments.

---

### tool-input-available

**When:** Tool arguments are complete and ready for execution

**Fields:**
- `type`: `"tool-input-available"`
- `toolCallId`: Tool call identifier
- `toolName`: Tool name
- `input`: Parsed tool arguments (object)

**Example:**
```json
{
  "type": "tool-input-available",
  "toolCallId": "call_abc123",
  "toolName": "get_weather",
  "input": {
    "city": "San Francisco"
  }
}
```

**Usage:**
```python
async for event in agent.run_stream({'message': 'Weather in NYC?'}):
    if event['type'] == 'tool-input-available':
        print(f"Calling {event['toolName']} with {event['input']}")
```

---

### tool-output-available

**When:** Tool execution completes with result

**Fields:**
- `type`: `"tool-output-available"`
- `toolCallId`: Tool call identifier
- `output`: Tool result (any JSON-serializable value)

**Example:**
```json
{
  "type": "tool-output-available",
  "toolCallId": "call_abc123",
  "output": {
    "temp_f": 72,
    "condition": "sunny",
    "city": "San Francisco"
  }
}
```

---

### finish-message

**When:** Message generation complete

**Fields:**
- `type`: `"finish-message"`
- `finishReason`: Reason for completion

**Finish Reasons:**
- `"stop"`: Natural completion
- `"length"`: Max tokens reached
- `"tool_calls"`: Completed with tool calls
- `"content_filter"`: Blocked by content filter
- `"error"`: Error occurred

**Example:**
```json
{
  "type": "finish-message",
  "finishReason": "stop"
}
```

---

### error

**When:** Error occurs during generation

**Fields:**
- `type`: `"error"`
- `error`: Error message (string)

**Example:**
```json
{
  "type": "error",
  "error": "Rate limit exceeded"
}
```

## Event Sequences

### Simple Text Response

```
1. text-start
2. text-delta (multiple)
3. text-end
4. finish-message
```

**Example:**
```json
{"type": "text-start", "id": "block_1"}
{"type": "text-delta", "id": "block_1", "delta": "Hello"}
{"type": "text-delta", "id": "block_1", "delta": " world"}
{"type": "text-end", "id": "block_1"}
{"type": "finish-message", "finishReason": "stop"}
```

### Tool Call (Single)

```
1. tool-input-start
2. tool-input-delta (multiple, OpenAI only)
3. tool-input-available
4. tool-output-available
5. text-start
6. text-delta (multiple)
7. text-end
8. finish-message
```

**Example:**
```json
{"type": "tool-input-start", "toolCallId": "call_1", "toolName": "get_weather"}
{"type": "tool-input-delta", "toolCallId": "call_1", "inputTextDelta": "{\"city\":"}
{"type": "tool-input-delta", "toolCallId": "call_1", "inputTextDelta": "\"NYC\"}"}
{"type": "tool-input-available", "toolCallId": "call_1", "toolName": "get_weather", "input": {"city": "NYC"}}
{"type": "tool-output-available", "toolCallId": "call_1", "output": {"temp_f": 65, "condition": "cloudy"}}
{"type": "text-start", "id": "block_1"}
{"type": "text-delta", "id": "block_1", "delta": "The weather in NYC is cloudy, 65°F"}
{"type": "text-end", "id": "block_1"}
{"type": "finish-message", "finishReason": "stop"}
```

### Multiple Tool Calls

```
1. tool-input-start (tool 1)
2. tool-input-start (tool 2)
3. tool-input-available (tool 1)
4. tool-input-available (tool 2)
5. tool-output-available (tool 1)
6. tool-output-available (tool 2)
7. text-start
8. text-delta (multiple)
9. text-end
10. finish-message
```

### Error During Generation

```
1. text-start
2. text-delta
3. error
```

**Example:**
```json
{"type": "text-start", "id": "block_1"}
{"type": "text-delta", "id": "block_1", "delta": "Let me"}
{"type": "error", "error": "Rate limit exceeded"}
```

## Handling Events

### Basic Text Streaming

```python
async def stream_text(agent, message):
    """Stream text to console"""
    async for event in agent.run_stream({'message': message}):
        if event['type'] == 'text-delta':
            print(event['delta'], end='', flush=True)
        elif event['type'] == 'finish-message':
            print()  # Newline
            break
```

### Collecting Full Response

```python
async def get_full_response(agent, message):
    """Collect full response from stream"""
    text_parts = []
    tool_calls = []

    async for event in agent.run_stream({'message': message}):
        if event['type'] == 'text-delta':
            text_parts.append(event['delta'])
        elif event['type'] == 'tool-input-available':
            tool_calls.append({
                'name': event['toolName'],
                'input': event['input']
            })

    return {
        'text': ''.join(text_parts),
        'tool_calls': tool_calls
    }
```

### Tool Call Monitoring

```python
async def monitor_tools(agent, message):
    """Monitor tool calls with real-time updates"""
    async for event in agent.run_stream({'message': message}):
        if event['type'] == 'tool-input-start':
            print(f"\n🔧 Calling tool: {event['toolName']}")
        elif event['type'] == 'tool-input-available':
            print(f"   Input: {event['input']}")
        elif event['type'] == 'tool-output-available':
            print(f"   Output: {event['output']}")
        elif event['type'] == 'text-delta':
            print(event['delta'], end='', flush=True)
```

### Error Handling

```python
async def safe_stream(agent, message):
    """Stream with error handling"""
    try:
        async for event in agent.run_stream({'message': message}):
            if event['type'] == 'error':
                print(f"\n❌ Error: {event['error']}")
                break
            elif event['type'] == 'text-delta':
                print(event['delta'], end='', flush=True)
    except Exception as e:
        print(f"\n❌ Stream error: {e}")
```

## Integration Examples

### FastAPI SSE Endpoint

```python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from vel import Agent

app = FastAPI()

@app.post("/stream")
async def stream_response(message: str):
    """SSE endpoint for real-time streaming"""
    agent = Agent(
        id='my-agent',
        model={'provider': 'openai', 'model': 'gpt-4o'}
    )

    async def event_generator():
        async for event in agent.run_stream({'message': message}):
            # SSE format: data: {json}\n\n
            yield f"data: {json.dumps(event)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream"
    )
```

### React Frontend (TypeScript)

```typescript
async function streamResponse(message: string) {
  const response = await fetch('/stream', {
    method: 'POST',
    body: JSON.stringify({ message }),
    headers: { 'Content-Type': 'application/json' }
  });

  const reader = response.body?.getReader();
  const decoder = new TextDecoder();

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    const chunk = decoder.decode(value);
    const lines = chunk.split('\n');

    for (const line of lines) {
      if (line.startsWith('data: ')) {
        const event = JSON.parse(line.slice(6));

        switch (event.type) {
          case 'text-delta':
            appendText(event.delta);
            break;
          case 'tool-input-available':
            showToolCall(event.toolName, event.input);
            break;
          case 'finish-message':
            onComplete();
            break;
        }
      }
    }
  }
}
```

### WebSocket Integration

```python
import websockets
import json

async def websocket_handler(websocket, path):
    """WebSocket handler for bidirectional streaming"""
    agent = Agent(
        id='chat-agent',
        model={'provider': 'openai', 'model': 'gpt-4o'}
    )

    async for message in websocket:
        data = json.loads(message)

        async for event in agent.run_stream({'message': data['message']}):
            await websocket.send(json.dumps(event))
```

## Provider Differences

### OpenAI
- ✓ Streams tool arguments incrementally (`tool-input-delta` events)
- ✓ Supports multiple tool calls per response
- ✓ Text streaming highly granular (word/character level)

### Gemini
- ✗ Tool arguments not streamed (no `tool-input-delta`)
- ✓ `tool-input-available` emitted with complete arguments
- ✓ Text streaming at sentence/phrase level

### Claude
- ✓ Streams tool arguments incrementally (`tool-input-delta` events)
- ✓ Supports multiple tool calls per response
- ✓ Text streaming highly granular (word/character level)
- ✓ Supports extended thinking/reasoning blocks (future)

**All providers emit identical event types**, just with different granularity and timing.

## Troubleshooting

### Events Arrive All at Once

**Problem:** Stream events aren't streaming, all arrive instantly.

**Solutions:**
1. Verify using `run_stream()` not `run()`
2. Check network path supports streaming (no buffering proxies)
3. Use `flush=True` when printing
4. Check provider API is actually streaming

### Missing text-delta Events

**Problem:** No text received, only tool events.

**Cause:** LLM called tool but didn't generate follow-up text.

**Solution:** This is normal behavior. Some responses are tool-only.

### tool-input-delta Never Fires (Gemini)

**Cause:** Gemini doesn't stream tool arguments.

**Solution:** Use `tool-input-available` instead. This is expected behavior.

### Events Out of Order

**Problem:** Events arrive in unexpected sequence.

**Cause:** Async processing or buffering.

**Solution:** Events are emitted in generation order. If receiving out of order, check your async handling code.

## Best Practices

### 1. Handle All Event Types

```python
# ✓ Good: Comprehensive handling
async for event in agent.run_stream({'message': msg}):
    event_type = event['type']
    if event_type == 'text-delta':
        handle_text(event)
    elif event_type == 'tool-input-available':
        handle_tool_call(event)
    elif event_type == 'tool-output-available':
        handle_tool_result(event)
    elif event_type == 'error':
        handle_error(event)
    elif event_type == 'finish-message':
        handle_finish(event)
```

### 2. Accumulate Text Properly

```python
# ✓ Good: Track block IDs
text_blocks = {}
async for event in agent.run_stream({'message': msg}):
    if event['type'] == 'text-delta':
        block_id = event['id']
        text_blocks.setdefault(block_id, []).append(event['delta'])

full_text = ''.join(text_blocks.get('block_1', []))
```

### 3. Match Tool Calls to Results

```python
# ✓ Good: Track by tool_call_id
tool_results = {}
async for event in agent.run_stream({'message': msg}):
    if event['type'] == 'tool-output-available':
        tool_results[event['toolCallId']] = event['output']
```

### 4. Graceful Error Handling

```python
# ✓ Good: Don't break stream on error
async for event in agent.run_stream({'message': msg}):
    if event['type'] == 'error':
        log_error(event['error'])
        # Continue processing remaining events
    else:
        process_event(event)
```

## Next Steps

- [API Reference](api-reference.md) - Complete API documentation
- [Providers](providers.md) - Provider-specific streaming behavior
- [Tools](tools.md) - Tool system with streaming events
