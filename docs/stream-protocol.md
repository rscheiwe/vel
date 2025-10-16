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
| `reasoning-start` | Reasoning block started |
| `reasoning-delta` | Reasoning chunk received |
| `reasoning-end` | Reasoning block ended |
| `response-metadata` | Response metadata (usage, model info) |
| `source` | Source/citation event (web search, documents) |
| `step-start` | Agent step started (multi-step agents) |
| `step-finish` | Agent step finished (multi-step agents) |
| `data-*` | Custom application data (e.g., `data-notification`, `data-progress`) |
| `finish-message` | Message generation complete |
| `finish` | Generation complete (V5 UI Stream Protocol) |
| `error` | Error occurred |

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

**Provider-Executed Tools:**

OpenAI Responses API supports provider-executed tools (`web_search_call`, `computer_call`) that are executed by OpenAI's servers. These emit `tool-output-available` with special metadata:

```json
{
  "type": "tool-output-available",
  "toolCallId": "call_web123",
  "output": {
    "results": [...],
    "sources": [...]
  },
  "callProviderMetadata": {
    "providerExecuted": true,
    "providerName": "openai",
    "toolType": "web_search_call"
  }
}
```

---

### reasoning-start

**When:** Reasoning/thinking block starts (OpenAI o1/o3, Claude thinking mode)

**Fields:**
- `type`: `"reasoning-start"`
- `id`: Reasoning block identifier

**Example:**
```json
{
  "type": "reasoning-start",
  "id": "reasoning_1"
}
```

**Supported Models:**
- **OpenAI**: o1, o3, o1-mini, o3-mini (reasoning encrypted/hidden)
- **Anthropic**: claude-sonnet-4 with extended thinking (reasoning visible)

---

### reasoning-delta

**When:** Reasoning chunk arrives

**Fields:**
- `type`: `"reasoning-delta"`
- `id`: Reasoning block identifier
- `delta`: Reasoning text chunk (string, may be empty for encrypted reasoning)

**Example (Anthropic visible thinking):**
```json
{
  "type": "reasoning-delta",
  "id": "reasoning_1",
  "delta": "Let me break this down step by step..."
}
```

**Example (OpenAI encrypted reasoning):**
```json
{
  "type": "reasoning-delta",
  "id": "reasoning_1",
  "delta": ""
}
```

**Note:** OpenAI encrypts reasoning content for o1/o3 models, so `delta` will be empty. Claude's extended thinking is visible.

**Usage:**
```python
reasoning_chunks = []
async for event in agent.run_stream({'message': 'Complex math problem'}):
    if event['type'] == 'reasoning-delta':
        delta = event.get('delta', '')
        if delta:  # Only print if not empty (OpenAI encrypts)
            reasoning_chunks.append(delta)
            print(delta, end='', flush=True)
```

---

### reasoning-end

**When:** Reasoning block completes

**Fields:**
- `type`: `"reasoning-end"`
- `id`: Reasoning block identifier

**Example:**
```json
{
  "type": "reasoning-end",
  "id": "reasoning_1"
}
```

---

### response-metadata

**When:** Response metadata is available (id, model, usage)

**Fields:**
- `type`: `"response-metadata"`
- `id` (optional): Message/response ID from provider
- `modelId` (optional): Model identifier
- `timestamp` (optional): ISO 8601 timestamp
- `usage` (optional): Token usage statistics

**AI SDK V5 Parity:**
Metadata is emitted **early** when id/model are known (before streaming completes), then **updated** when usage data arrives.

**Early metadata example:**
```json
{
  "type": "response-metadata",
  "id": "resp_abc123",
  "modelId": "gpt-4o",
  "usage": null
}
```

**Usage update example:**
```json
{
  "type": "response-metadata",
  "id": "resp_abc123",
  "modelId": "gpt-4o",
  "usage": {
    "promptTokens": 50,
    "completionTokens": 120,
    "totalTokens": 170
  }
}
```

**Usage:**
```python
async for event in agent.run_stream({'message': 'Hello'}):
    if event['type'] == 'response-metadata':
        if event.get('usage'):
            tokens = event['usage']['totalTokens']
            print(f"Total tokens used: {tokens}")
```

---

### source

**When:** Sources/citations are available (web search results, document references)

**Fields:**
- `type`: `"source"`
- `sources`: Array of source objects

**Source Object Structure:**
```json
{
  "type": "web" | "file",
  "url": "https://...",
  "title": "Source title",
  "snippet": "Excerpt from source",
  "sourceId": "provider-specific-id"
}
```

**Example (Web search sources):**
```json
{
  "type": "source",
  "sources": [
    {
      "type": "web",
      "url": "https://example.com/article",
      "title": "Weather Patterns in California",
      "snippet": "Recent studies show...",
      "sourceId": "src_abc123"
    },
    {
      "type": "web",
      "url": "https://weather.gov/sf",
      "title": "San Francisco Forecast",
      "snippet": "Current conditions: 72°F, sunny"
    }
  ]
}
```

**Provider Support:**
- **OpenAI Responses API**: web_search_call sources, file citations
- **Gemini**: Grounding sources (web citations)
- **Anthropic**: Not yet supported

**Usage:**
```python
async for event in agent.run_stream({'message': 'Latest weather news'}):
    if event['type'] == 'source':
        for src in event['sources']:
            print(f"Source: {src['title']} - {src['url']}")
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

---

### data-* (Custom Data Events)

**When:** Application-specific data needs to be streamed

**Pattern:** `data-{customName}` (e.g., `data-notification`, `data-progress`, `data-stage-data`)

**Fields:**
- `type`: Custom type following `data-*` pattern (string)
- `data`: Custom payload (any JSON-serializable value)
- `transient` (optional): If `true`, event NOT saved to message history (boolean)

**Use Cases:**
- UI notifications (transient)
- Progress updates (transient)
- Stage transitions (persistent)
- Real-time metrics (persistent or transient)

**Examples:**

Transient notification (not saved to history):
```json
{
  "type": "data-notification",
  "data": {
    "message": "Processing your request...",
    "level": "info"
  },
  "transient": true
}
```

Persistent stage data (saved to history):
```json
{
  "type": "data-stage-data",
  "data": {
    "stage": "analyzing",
    "step": 1,
    "total_steps": 5,
    "description": "Breaking down the problem"
  },
  "transient": false
}
```

Progress update (transient):
```json
{
  "type": "data-progress",
  "data": {
    "percent": 50,
    "message": "Halfway there..."
  },
  "transient": true
}
```

**Backend Usage:**
```python
from vel import DataEvent

# Emit transient notification
yield DataEvent(
    type='data-notification',
    data={'message': 'Processing...', 'level': 'info'},
    transient=True
).to_dict()

# Emit persistent stage data
yield DataEvent(
    type='data-stage-data',
    data={'stage': 'analyzing'},
    transient=False
).to_dict()
```

**Frontend Integration (Vercel AI SDK):**
```typescript
import { useChat } from 'ai/react';

const { messages } = useChat({
  api: '/api/chat',

  // Transient events (NOT in message.parts)
  onData: (dataPart) => {
    if (dataPart.type === 'data-notification') {
      toast.info(dataPart.data.message);
    }
    if (dataPart.type === 'data-progress') {
      setProgress(dataPart.data.percent);
    }
  }
});

// Persistent events (in message.parts)
messages.forEach(msg => {
  msg.parts.forEach(part => {
    if (part.type === 'data-stage-data') {
      console.log('Stage:', part.data.stage);
    }
  });
});
```

**Transient vs Persistent:**
- `transient: true` - Event sent to client but NOT added to message history (real-time UI updates only)
- `transient: false` (default) - Event saved to message parts array (can be replayed/stored)

**See also:** `examples/custom_data_events.py` for comprehensive examples.

---

## Why Use Custom Data Events?

Custom `data-*` events solve the problem of streaming **application-specific metadata** alongside LLM responses. They enable richer, more interactive UIs without polluting the core message stream.

### Primary Use Cases

#### 1. RAG Source Citations

Stream which documents/sources were used for retrieval-augmented generation:

```python
from vel import DataEvent

# After retrieving documents
for doc in retrieved_docs:
    yield DataEvent(
        type='data-source',
        data={
            'url': doc.url,
            'title': doc.title,
            'snippet': doc.excerpt,
            'relevance_score': doc.score
        },
        transient=False  # Save for citation display
    ).to_dict()
```

**Frontend displays citations:**
```typescript
const { messages } = useChat({
  api: '/api/chat'
});

// Sources saved in message.parts
messages.forEach(msg => {
  const sources = msg.parts.filter(p => p.type === 'data-source');
  sources.forEach(src => {
    console.log(`Source: ${src.data.title} - ${src.data.url}`);
  });
});
```

#### 2. Multi-Step Agent Progress

Show users what the agent is doing at each step:

```python
# Emit transient progress (don't save to DB)
yield DataEvent(
    type='data-stage',
    data={
        'stage': 'searching',
        'message': 'Searching knowledge base...',
        'progress': 30
    },
    transient=True  # Real-time UI only
).to_dict()

yield DataEvent(
    type='data-stage',
    data={
        'stage': 'analyzing',
        'message': 'Analyzing results...',
        'progress': 60
    },
    transient=True
).to_dict()
```

**Frontend shows real-time progress:**
```typescript
const { messages } = useChat({
  onData: (dataPart) => {
    if (dataPart.type === 'data-stage') {
      setStatus(dataPart.data.message);
      setProgress(dataPart.data.progress);
    }
  }
});
```

#### 3. Tool Execution Metadata

Stream additional context about tool calls:

```python
# After tool execution
yield DataEvent(
    type='data-tool-metadata',
    data={
        'tool_name': 'get_weather',
        'execution_time_ms': 450,
        'api_latency_ms': 380,
        'cache_hit': False,
        'cost_usd': 0.0001
    },
    transient=False  # Save for analytics
).to_dict()
```

#### 4. Temporary UI Notifications

Send notifications that shouldn't clutter message history:

```python
# Temporary status update
yield DataEvent(
    type='data-notification',
    data={
        'message': 'Rate limited, retrying in 1 second...',
        'level': 'warning'
    },
    transient=True  # Don't save
).to_dict()

# Another notification
yield DataEvent(
    type='data-notification',
    data={
        'message': 'Connected to external API',
        'level': 'info'
    },
    transient=True
).to_dict()
```

**Frontend shows toast notifications:**
```typescript
onData: (dataPart) => {
  if (dataPart.type === 'data-notification') {
    toast[dataPart.data.level](dataPart.data.message);
  }
}
```

#### 5. Analytics and Usage Tracking

Stream metrics for monitoring and billing:

```python
# After completion
yield DataEvent(
    type='data-metrics',
    data={
        'tokens_used': 1250,
        'model': 'gpt-4o',
        'cost_usd': 0.0375,
        'duration_ms': 2800,
        'tools_called': 3
    },
    transient=False  # Save for analytics
).to_dict()
```

### Why Not Use Standard Events?

| Scenario | Why Custom `data-*` Events? |
|----------|---------------------------|
| **RAG sources** | Need custom structure (URL, title, snippet) not in standard tool/text events |
| **Progress updates** | Transient (don't want cluttering message history) |
| **Business metrics** | Application-specific data (latency, costs) not part of LLM protocol |
| **Dynamic UI state** | Need to update React components beyond text streaming |
| **Multi-source data** | Combining data from multiple tools/APIs into structured format |

### Separation of Concerns

**Standard events = LLM response layer:**
```json
{"type": "text-delta", "delta": "The weather is sunny..."}
{"type": "tool-input-available", "toolName": "get_weather", ...}
```

**Custom data-* events = Application layer:**
```json
{"type": "data-source", "data": {"url": "...", "title": "..."}}
{"type": "data-metrics", "data": {"cost": 0.02, "tokens": 500}}
{"type": "data-progress", "data": {"step": 3, "total": 5}}
```

### Real-World Example: RAG Chatbot

**Backend:**
```python
from vel import Agent, DataEvent

async def rag_chat(message: str):
    # 1. Retrieve relevant documents
    docs = await vector_search(message)

    # 2. Stream sources as data events (persistent)
    for doc in docs:
        yield DataEvent(
            type='data-source',
            data={'title': doc.title, 'url': doc.url, 'score': doc.score},
            transient=False  # Save for citations
        ).to_dict()

    # 3. Stream agent response (standard events)
    agent = Agent(id='rag-agent', model={'provider': 'openai', 'model': 'gpt-4o'})
    prompt = f"Context: {format_docs(docs)}\n\nQuestion: {message}"

    async for event in agent.run_stream({'message': prompt}):
        yield event

    # 4. Stream usage metrics (persistent)
    yield DataEvent(
        type='data-metrics',
        data={'tokens': 1200, 'sources_used': len(docs)},
        transient=False  # Save for analytics
    ).to_dict()
```

**Frontend:**
```typescript
import { useChat } from 'ai/react';

const { messages } = useChat({
  api: '/api/rag-chat'
});

// Render messages with citations
<div>
  {messages.map(msg => (
    <div key={msg.id}>
      {/* Standard text parts */}
      {msg.parts
        .filter(p => p.type === 'text')
        .map(p => <p>{p.text}</p>)}

      {/* Citations from data-source events */}
      <div className="citations">
        {msg.parts
          .filter(p => p.type === 'data-source')
          .map(src => (
            <a href={src.data.url}>
              {src.data.title} (score: {src.data.score})
            </a>
          ))}
      </div>

      {/* Metrics from data-metrics events */}
      {msg.parts
        .filter(p => p.type === 'data-metrics')
        .map(m => (
          <small>
            {m.data.tokens} tokens, {m.data.sources_used} sources
          </small>
        ))}
    </div>
  ))}
</div>
```

### Best Practices

**Use `transient: true` for:**
- Real-time status updates ("Processing...")
- Progress indicators (loading bars)
- Temporary notifications (toasts)
- UI state that doesn't need persistence

**Use `transient: false` for:**
- RAG source citations
- Tool execution metadata
- Analytics/metrics data
- Stage transitions you want to replay
- Anything needed for debugging/audit trails

**Naming convention:**
- Use descriptive names: `data-source`, `data-metrics`, `data-stage`
- Avoid generic names: `data-info`, `data-stuff`
- Follow pattern: `data-{domain}-{type}` (e.g., `data-rag-source`, `data-analytics-metric`)

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

## Message Aggregation

### MessageReducer

**MessageReducer** aggregates streaming events into the Vercel AI SDK message format for database storage and frontend integration.

**Why use MessageReducer?**
- ✓ Compatible with Vercel AI SDK `useChat` hook
- ✓ Ready for database storage
- ✓ Aggregates multiple events into structured parts array
- ✓ Handles provider metadata (message IDs, function call IDs)
- ✓ Supports custom message IDs and metadata
- ✓ Tracks tool calls with proper callProviderMetadata

### Basic Usage

```python
from vel import Agent, MessageReducer

# Create reducer
reducer = MessageReducer()

# Add user message
user_msg = reducer.add_user_message("Hello!")

# Stream agent response
agent = Agent(
    id='chat-agent',
    model={'provider': 'openai', 'model': 'gpt-4o'}
)

async for event in agent.run_stream({'message': 'Hello!'}):
    reducer.process_event(event)

# Get messages for database storage
messages = reducer.get_messages()
# [
#   {user message},
#   {assistant message with aggregated parts}
# ]
```

### Message Structure

Messages follow the Vercel AI SDK format:

```python
{
  "id": "msg-abc123",           # UUID or custom ID
  "role": "user" | "assistant", # Message role
  "parts": [                     # Array of parts
    {
      "type": "text",
      "text": "Hello world!",
      "state": "done",
      "providerMetadata": {      # OpenAI message ID
        "openai": {
          "itemId": "msg_xyz"
        }
      }
    },
    {
      "type": "tool-call",
      "toolCallId": "call_123",
      "toolName": "get_weather",
      "input": {"city": "SF"},
      "state": "call",
      "callProviderMetadata": {  # OpenAI function call ID
        "openai": {
          "itemId": "call_123"
        }
      }
    },
    {
      "type": "tool-result",
      "toolCallId": "call_123",
      "toolName": "get_weather",
      "output": {"temp_f": 72},
      "state": "result"
    }
  ],
  "metadata": {}                 # Optional custom metadata
}
```

### Tool Call Example

```python
from vel import Agent, MessageReducer

reducer = MessageReducer()
user_msg = reducer.add_user_message("What's the weather in San Francisco?")

agent = Agent(
    id='weather-agent',
    model={'provider': 'openai', 'model': 'gpt-4o'},
    tools=['get_weather']
)

async for event in agent.run_stream({'message': "What's the weather in SF?"}):
    reducer.process_event(event)

messages = reducer.get_messages()
assistant_msg = messages[1]

# Assistant message parts:
# [
#   {
#     "type": "tool-call",
#     "toolCallId": "call_123",
#     "toolName": "get_weather",
#     "input": {"city": "San Francisco"},
#     "state": "call",
#     "callProviderMetadata": {"openai": {"itemId": "call_123"}}
#   },
#   {
#     "type": "tool-result",
#     "toolCallId": "call_123",
#     "toolName": "get_weather",
#     "output": {"temp_f": 72, "condition": "sunny"},
#     "state": "result"
#   },
#   {
#     "type": "text",
#     "text": "The weather in San Francisco is sunny and 72°F.",
#     "state": "done",
#     "providerMetadata": {"openai": {"itemId": "msg_abc"}}
#   }
# ]
```

### Multi-Turn Conversations

```python
from vel import Agent, MessageReducer

agent = Agent(
    id='chat-agent',
    model={'provider': 'openai', 'model': 'gpt-4o'},
    session_persistence='transient'
)

session_id = 'user-123'
all_messages = []

# Turn 1
reducer1 = MessageReducer()
reducer1.add_user_message("My name is Alice")

async for event in agent.run_stream({'message': 'My name is Alice'}, session_id):
    reducer1.process_event(event)

messages1 = reducer1.get_messages()
all_messages.extend(messages1)  # Store in database

# Turn 2
reducer2 = MessageReducer()
reducer2.add_user_message("What is my name?")

async for event in agent.run_stream({'message': 'What is my name?'}, session_id):
    reducer2.process_event(event)

messages2 = reducer2.get_messages()
all_messages.extend(messages2)  # Store in database

# all_messages now contains 4 messages:
# [user_msg1, assistant_msg1, user_msg2, assistant_msg2]
```

### Custom IDs and Metadata

```python
from vel import MessageReducer

reducer = MessageReducer()

# Custom message ID (e.g., from frontend)
user_msg = reducer.add_user_message(
    "Hello",
    message_id="frontend-msg-001",
    metadata={"source": "web-app", "user_id": "user-456"}
)

# Process stream...
async for event in agent.run_stream({'message': 'Hello'}):
    reducer.process_event(event)

# Get messages with metadata
messages = reducer.get_messages(
    user_metadata={"source": "web-app"},
    assistant_metadata={"model": "gpt-4o", "tokens": 150}
)

# Or get just assistant message with custom ID
assistant_msg = reducer.get_assistant_message(
    message_id="frontend-msg-002"
)
```

### Database Storage Pattern

```python
from vel import Agent, MessageReducer
import json

async def chat_with_storage(user_input: str, conversation_id: str, db):
    """Chat pattern with database storage"""

    # Create reducer
    reducer = MessageReducer()

    # Add user message with metadata
    user_msg = reducer.add_user_message(
        user_input,
        metadata={
            "conversation_id": conversation_id,
            "timestamp": datetime.utcnow().isoformat()
        }
    )

    # Store user message
    await db.insert_message(conversation_id, user_msg)

    # Stream agent response
    agent = Agent(
        id='chat-agent',
        model={'provider': 'openai', 'model': 'gpt-4o'},
        tools=['get_weather', 'web_search']
    )

    async for event in agent.run_stream({'message': user_input}):
        reducer.process_event(event)

        # Real-time: forward event to frontend via WebSocket/SSE
        await websocket.send(json.dumps(event))

    # Get complete assistant message
    assistant_msg = reducer.get_assistant_message(
        metadata={
            "conversation_id": conversation_id,
            "timestamp": datetime.utcnow().isoformat()
        }
    )

    # Store assistant message
    await db.insert_message(conversation_id, assistant_msg)

    return reducer.get_messages()
```

### Multi-Step Agent Pattern

MessageReducer automatically handles step-start events for multi-step agents:

```python
from vel import Agent, MessageReducer

reducer = MessageReducer()
reducer.add_user_message("Should I invest in cryptocurrency?")

agent = Agent(
    id='multi-step-agent',
    model={'provider': 'openai', 'model': 'gpt-4o'},
    tools=['websearch', 'analyze', 'decide', 'provideAnswer'],
    policies={'max_steps': 8}
)

async for event in agent.run_stream({'message': "Should I invest in crypto?"}):
    reducer.process_event(event)

messages = reducer.get_messages()
assistant_msg = messages[1]

# Assistant message parts include step-start events:
# [
#   {"type": "step-start"},
#   {"type": "tool-call", "toolName": "websearch", ...},
#   {"type": "tool-result", ...},
#   {"type": "step-start"},
#   {"type": "tool-call", "toolName": "analyze", ...},
#   {"type": "tool-result", ...},
#   {"type": "step-start"},
#   {"type": "text", "text": "Based on my analysis...", ...}
# ]
```

### API Reference

**MessageReducer Class:**

```python
class MessageReducer:
    def __init__(self):
        """Create a new message reducer"""

    def reset(self):
        """Reset state for new user-assistant exchange"""

    def add_user_message(
        self,
        text: str,
        message_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Add user message, returns message dict"""

    def process_event(self, event: Dict[str, Any]) -> None:
        """Process streaming event, accumulates into parts array"""

    def get_messages(
        self,
        user_metadata: Optional[Dict[str, Any]] = None,
        assistant_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Get [user_message, assistant_message]"""

    def get_assistant_message(
        self,
        message_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Get assistant message with optional custom ID and metadata"""
```

**Event Processing:**
- `start` → Captures message ID for providerMetadata
- `text-delta` → Accumulates text chunks
- `tool-input-available` → Creates tool-call part
- `tool-output-available` → Creates tool-result part
- `step-start` → Creates step-start part
- `finish-message` → Flushes accumulated text to parts array

**See also:** `examples/message_reducer_example.py` for comprehensive usage examples.

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
