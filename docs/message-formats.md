---
layout: default
title: Message Formats & Translation
nav_order: 8
---

# Message Formats & Translation

Vel supports multiple message format patterns to work seamlessly with frontend frameworks (React, Next.js) and Python-only applications.

## Overview

There are **three message formats** in the Vel ecosystem:

1. **UIMessage** - Frontend UI state (from `useChat` hook)
2. **ModelMessage** - Unified LLM format (provider-agnostic)
3. **Provider Format** - OpenAI/Anthropic/Gemini-specific formats

```
UIMessage → ModelMessage → Provider Format → LLM API
   ↑            ↑              ↑
Frontend    Converter      Vel Provider
  State      (optional)    (automatic)
```

---

## Message Format Comparison

### UIMessage (Frontend State)

What the Vercel AI SDK's `useChat` hook produces. Contains UI state with executed tools.

```typescript
{
  id: 'msg-1',
  role: 'assistant',
  parts: [
    {
      type: 'tool-websearch',
      toolCallId: 'call_123',
      state: 'output-available',  // Tool already executed
      input: { query: 'AI trends' },
      output: { results: [...] }   // Both input AND output present
    }
  ]
}
```

**Characteristics:**
- Has `id` field for UI tracking
- Uses `parts` array for content
- Tool executions include both input and output
- Contains UI-only elements (`step-start`, `step-finish`)

### ModelMessage (Unified Format)

Provider-agnostic format that separates tool calls from results.

```python
# Tool call (assistant message)
{
  'role': 'assistant',
  'content': [
    {
      'type': 'tool-call',
      'toolCallId': 'call_123',
      'toolName': 'tool-websearch',
      'input': { 'query': 'AI trends' }
    }
  ]
}

# Tool result (separate message)
{
  'role': 'tool',
  'content': [
    {
      'type': 'tool-result',
      'toolCallId': 'call_123',
      'toolName': 'tool-websearch',
      'output': { 'results': [...] }
    }
  ]
}
```

**Characteristics:**
- No `id` field (not needed for LLM)
- Tool calls and results are separate messages
- No UI-only elements
- Works with any LLM provider

### Provider Format (OpenAI/Anthropic/Gemini)

Each provider has its own specific format that Vel translates to automatically.

**OpenAI:**
```python
{
  'role': 'assistant',
  'content': '',
  'tool_calls': [{
    'id': 'call_123',
    'type': 'function',
    'function': {
      'name': 'tool-websearch',
      'arguments': '{"query":"AI trends"}'
    }
  }]
}
```

**Anthropic:**
```python
{
  'role': 'assistant',
  'content': [{
    'type': 'tool_use',
    'id': 'call_123',
    'name': 'tool-websearch',
    'input': {'query': 'AI trends'}
  }]
}
```

**Gemini:**
```python
{
  'role': 'model',  # Not 'assistant'
  'parts': [{
    'function_call': {
      'name': 'tool-websearch',
      'args': {'query': 'AI trends'}
    }
  }]
}
```

---

## Pattern 1: React Frontend + FastAPI Backend

**Architecture:**

```
┌─────────────────────────────────────────────────────────────┐
│                      React Frontend                         │
│                                                             │
│  useChat hook                                               │
│    ↓                                                        │
│  UIMessages (UI state)                                      │
│    ↓                                                        │
│  convertToModelMessages() ← Vercel AI SDK                   │
│    ↓                                                        │
│  ModelMessages                                              │
└─────────────────────┬───────────────────────────────────────┘
                      │ HTTP Request
                      │ { messages: [...] }
                      ↓
┌─────────────────────────────────────────────────────────────┐
│                    FastAPI Backend                          │
│                                                             │
│  FastAPI Endpoint                                           │
│    ↓                                                        │
│  ModelMessages (from request)                               │
│    ↓                                                        │
│  Vel Agent.run({ messages })                                │
│    ↓                                                        │
│  Vel Provider Translation ← Automatic                       │
│    ↓                                                        │
│  Provider Format (OpenAI/Anthropic/Gemini)                  │
│    ↓                                                        │
│  LLM API                                                    │
└─────────────────────────────────────────────────────────────┘
```

**Frontend Code:**

```typescript
// React component
import { useChat, convertToModelMessages } from 'ai';

function ChatComponent() {
  const { messages, input, handleSubmit } = useChat({
    api: '/api/chat'
  });

  const sendMessage = async () => {
    // Convert UIMessages to ModelMessages
    const modelMessages = convertToModelMessages(messages);

    // Send to backend
    const response = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ messages: modelMessages })
    });

    return response.json();
  };

  // ...
}
```

**Backend Code:**

```python
# FastAPI endpoint
from fastapi import FastAPI
from vel import Agent

app = FastAPI()

@app.post("/api/chat")
async def chat(request: dict):
    # ModelMessages already in correct format
    messages = request["messages"]

    # Vel translates ModelMessage → Provider format automatically
    agent = Agent(
        id='chat-agent',
        model={'provider': 'openai', 'model': 'gpt-4o'}
    )

    # Translation happens inside agent.run()
    response = await agent.run({'messages': messages})

    return {'response': response}
```

**What Vel Does:**

1. Receives ModelMessages from request
2. Detects `messages` array in input
3. Calls `translate_to_openai(messages)` (or anthropic/gemini based on provider)
4. Sends translated messages to LLM API
5. Returns response

---

## Pattern 2: Python-Only (No Frontend)

**Architecture:**

```
┌─────────────────────────────────────────────────────────────┐
│                     Python Application                      │
│                                                             │
│  Manually build ModelMessages                               │
│    OR                                                       │
│  Load UIMessages from DB                                    │
│    ↓                                                        │
│  convert_to_model_messages() ← Vel utility (optional)       │
│    ↓                                                        │
│  ModelMessages                                              │
│    ↓                                                        │
│  Vel Agent.run({ messages })                                │
│    ↓                                                        │
│  Vel Provider Translation ← Automatic                       │
│    ↓                                                        │
│  Provider Format (OpenAI/Anthropic/Gemini)                  │
│    ↓                                                        │
│  LLM API                                                    │
└─────────────────────────────────────────────────────────────┘
```

**Option A: Manually Build ModelMessages**

```python
from vel import Agent

agent = Agent(
    id='chat-agent',
    model={'provider': 'openai', 'model': 'gpt-4o'}
)

# Build ModelMessage format manually
messages = [
    {
        'role': 'user',
        'content': 'What is the weather in SF?'
    },
    {
        'role': 'assistant',
        'content': [
            {
                'type': 'tool-call',
                'toolCallId': 'call_123',
                'toolName': 'get_weather',
                'input': {'city': 'San Francisco'}
            }
        ]
    },
    {
        'role': 'tool',
        'content': [
            {
                'type': 'tool-result',
                'toolCallId': 'call_123',
                'toolName': 'get_weather',
                'output': {'temp': 72, 'condition': 'sunny'}
            }
        ]
    }
]

# Vel translates ModelMessage → OpenAI format automatically
response = await agent.run({'messages': messages})
```

**Option B: Convert UIMessages from Database**

```python
from vel import Agent
from vel.utils import convert_to_model_messages

# Load UIMessages from database (e.g., saved from React frontend)
ui_messages = database.get_conversation(user_id)
# ui_messages = [
#     {
#         'id': 'msg-1',
#         'role': 'assistant',
#         'parts': [
#             {
#                 'type': 'tool-websearch',
#                 'state': 'output-available',
#                 'input': {...},
#                 'output': {...}
#             }
#         ]
#     }
# ]

# Convert UIMessages to ModelMessages
model_messages = convert_to_model_messages(ui_messages)

# Use with Vel agent
agent = Agent(
    id='chat-agent',
    model={'provider': 'openai', 'model': 'gpt-4o'}
)

response = await agent.run({'messages': model_messages})
```

---

## Pattern 3: Legacy Format (Backwards Compatible)

Vel also supports the old format with session management for backwards compatibility.

```python
from vel import Agent

agent = Agent(
    id='chat-agent',
    model={'provider': 'openai', 'model': 'gpt-4o'}
)

# Old format: single message + session_id
# Vel manages conversation history internally
response1 = await agent.run(
    {'message': 'My name is Alice'},
    session_id='session-123'
)

response2 = await agent.run(
    {'message': 'What is my name?'},
    session_id='session-123'  # Vel remembers previous message
)
```

**Note:** This pattern uses Vel's internal ContextManager to build the messages array. The new patterns give you full control over conversation history.

---

## Translation Details

### Vel's Automatic Translation

When you send ModelMessages to Vel, translation happens automatically:

```python
# You provide
messages = [
    {'role': 'assistant', 'content': [{'type': 'tool-call', ...}]}
]

# Vel calls translate_to_openai(messages) internally
# Result sent to OpenAI API:
openai_messages = [
    {
        'role': 'assistant',
        'content': '',
        'tool_calls': [{'id': '...', 'type': 'function', ...}]
    }
]
```

### Supported Content Types

All translators support:

| Content Type | Description | Example |
|--------------|-------------|---------|
| Text | Simple string or text part | `'Hello'` or `{type: 'text', text: 'Hello'}` |
| Reasoning | LLM reasoning steps (o1/o3, extended thinking) | `{type: 'reasoning', text: 'Let me think...'}` |
| Images | Base64 or URL | `{type: 'image', image: 'base64...', mimeType: 'image/png'}` |
| Files | PDF, documents | `{type: 'file', data: 'base64...', mimeType: 'application/pdf'}` |
| Tool Calls | Function invocations | `{type: 'tool-call', toolCallId: '...', toolName: '...', input: {...}}` |
| Tool Results | Function outputs | `{type: 'tool-result', toolCallId: '...', output: {...}}` |

#### Reasoning Content

Reasoning content captures the LLM's internal thinking process before producing a final answer. This is used by:
- **OpenAI**: o1, o3 models (via `reasoning-*` stream events)
- **Anthropic**: Extended thinking models (via `thinking` content blocks)

**UIMessage format** (from frontend, includes UI-only markers):
```json
{
  "role": "assistant",
  "parts": [
    {"type": "step-start"},
    {
      "type": "reasoning",
      "text": "The human is asking about...",
      "state": "done"
    },
    {
      "type": "text",
      "text": "Here is my answer",
      "state": "done"
    },
    {"type": "step-finish"}
  ]
}
```

**ModelMessage format** (after conversion, UI markers filtered):
```json
{
  "role": "assistant",
  "content": [
    {"type": "reasoning", "text": "The human is asking about..."},
    {"type": "text", "text": "Here is my answer"}
  ]
}
```

**Note**: The `step-start` and `step-finish` markers are UI-only elements that get filtered out during conversion to ModelMessage format.

**Provider Translation**:
- **OpenAI**: Reasoning text is included in the `content` field alongside other text
- **Anthropic**: Reasoning is converted to `text` content blocks
- **Gemini**: Reasoning is converted to `text` parts

The reasoning content is preserved through the entire pipeline: UIMessage → ModelMessage → Provider Format.

### Error Handling

Translation errors are caught and reported clearly:

```python
try:
    response = await agent.run({'messages': messages})
except ValueError as e:
    # MessageTranslationError converted to ValueError
    print(f"Translation failed: {e}")
    # Error message includes:
    # - Which provider failed (openai/anthropic/gemini)
    # - What went wrong (missing field, invalid format)
    # - Which message caused the issue (index)
```

---

## When to Use Each Pattern

| Use Case | Pattern | Converter Needed |
|----------|---------|------------------|
| React + FastAPI | Pattern 1 | `convertToModelMessages()` (frontend) |
| Python-only, building messages | Pattern 2A | None (manual ModelMessage) |
| Python-only, UIMessages in DB | Pattern 2B | `convert_to_model_messages()` (Python) |
| Simple chat, no history management | Pattern 3 | None (Vel manages) |

---

## Examples

### Full Example: React → FastAPI → OpenAI

**Frontend:**
```typescript
import { useChat, convertToModelMessages } from 'ai';

const { messages } = useChat();

// Convert and send
const modelMessages = convertToModelMessages(messages);
await fetch('/api/chat', {
  method: 'POST',
  body: JSON.stringify({ messages: modelMessages })
});
```

**Backend:**
```python
from fastapi import FastAPI
from vel import Agent

@app.post("/api/chat")
async def chat(request: dict):
    agent = Agent(
        id='chat',
        model={'provider': 'openai', 'model': 'gpt-4o'}
    )

    # ModelMessage → OpenAI format (automatic)
    response = await agent.run({'messages': request['messages']})

    return {'response': response}
```

### Full Example: Python-Only with Database

```python
from vel import Agent
from vel.utils import convert_to_model_messages

# Load conversation from DB (UIMessage format)
ui_messages = db.conversations.find_one({'user_id': user_id})['messages']

# Convert to ModelMessage
model_messages = convert_to_model_messages(ui_messages)

# Add new user message
model_messages.append({
    'role': 'user',
    'content': 'Tell me more'
})

# Send to Vel
agent = Agent(
    id='chat',
    model={'provider': 'anthropic', 'model': 'claude-3-5-sonnet-20241022'}
)

# ModelMessage → Anthropic format (automatic)
response = await agent.run({'messages': model_messages})

# Save response back to DB (UIMessage format)
ui_messages.append({
    'role': 'assistant',
    'parts': [{'type': 'text', 'text': response}]
})
db.conversations.update_one(
    {'user_id': user_id},
    {'$set': {'messages': ui_messages}}
)
```

---

## Summary

**Key Points:**

1. **UIMessage** = Frontend state (use with React/Next.js)
2. **ModelMessage** = Unified format (use with Vel)
3. **Provider Format** = API-specific (Vel handles automatically)

**Conversion Tools:**

- **Frontend:** `convertToModelMessages()` from `'ai'` package
- **Python:** `convert_to_model_messages()` from `vel.utils`

**Vel's Role:**

Vel automatically translates ModelMessage → Provider Format based on your chosen provider (OpenAI/Anthropic/Gemini). You never need to write provider-specific message formatting code.

---

## See Also

- [Vercel AI SDK ModelMessage Docs](https://ai-sdk.dev/docs/reference/ai-sdk-core/model-message)
- [Examples: vercel_ai_sdk_messages.py](/examples/vercel_ai_sdk_messages.py)
- [Examples: convert_ui_messages.py](/examples/convert_ui_messages.py)
- [Providers Documentation](/providers)
