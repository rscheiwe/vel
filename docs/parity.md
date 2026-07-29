---
layout: default
title: AI SDK V5 Parity
nav_order: 11
---

# Vercel AI SDK V5 Stream Protocol Parity

Reference for Vel's compatibility with the Vercel AI SDK V5 UI Stream Protocol.

## Overview

Vel implements the core Vercel AI SDK V5 UI Stream Protocol used by frontend components like `useChat` and `useCompletion`. Some protocol events remain tracked gaps and are called out below.

**Implemented coverage:**
- ✅ Core stream protocol events match AI SDK V5 naming and payload conventions
- ✅ Provider-executed tools support (OpenAI web_search, computer)
- ✅ Source/citation events for RAG and grounding
- ✅ Early metadata emission with usage updates
- ✅ Robust handling of malformed provider responses
- ✅ Reasoning events for o1/o3 and Claude thinking

**Reference:** [Vercel AI SDK Stream Protocol](https://sdk.vercel.ai/docs/ai-sdk-ui/stream-protocol)

---

## Parity Implementation Summary

### Gap Analysis Addressed

Based on comprehensive review of Vercel AI SDK source code and gap analysis document, Vel now addresses all identified parity gaps:

| Gap | Status | Implementation |
|-----|--------|----------------|
| **A. Automatic `start` event** | ✅ Complete | Agent emits `start` at stream initialization |
| **C. Reasoning normalization** | ✅ Complete | All variants normalized, deduplicated |
| **D. Provider-executed tools** | ✅ Complete | web_search_call, computer_call mapped |
| **E. Sources/citations** | ✅ Complete | OpenAI & Gemini sources extracted |
| **F. Metadata timing** | ✅ Complete | Early emission + usage updates |
| **I. Malformed tool_calls** | ✅ Complete | Robust index-based tracking |

---

## Event Mapping Reference

### Core Events

| AI SDK V5 Event | Vel Implementation | Notes |
|----------------|-------------------|-------|
| `start` | `StartEvent` | ✅ Auto-emitted by Agent |
| `text-start` | `TextStartEvent` | ✅ Block-scoped with stable IDs |
| `text-delta` | `TextDeltaEvent` | ✅ Streaming text chunks |
| `text-end` | `TextEndEvent` | ✅ Block completion |
| `tool-input-start` | `ToolInputStartEvent` | ✅ Tool call initiation |
| `tool-input-delta` | `ToolInputDeltaEvent` | ✅ Streaming tool arguments |
| `tool-input-available` | `ToolInputAvailableEvent` | ✅ Complete tool arguments |
| `tool-output-available` | `ToolOutputAvailableEvent` | ✅ Tool execution results |
| `tool-output-error` | `ToolOutputErrorEvent` | ✅ Recoverable tool execution errors; strict `{type, toolCallId, errorText}` shape |
| `reasoning-start` | `ReasoningStartEvent` | ✅ Reasoning block start |
| `reasoning-delta` | `ReasoningDeltaEvent` | ✅ Reasoning chunks |
| `reasoning-end` | `ReasoningEndEvent` | ✅ Reasoning completion |
| `response-metadata` | `ResponseMetadataEvent` | ✅ Early + usage updates |
| `source` | `SourceEvent` | ✅ Web/file citations |
| `start-step` | `StepStartEvent` | ✅ Multi-step agents |
| `finish-step` | `StepFinishEvent` | ✅ Step completion |
| `finish-message` | `FinishMessageEvent` | ✅ Message completion |
| `finish` | `FinishEvent` | ✅ Generation complete |
| `error` | `ErrorEvent` | ✅ Enhanced error context |
| `data-*` | `DataEvent` | ✅ Custom data streaming |

---

## Provider-Specific Parity

### OpenAI Chat Completions API

**Translation:** `OpenAIAPITranslator` (vel/providers/translators.py:43-270)

**AI SDK Reference:** `packages/openai/src/openai-chat-language-model.ts`

#### Enhancements

1. **Malformed tool_calls handling** (Gap I)
   - Issue: Some providers send `tool_calls[].type: ""` after first delta
   - Solution: Index-based tracking, defensive field extraction
   - Reference: [vercel/ai#7255](https://github.com/vercel/ai/issues/7255)

2. **Guaranteed tool-input-available** (Gap D)
   - Issue: Args may only appear at `.done` (no streaming deltas)
   - Solution: `finalize_tool_calls()` ensures emission even without deltas
   - Tracks `input_available_emitted` flag to prevent duplicates

3. **Reasoning field aliases**
   - Supports OpenAI-compatible `reasoning_content`, `reasoning`, and
     `reasoning_details`
   - Chunks that carry both content and reasoning emit both event streams
   - Reasoning blocks close before answer text starts

**Parity Status:** ✅ Core compatible, with remaining tracked events listed in the summary.

---

### OpenAI Responses API

**Translation:** `OpenAIResponsesAPITranslator` (vel/providers/translators.py:753-1118)

**AI SDK Reference:** `packages/openai/src/responses/openai-responses-language-model.ts`

#### Enhancements

1. **Reasoning normalization** (Gap C)
   - Normalizes ALL variants:
     - `response.reasoning.delta` → `reasoning-delta`
     - `response.reasoning_summary.delta` → `reasoning-delta`
     - `response.reasoning_summary_text.delta` → `reasoning-delta`
   - Deduplicates `reasoning-start` using `_seen_reasoning_ids` set
   - Stable block IDs from OpenAI's item IDs
   - Reference: [vercel/ai#6742](https://github.com/vercel/ai/issues/6742)

2. **Provider-executed tools** (Gap D)
   - Maps `web_search_call` and `computer_call` to `tool-output-available`
   - Includes metadata: `providerExecuted: true`, `toolType`, `providerName`
   - Guaranteed `tool-input-available` emission (handles args-at-.done case)
   - Reference: [vercel/ai#5164](https://github.com/vercel/ai/discussions/5164)

3. **Sources/citations** (Gap E)
   - Extracts web search sources from `web_search_call.result.sources`
   - Maps to `SourceEvent` with structure:
     ```json
     {
       "type": "web",
       "url": "...",
       "title": "...",
       "snippet": "...",
       "sourceId": "..." // Preserved from OpenAI
     }
     ```
   - Handles file citations (preserves `file_id`)

4. **Metadata timing** (Gap F)
   - Emits `ResponseMetadataEvent` on `response.created` with id/model
   - Updates with usage data on `response.completed`
   - Follows AI SDK pattern: early metadata → usage update

**Parity Status:** ✅ Core compatible, with remaining tracked events listed in the summary.

---

### Anthropic Messages API

**Translation:** `AnthropicAPITranslator` (vel/providers/translators.py:404-634)

**AI SDK Reference:** `packages/anthropic/src/anthropic-messages-language-model.ts`

#### Enhancements

1. **Metadata timing** (Gap F)
   - Emits early metadata on `message_start` with id/model
   - Updates with usage on `message_delta`
   - Tracks `_metadata_emitted` flag for conditional emission

2. **Thinking blocks** (native support)
   - Maps Anthropic `thinking` content blocks to `reasoning-*` events
   - Fully visible reasoning (unlike OpenAI's encrypted reasoning)

**Parity Status:** ✅ Core compatible, with remaining tracked events listed in the summary.

---

### Google Gemini API

**Translation:** `GeminiAPITranslator` (vel/providers/translators.py:637-775)

**AI SDK Reference:** `packages/google/src/google-generative-ai-language-model.ts`

#### Enhancements

1. **Complete function call events** (Gap D related)
   - Issue: Gemini emits complete function calls (no streaming)
   - Solution: Queue `tool-input-available` after `tool-input-start`
   - Provider drains pending events via `get_pending_event()`

2. **Grounding sources** (native support)
   - Extracts sources from `grounding_metadata.grounding_sources`
   - Maps to `SourceEvent` with web citation structure
   - Deduplicates sources by URL

**Parity Status:** ✅ Core compatible, with remaining tracked events listed in the summary.

---

## Design Decisions

### 1. Opt-In Backwards Compatibility

All parity enhancements are **backwards compatible**:

```python
# Existing code works unchanged
agent = Agent(id='chat', model={'provider': 'openai', 'model': 'gpt-4o'})
async for event in agent.run_stream({'message': 'Hello'}):
    print(event)
```

New features activate automatically based on provider capabilities:
- OpenAI Responses: provider-executed tools, sources
- Gemini: grounding sources
- Anthropic: thinking blocks

### 2. Early Metadata Emission

**Rationale:** Matches AI SDK behavior for better UX

**Implementation:**
- Emit metadata as soon as id/model available
- Update with usage when available
- Frontend can display model info immediately

**Example:**
```python
# First metadata event
{"type": "response-metadata", "id": "resp_123", "modelId": "gpt-4o", "usage": null}

# Usage update event
{"type": "response-metadata", "id": "resp_123", "usage": {"totalTokens": 150}}
```

### 3. Guaranteed tool-input-available

**Rationale:** Some providers only send args at completion (no deltas)

**Implementation:**
- Track emission state: `input_available_emitted` flag
- Fallback in `finalize_tool_calls()` ensures emission
- Prevents duplicate emissions

**Benefit:** Reliable tool execution even with non-streaming args

### 4. Reasoning Variant Normalization

**Rationale:** OpenAI uses multiple reasoning event names

**Implementation:**
- Normalize to single event type: `reasoning-delta`
- Deduplicate starts: `_seen_reasoning_ids` set
- Stable block IDs from provider

**Variants handled:**
- `response.reasoning.delta`
- `response.reasoning_summary.delta`
- `response.reasoning_summary_text.delta`

### 5. Provider Metadata Preservation

**Rationale:** Preserve provider-specific IDs for tracing

**Implementation:**
- `providerMetadata` in tool events
- `sourceId` in source events
- `providerExecuted` flag for server-side tools

**Example:**
```json
{
  "type": "tool-output-available",
  "providerMetadata": {
    "providerExecuted": true,
    "providerName": "openai",
    "toolType": "web_search_call"
  }
}
```

---

## Testing Strategy

### Golden Trace Testing

The user will provide a golden trace test harness to verify parity. The approach:

1. **Neutral trace format**: JSON-Lines with normalized keys
   ```json
   {"t":"start"}
   {"t":"text-start","id":"m1"}
   {"t":"text-delta","id":"m1","d":"Hello"}
   ```

2. **Parallel execution**: Run same prompt through Vel + AI SDK

3. **Relaxed comparison**:
   - Match event types and order
   - Compare final text (not per-chunk boundaries)
   - Treat `finish-message` as optional
   - Allow metadata re-emission

### Test Scenarios

Key scenarios to validate:

1. **Plain text**: Short and long responses
2. **Tool calls streaming**: OpenAI Chat with streaming args
3. **Tool calls non-streaming**: Args only at `.done`
4. **Provider-executed tools**: OpenAI Responses `web_search_call`
5. **Reasoning**:
   - OpenAI o1/o3: Encrypted deltas (empty strings)
   - Anthropic thinking: Visible deltas
6. **Sources**: Web search results, file citations
7. **Error paths**: Early and late errors
8. **Multi-step**: Tool loops with step markers

---

## Implementation Files

### Core Translators

**File:** `vel/providers/translators.py`

| Translator | Lines | Key Features |
|------------|-------|--------------|
| `OpenAIAPITranslator` | 43-270 | Malformed tool_calls, guaranteed tool-input-available |
| `OpenAIResponsesAPITranslator` | 753-1118 | Reasoning normalization, provider tools, sources, metadata timing |
| `AnthropicAPITranslator` | 404-634 | Metadata timing, thinking blocks |
| `GeminiAPITranslator` | 637-775 | Complete function calls, grounding sources |

### Event Definitions

**File:** `vel/events.py`

All stream protocol events:
- `StartEvent`, `TextStartEvent`, `TextDeltaEvent`, `TextEndEvent`
- `ToolInputStartEvent`, `ToolInputDeltaEvent`, `ToolInputAvailableEvent`, `ToolOutputAvailableEvent`
- `ReasoningStartEvent`, `ReasoningDeltaEvent`, `ReasoningEndEvent`
- `ResponseMetadataEvent`, `SourceEvent`, `FileEvent`
- `StepStartEvent`, `StepFinishEvent`, `FinishMessageEvent`, `FinishEvent`
- `ErrorEvent`, `DataEvent`

### Provider Implementations

**Files:**
- `vel/providers/openai.py` - OpenAI Chat + Responses providers
- `vel/providers/anthropic.py` - Anthropic Messages provider
- `vel/providers/google.py` - Google Gemini provider

---

## AI SDK Source References

### Key Files Referenced

These are the Vercel AI SDK source files used as parity references:

1. **OpenAI Responses API**
   - File: `packages/openai/src/responses/openai-responses-language-model.ts`
   - GitHub: [vercel/ai#5164](https://github.com/vercel/ai/discussions/5164)
   - Features: reasoning, web_search, computer, sources

2. **Anthropic Messages API**
   - File: `packages/anthropic/src/anthropic-messages-language-model.ts`
   - GitHub: [vercel/ai#9540](https://github.com/vercel/ai/issues/9540)
   - Features: thinking blocks, max tokens handling

3. **Google Gemini API**
   - File: `packages/google/src/google-generative-ai-language-model.ts`
   - GitHub: [vercel/ai#4661](https://github.com/vercel/ai/issues/4661)
   - Features: grounding sources, function calls

4. **Stream Text Core**
   - File: `packages/ai/core/generate-text/stream-text.ts`
   - GitHub: [vercel/ai#4609](https://github.com/vercel/ai/discussions/4609)
   - Features: part batching, UI stream protocol serialization

5. **Known Issues**
   - [vercel/ai#7255](https://github.com/vercel/ai/issues/7255) - Malformed tool_calls with empty type
   - [vercel/ai#6742](https://github.com/vercel/ai/issues/6742) - Reasoning summary splitting

---

## Known Differences

### Intentional Enhancements

Vel includes enhancements beyond AI SDK baseline:

1. **Enhanced error events**
   - Additional fields: `statusCode`, `errorType`, `provider`, `details`
   - Better debugging context
   - AI SDK compatible (includes base `error` field)

2. **RLM (Recursive Language Model) events**
   - Custom `data-rlm-*` events for long-context reasoning
   - Optional feature, disabled by default
   - See: RLM documentation

3. **Memory system integration**
   - Optional FactStore and ReasoningBank
   - Runtime-owned, not part of stream protocol
   - See: Memory documentation

### Minor Variations

1. **finish-message timing**
   - Vel: Always emits `finish-message`
   - AI SDK: May skip in some codepaths
   - Impact: None (golden trace comparator treats as optional)

2. **Metadata emission count**
   - Vel: May emit 1-2 metadata events (early + usage)
   - AI SDK: Varies by provider
   - Impact: None (both valid per spec)

---

## Future Work

### Potential Enhancements

1. **Additional providers**
   - Groq (Llama hosting)
   - AWS Bedrock (Llama on Bedrock)
   - Perplexity (Sonar)

2. **Extended protocol features**
   - File attachments streaming
   - Code execution results
   - Multi-modal content

3. **Performance optimizations**
   - Event batching strategies
   - Incremental JSON parsing
   - Memory-efficient streaming

---

## Migration Guide

### For Existing Vel Users

**No breaking changes!** All existing code continues to work:

```python
# Existing code - unchanged
agent = Agent(id='chat', model={'provider': 'openai', 'model': 'gpt-4o'})
async for event in agent.run_stream({'message': 'Hello'}):
    if event['type'] == 'text-delta':
        print(event['delta'], end='')
```

**New features activate automatically:**

```python
# OpenAI Responses with web search
agent = Agent(
    id='search-agent',
    model={'provider': 'openai-responses', 'model': 'gpt-4o'}
)

async for event in agent.run_stream({'message': 'Latest AI news'}):
    # Provider-executed web search
    if event['type'] == 'tool-output-available':
        metadata = event.get('providerMetadata', {})
        if metadata.get('providerExecuted'):
            print(f"Web search by {metadata['providerName']}")

    # Sources from web search
    if event['type'] == 'source':
        for src in event['sources']:
            print(f"Source: {src['title']} - {src['url']}")

    # Early metadata
    if event['type'] == 'response-metadata':
        if event.get('modelId'):
            print(f"Using model: {event['modelId']}")
```

### For AI SDK Users

**Drop-in compatibility** with Vercel AI SDK frontend components:

```typescript
// Frontend using Vercel AI SDK useChat
import { useChat } from 'ai/react';

const { messages } = useChat({
  api: '/api/chat', // Points to Vel backend
});

// Works seamlessly - same event format
```

---

## Summary

Vel covers the core Vercel AI SDK V5 UI Stream Protocol through:

✅ **Provider-executed tools** (web_search, computer)
✅ **Sources/citations** (web, file)
✅ **Reasoning normalization** (all variants)
✅ **Metadata timing** (early + updates)

Tracked gaps include `tool-output-denied`, `abort`, and `message-metadata`
events.
✅ **Robust error handling** (malformed responses)
✅ **Backwards compatible** (no breaking changes)

**Result:** Seamless integration with Vercel AI SDK frontend components while maintaining Vel's 12-Factor Agent principles.

---

## References

- [Vercel AI SDK Stream Protocol](https://sdk.vercel.ai/docs/ai-sdk-ui/stream-protocol)
- [Vercel AI SDK GitHub](https://github.com/vercel/ai)
- [Vel Stream Protocol Documentation](stream-protocol.md)
- [12-Factor Agents](https://github.com/humanlayer/12-factor-agents)
