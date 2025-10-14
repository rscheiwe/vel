# Vel Stream Protocol Parity Remediation Plan

**Date:** 2025-10-14
**Target:** Achieve 100% parity with Vercel AI SDK V3 stream protocol
**Current Status:** 72% parity (18/25 event types)

## Overview

This plan outlines specific implementation steps to achieve full parity with Vercel AI SDK's V3 stream protocol across all three providers (OpenAI, Anthropic, Google Gemini).

**Reference Documents:**
- `PARITY_COMPARISON_MATRIX.md` - Detailed gap analysis
- `PARITY_VALIDATION_PLAN.md` - Original validation plan
- Vercel AI SDK: https://github.com/vercel/ai

---

## Phase 1: Critical Parity (P0) - MUST HAVE

**Timeline:** 1-2 weeks
**Goal:** Core functionality parity with V3

### Task 1.1: Add ResponseMetadataEvent

**Priority:** P0
**Complexity:** Medium
**Impact:** High (enables token tracking, debugging)

**Implementation:**

1. **Add event class to `vel/events.py`:**

```python
@dataclass
class ResponseMetadataEvent(StreamEvent):
    """Response metadata (usage, model info, timing)"""
    type: Literal['response-metadata'] = 'response-metadata'
    id: Optional[str] = None
    model_id: Optional[str] = None
    timestamp: Optional[str] = None  # ISO 8601
    usage: Optional[Dict[str, int]] = None  # {promptTokens, completionTokens, totalTokens}

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        if self.id:
            d['id'] = self.id
        if self.model_id:
            d['modelId'] = self.model_id
        if self.timestamp:
            d['timestamp'] = self.timestamp
        if self.usage:
            d['usage'] = self.usage
        return d
```

2. **Update `EventType` literal:**

```python
EventType = Literal[
    'start',
    'text-start',
    'text-delta',
    'text-end',
    'reasoning-start',
    'reasoning-delta',
    'reasoning-end',
    'tool-input-start',
    'tool-input-delta',
    'tool-input-available',
    'tool-output-available',
    'response-metadata',  # NEW
    'start-step',
    'finish-step',
    'finish-message',
    'error'
]
```

3. **Extract usage in `OpenAIAPITranslator`:**

```python
def translate_chunk(self, chunk: Dict[str, Any]) -> Optional[StreamEvent]:
    # ... existing code ...

    # Handle usage (in final chunk)
    usage = chunk.get('usage')
    if usage:
        return ResponseMetadataEvent(
            model_id=chunk.get('model'),
            usage={
                'promptTokens': usage.get('prompt_tokens', 0),
                'completionTokens': usage.get('completion_tokens', 0),
                'totalTokens': usage.get('total_tokens', 0)
            }
        )
```

4. **Extract usage in `AnthropicAPITranslator`:**

```python
def translate_event(self, data: Dict[str, Any]) -> Optional[StreamEvent]:
    # ... existing code ...

    # Handle message_delta (usage)
    elif event_type == 'message_delta':
        delta = data.get('delta', {})
        usage = data.get('usage')  # Input tokens here
        if usage:
            self._input_tokens = usage.get('input_tokens', 0)

    # Handle message_stop (final usage)
    elif event_type == 'message_stop':
        # Emit metadata before finish
        if hasattr(self, '_input_tokens') or hasattr(self, '_output_tokens'):
            metadata_event = ResponseMetadataEvent(
                usage={
                    'promptTokens': getattr(self, '_input_tokens', 0),
                    'completionTokens': getattr(self, '_output_tokens', 0),
                    'totalTokens': getattr(self, '_input_tokens', 0) + getattr(self, '_output_tokens', 0)
                }
            )
            # Need to yield both metadata and finish - requires buffering
            # or changing return type to list
```

5. **Extract usage in `GeminiAPITranslator`:**

```python
def translate_chunk(self, chunk: Any) -> Optional[StreamEvent]:
    # ... existing code ...

    # Handle usage_metadata
    if hasattr(chunk, 'usage_metadata'):
        usage = chunk.usage_metadata
        return ResponseMetadataEvent(
            usage={
                'promptTokens': getattr(usage, 'prompt_token_count', 0),
                'completionTokens': getattr(usage, 'candidates_token_count', 0),
                'totalTokens': getattr(usage, 'total_token_count', 0)
            }
        )
```

6. **Update exports in `vel/events.py`**

**Testing:**
- Test usage tracking with all providers
- Verify metadata appears in stream
- Test with and without usage data

---

### Task 1.2: Add Anthropic Thinking Block Support

**Priority:** P0
**Complexity:** Medium
**Impact:** High (missing native Anthropic feature)

**Implementation:**

1. **Update `AnthropicAPITranslator.translate_event()`:**

```python
# In content_block_start handler
elif event_type == 'content_block_start':
    index = data.get('index', 0)
    content_block = data.get('content_block', {})
    block_type = content_block.get('type')

    # ... existing text handler ...

    # NEW: Handle thinking blocks
    elif block_type == 'thinking':
        block_id = str(uuid.uuid4())
        self._content_blocks[index] = {
            'type': 'thinking',
            'block_id': block_id,
            'buffer': []
        }
        return ReasoningStartEvent(block_id=block_id)

    # ... existing tool_use handler ...
```

2. **Handle thinking deltas:**

```python
# In content_block_delta handler
elif event_type == 'content_block_delta':
    index = data.get('index', 0)
    delta = data.get('delta', {})
    delta_type = delta.get('type')

    if index in self._content_blocks:
        block = self._content_blocks[index]

        # ... existing text_delta handler ...

        # NEW: Handle thinking deltas
        elif delta_type == 'thinking_delta':
            thinking = delta.get('thinking', '')
            block['buffer'].append(thinking)
            return ReasoningDeltaEvent(
                block_id=block['block_id'],
                delta=thinking
            )

        # ... existing input_json_delta handler ...
```

3. **Handle thinking block end:**

```python
# In content_block_stop handler
elif event_type == 'content_block_stop':
    index = data.get('index', 0)
    if index in self._content_blocks:
        block = self._content_blocks[index]

        # ... existing text handler ...

        # NEW: Handle thinking end
        elif block['type'] == 'thinking':
            return ReasoningEndEvent(block_id=block['block_id'])

        # ... existing tool_use handler ...
```

**Testing:**
- Test with Anthropic models that support thinking (Claude 3.7 Sonnet+)
- Verify reasoning events are emitted
- Test mixed content (text + thinking + tools)

---

### Task 1.3: Add SourceEvent for Citations

**Priority:** P0
**Complexity:** Medium
**Impact:** High (missing grounding/provenance)

**Implementation:**

1. **Add event class to `vel/events.py`:**

```python
@dataclass
class SourceEvent(StreamEvent):
    """Source/citation event (web search results, document references)"""
    type: Literal['source'] = 'source'
    sources: list[Dict[str, Any]] = None  # [{type, url, title, snippet}, ...]

    def __post_init__(self):
        if self.sources is None:
            self.sources = []

    def to_dict(self) -> Dict[str, Any]:
        return {
            **super().to_dict(),
            'sources': self.sources
        }
```

2. **Update `EventType` literal:**

```python
EventType = Literal[
    # ... existing ...
    'response-metadata',
    'source',  # NEW
    'start-step',
    # ... rest ...
]
```

3. **Handle Google grounding in `GeminiAPITranslator`:**

```python
class GeminiAPITranslator:
    def __init__(self):
        self._text_block_id: Optional[str] = None
        self._seen_source_urls: set[str] = set()  # Deduplicate

    def translate_chunk(self, chunk: Any) -> Optional[StreamEvent]:
        # ... existing code ...

        # NEW: Handle grounding metadata
        if hasattr(chunk, 'candidates'):
            for candidate in chunk.candidates:
                if hasattr(candidate, 'grounding_metadata'):
                    metadata = candidate.grounding_metadata
                    if hasattr(metadata, 'grounding_sources'):
                        sources = []
                        for source in metadata.grounding_sources:
                            # Extract web source
                            if hasattr(source, 'web'):
                                web = source.web
                                url = getattr(web, 'uri', '')

                                # Deduplicate
                                if url and url not in self._seen_source_urls:
                                    self._seen_source_urls.add(url)
                                    sources.append({
                                        'type': 'web',
                                        'url': url,
                                        'title': getattr(web, 'title', ''),
                                    })

                        if sources:
                            return SourceEvent(sources=sources)

    def reset(self):
        self._text_block_id = None
        self._seen_source_urls.clear()
```

4. **Handle OpenAI annotations in `OpenAIAPITranslator`:**

```python
def translate_chunk(self, chunk: Dict[str, Any]) -> Optional[StreamEvent]:
    delta = chunk.get('choices', [{}])[0].get('delta', {})

    # ... existing code ...

    # NEW: Handle annotations (URL citations)
    if 'annotations' in delta:
        annotations = delta['annotations']
        sources = []
        for ann in annotations:
            if ann.get('type') == 'url_citation':
                url = ann.get('url', '')
                if url:
                    sources.append({
                        'type': 'web',
                        'url': url,
                        'title': ann.get('title', ''),
                        'snippet': ann.get('text', '')
                    })

        if sources:
            return SourceEvent(sources=sources)
```

**Testing:**
- Test Gemini with grounding enabled
- Test OpenAI with URL citations
- Verify deduplication works

---

### Task 1.4: Enhance ErrorEvent

**Priority:** P0
**Complexity:** Low
**Impact:** Medium (better debugging)

**Implementation:**

1. **Update `ErrorEvent` in `vel/events.py`:**

```python
@dataclass
class ErrorEvent(StreamEvent):
    """Error event"""
    type: Literal['error'] = 'error'
    error: str = ''
    error_code: Optional[str] = None  # NEW
    error_type: Optional[str] = None  # NEW

    def to_dict(self) -> Dict[str, Any]:
        d = {**super().to_dict(), 'error': self.error}
        if self.error_code:
            d['errorCode'] = self.error_code
        if self.error_type:
            d['errorType'] = self.error_type
        return d
```

2. **Extract error details in all translators:**

```python
# OpenAIAPITranslator
# (errors typically come from HTTP layer, not chunks)

# AnthropicAPITranslator
elif event_type == 'error':
    error_data = data.get('error', {})
    return ErrorEvent(
        error=error_data.get('message', 'Unknown error'),
        error_code=error_data.get('code'),  # NEW
        error_type=error_data.get('type')   # NEW
    )

# GeminiAPITranslator
# (errors typically come as exceptions)
```

**Testing:**
- Test error scenarios with all providers
- Verify error details are captured

---

## Phase 2: Feature Completeness (P1) - SHOULD HAVE

**Timeline:** 2-3 weeks
**Goal:** Advanced features for production use

### Task 2.1: Add FileEvent

**Priority:** P1
**Complexity:** Medium
**Impact:** Medium (multi-modal support)

**Implementation:**

1. **Add event class to `vel/events.py`:**

```python
@dataclass
class FileEvent(StreamEvent):
    """File attachment event (inline data, images, PDFs)"""
    type: Literal['file'] = 'file'
    content: Any = None  # base64 string or bytes
    name: str = ''
    mime_type: str = ''

    def to_dict(self) -> Dict[str, Any]:
        return {
            **super().to_dict(),
            'content': self.content,
            'name': self.name,
            'mimeType': self.mime_type
        }
```

2. **Handle inline_data in `GeminiAPITranslator`:**

```python
def translate_chunk(self, chunk: Any) -> Optional[StreamEvent]:
    # ... existing code ...

    # Handle inline data (files)
    if hasattr(chunk, 'parts'):
        for part in chunk.parts:
            if hasattr(part, 'inline_data'):
                inline = part.inline_data
                return FileEvent(
                    content=getattr(inline, 'data', ''),  # base64
                    mime_type=getattr(inline, 'mime_type', '')
                )
```

**Testing:**
- Test Gemini with image inputs
- Test Gemini with PDF inputs

---

### Task 2.2: Standardize Block ID Generation

**Priority:** P1
**Complexity:** Low
**Impact:** Medium (consistency)

**Decision needed:**
- **Option A:** Use indices (like V3) - `'0'`, `'1'`, `'2'`
- **Option B:** Use UUIDs (current) - `'550e8400-e29b-41d4-a716-446655440000'`
- **Option C:** Use provider IDs where available, fallback to indices

**Recommendation:** Option A (indices) for consistency with V3

**Implementation:**

```python
# OpenAIAPITranslator
def __init__(self):
    self._text_block_id: Optional[str] = None
    self._next_block_index: int = 0  # NEW

def translate_chunk(self, chunk: Dict[str, Any]) -> Optional[StreamEvent]:
    if 'content' in delta and delta['content']:
        if self._text_block_id is None:
            self._text_block_id = str(self._next_block_index)  # Changed from UUID
            self._next_block_index += 1
            return TextStartEvent(block_id=self._text_block_id)
```

---

### Task 2.3: Add Gemini Code Execution Support

**Priority:** P1
**Complexity:** Low
**Impact:** Low (niche feature)

**Implementation:**

```python
# GeminiAPITranslator
def translate_chunk(self, chunk: Any) -> Optional[StreamEvent]:
    # ... existing code ...

    if hasattr(chunk, 'parts'):
        for part in chunk.parts:
            # Handle executable code
            if hasattr(part, 'executable_code'):
                code = part.executable_code
                # Could emit as custom event or metadata
                # For now, skip or log

            # Handle code execution result
            if hasattr(part, 'code_execution_result'):
                result = part.code_execution_result
                # Could emit as metadata or custom event
```

**Note:** This is a niche feature. Consider if needed for MVP.

---

### Task 2.4: Add Gemini Safety Ratings

**Priority:** P1
**Complexity:** Low
**Impact:** Medium (content moderation)

**Implementation:**

```python
# Add to ResponseMetadataEvent
@dataclass
class ResponseMetadataEvent(StreamEvent):
    # ... existing fields ...
    safety_ratings: Optional[List[Dict[str, Any]]] = None  # NEW

# GeminiAPITranslator
def translate_chunk(self, chunk: Any) -> Optional[StreamEvent]:
    # ... existing code ...

    if hasattr(chunk, 'candidates'):
        for candidate in chunk.candidates:
            if hasattr(candidate, 'safety_ratings'):
                ratings = []
                for rating in candidate.safety_ratings:
                    ratings.append({
                        'category': str(rating.category),
                        'probability': str(rating.probability),
                        'blocked': getattr(rating, 'blocked', False)
                    })

                return ResponseMetadataEvent(
                    safety_ratings=ratings
                )
```

---

## Phase 3: Alignment & Polish (P2) - NICE TO HAVE

**Timeline:** 1-2 weeks
**Goal:** Full alignment with V3, optimization

### Task 3.1: Event Naming Alignment

**Priority:** P2
**Complexity:** Medium (backward compatibility)
**Impact:** Low (cosmetic)

**Options:**

**Option A: Hard rename (breaking change)**
```python
# Rename in vel/events.py
'tool-input-available' → 'tool-call'
'tool-output-available' → 'tool-result'
```

**Option B: Aliases (backward compatible)**
```python
# Add event type aliases
EventType = Literal[
    # ... existing ...
    'tool-call',  # Alias for tool-input-available
    'tool-result',  # Alias for tool-output-available
]

# Support both in translators
class ToolCallEvent(StreamEvent):  # NEW name
    type: Literal['tool-call'] = 'tool-call'
    # ... same fields as ToolInputAvailableEvent ...

# Deprecate old names in docs
```

**Recommendation:** Option B (aliases) for backward compatibility

---

### Task 3.2: Review Custom Events

**Priority:** P2
**Complexity:** Low
**Impact:** Low (documentation)

**Tasks:**
- Document `start-step`, `finish-step` as Vel-specific extensions
- Evaluate if needed or can be deprecated
- Add to V3 compatibility docs

---

### Task 3.3: Update Documentation

**Priority:** P2
**Complexity:** Low
**Impact:** High (user-facing)

**Files to update:**
- `docs/stream-protocol.md` - Add all new events
- `docs/event-translators.md` - Update translator capabilities
- `README.md` - Update feature list
- `CLAUDE.md` - Update architecture docs

---

## Implementation Order

### Week 1-2: P0 Tasks
1. Day 1-2: Task 1.1 (ResponseMetadataEvent)
2. Day 3-4: Task 1.2 (Thinking blocks)
3. Day 5-6: Task 1.3 (SourceEvent)
4. Day 7: Task 1.4 (ErrorEvent enhancement)
5. Day 8-10: Testing, bug fixes

### Week 3-4: P1 Tasks (if needed)
1. Day 11-12: Task 2.1 (FileEvent)
2. Day 13-14: Task 2.2 (Block ID standardization)
3. Day 15: Task 2.4 (Safety ratings)
4. Day 16-17: Testing

### Week 5: P2 Tasks (polish)
1. Day 18-19: Task 3.1 (Event naming)
2. Day 20-21: Task 3.3 (Documentation)
3. Day 22: Final testing

---

## Testing Strategy

### Unit Tests

For each new event:
```python
# tests/test_translators.py
class TestOpenAIAPITranslator:
    def test_metadata_extraction(self):
        translator = OpenAIAPITranslator()
        chunk = {
            'usage': {
                'prompt_tokens': 10,
                'completion_tokens': 20,
                'total_tokens': 30
            },
            'model': 'gpt-4o'
        }
        event = translator.translate_chunk(chunk)
        assert isinstance(event, ResponseMetadataEvent)
        assert event.usage['promptTokens'] == 10
        assert event.model_id == 'gpt-4o'
```

### Integration Tests

Test with real providers:
```python
# tests/test_provider_parity.py
async def test_openai_metadata():
    provider = OpenAIProvider()
    events = []
    async for event in provider.stream(
        messages=[{'role': 'user', 'content': 'Hello'}],
        model='gpt-4o',
        tools={}
    ):
        events.append(event)

    # Verify metadata event exists
    metadata_events = [e for e in events if e.type == 'response-metadata']
    assert len(metadata_events) > 0
    assert metadata_events[0].usage is not None
```

---

## Success Criteria

### P0 (Must Have)
- ✅ ResponseMetadataEvent implemented and tested
- ✅ Anthropic thinking blocks fully supported
- ✅ SourceEvent implemented for Gemini + OpenAI
- ✅ ErrorEvent enhanced with code/type
- ✅ All P0 tests passing

### P1 (Should Have)
- ✅ FileEvent implemented for Gemini
- ✅ Block IDs standardized
- ✅ Safety ratings captured
- ✅ All P1 tests passing

### P2 (Nice to Have)
- ✅ Event naming aligned or aliased
- ✅ Documentation updated
- ✅ All tests passing

### Final Parity Score
- **Target:** 95%+ parity (24/25 events)
- **Acceptable:** 90%+ parity (23/25 events)

---

## Risks & Mitigation

### Risk 1: Breaking Changes
**Mitigation:** Use feature flags, aliases for backward compatibility

### Risk 2: Provider API Changes
**Mitigation:** Pin to specific SDK versions, add version detection

### Risk 3: Performance Impact
**Mitigation:** Profile new code, optimize hot paths

### Risk 4: Complexity Creep
**Mitigation:** Keep translators focused, avoid overengineering

---

## Next Steps

1. **Review this plan** - Get team approval
2. **Create GitHub issues** - One per task
3. **Set up project board** - Track progress
4. **Begin Phase 1** - Start with Task 1.1 (ResponseMetadataEvent)

---

**Document Status:** Ready for implementation
**Approval Status:** Pending review
**Owner:** TBD
**Last Updated:** 2025-10-14
