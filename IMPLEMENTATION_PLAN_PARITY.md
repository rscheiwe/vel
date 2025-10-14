# Vel Stream Protocol Parity Implementation Plan

**Date:** 2025-10-14
**Branch:** main
**Goal:** Achieve 95%+ parity with Vercel AI SDK V3 stream protocol
**Current Parity:** 72% (18/25 event types)
**Target Parity:** 95%+ (24/25 event types)

**Exclusions (as requested):**
- ❌ Phase 2, Task 2.4: Safety ratings (skipping for now)
- ❌ Phase 3, Task 3.1: Event naming alignment (skipping for now)

---

## Table of Contents

1. [Overview](#overview)
2. [Pre-Implementation Checklist](#pre-implementation-checklist)
3. [Phase 1: Critical Events (P0)](#phase-1-critical-events-p0)
4. [Phase 2: Feature Completeness (P1)](#phase-2-feature-completeness-p1)
5. [Phase 3: Documentation & Polish (P2)](#phase-3-documentation--polish-p2)
6. [Testing Strategy](#testing-strategy)
7. [Validation Checklist](#validation-checklist)
8. [Rollback Plan](#rollback-plan)

---

## Overview

### Scope of Work

**Files to Modify:**
1. `vel/events.py` - Add 3 new event types (ResponseMetadata, Source, File)
2. `vel/providers/translators.py` - Update all 4 translators
3. `vel/providers/openai.py` - Update provider if needed
4. `vel/providers/anthropic.py` - Update provider if needed
5. `vel/providers/google.py` - Update provider if needed
6. `tests/test_translators.py` - Add comprehensive tests (create if needed)
7. `docs/stream-protocol.md` - Document all new events
8. `docs/event-translators.md` - Update translator capabilities
9. `README.md` - Update feature list

**New Event Types to Add:**
1. `ResponseMetadataEvent` - Token usage, model info
2. `SourceEvent` - Citations, grounding sources
3. `FileEvent` - Inline file data (images, PDFs)

**New Event Handlers to Add:**
1. Anthropic thinking blocks (3 event types: start/delta/end)
2. Gemini grounding sources
3. OpenAI annotations
4. Gemini inline data
5. Gemini code execution (basic support)
6. Enhanced error details (all providers)
7. Metadata extraction (all providers)

**Lines of Code Estimate:**
- Events: ~100 lines
- Translators: ~300 lines
- Tests: ~400 lines
- Docs: ~200 lines
- **Total:** ~1000 lines

---

## Pre-Implementation Checklist

Before starting any code changes:

- [ ] **Backup current state**
  - [ ] Create feature branch: `git checkout -b feat/stream-protocol-parity`
  - [ ] Verify clean working directory: `git status`
  - [ ] Run existing tests to establish baseline: `pytest -v`

- [ ] **Review current implementation**
  - [ ] Read `vel/events.py` completely
  - [ ] Read all 4 translators in `vel/providers/translators.py`
  - [ ] Understand current event flow
  - [ ] Document any edge cases found

- [ ] **Set up testing environment**
  - [ ] Verify API keys are set (OPENAI_API_KEY, ANTHROPIC_API_KEY, GOOGLE_API_KEY)
  - [ ] Create test fixtures for new events
  - [ ] Set up test data directory if needed

- [ ] **Create implementation tracking**
  - [ ] Use TodoWrite tool to track all tasks
  - [ ] Mark tasks as in_progress/completed as we go
  - [ ] Document any blockers immediately

---

## Phase 1: Critical Events (P0)

**Goal:** Add missing critical event types and handlers
**Timeline:** Tasks 1.1 → 1.2 → 1.3 → 1.4 (sequential)
**Estimated Effort:** ~4-6 hours

---

### Task 1.1: Add ResponseMetadataEvent

**Priority:** P0 - CRITICAL
**Complexity:** Medium
**Files Modified:** `vel/events.py`, `vel/providers/translators.py` (all 4 translators)

#### Subtasks

- [ ] **1.1.1: Add ResponseMetadataEvent class to vel/events.py**
  - [ ] Add dataclass with fields: id, model_id, timestamp, usage
  - [ ] Implement `to_dict()` method with proper field mapping
  - [ ] Add to `EventType` literal
  - [ ] Export in `__all__`

  **Code to add:**
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

- [ ] **1.1.2: Update EventType literal**
  - [ ] Add `'response-metadata'` to EventType
  - [ ] Verify alphabetical ordering (if applicable)

- [ ] **1.1.3: Extract metadata in OpenAIAPITranslator**
  - [ ] Handle `usage` field in chunks
  - [ ] Extract `model` field
  - [ ] Map to ResponseMetadataEvent
  - [ ] Add to `translate_chunk()` method

  **Implementation location:** `OpenAIAPITranslator.translate_chunk()`

  **Code pattern:**
  ```python
  # In translate_chunk(), after existing handlers
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

- [ ] **1.1.4: Extract metadata in AnthropicAPITranslator**
  - [ ] Track input tokens from `message_delta` event
  - [ ] Track output tokens from `message_stop` event
  - [ ] Need to handle two-phase usage (input separate from output)
  - [ ] Add state tracking: `_input_tokens`, `_output_tokens`
  - [ ] Emit metadata before finish event

  **Challenge:** `translate_event()` returns single event, need to emit metadata + finish

  **Solution:** Return list from `translate_event()` OR buffer metadata and emit in `message_stop`

  **Approach:** Add `get_pending_events()` method that returns buffered events

- [ ] **1.1.5: Extract metadata in GeminiAPITranslator**
  - [ ] Handle `usage_metadata` in chunks
  - [ ] Extract prompt_token_count, candidates_token_count, total_token_count
  - [ ] Map to ResponseMetadataEvent

  **Implementation location:** `GeminiAPITranslator.translate_chunk()`

- [ ] **1.1.6: Update imports in vel/events.py**
  - [ ] Ensure ResponseMetadataEvent is imported in translators

- [ ] **1.1.7: Write tests for ResponseMetadataEvent**
  - [ ] Test OpenAI usage extraction
  - [ ] Test Anthropic two-phase usage
  - [ ] Test Gemini usage extraction
  - [ ] Test `to_dict()` formatting

- [ ] **1.1.8: Manual validation**
  - [ ] Test with real OpenAI API call
  - [ ] Test with real Anthropic API call
  - [ ] Test with real Gemini API call
  - [ ] Verify metadata appears in stream

**Acceptance Criteria:**
- ✅ ResponseMetadataEvent class exists in vel/events.py
- ✅ All 3 translators extract usage correctly
- ✅ Tests pass for all providers
- ✅ Manual testing confirms metadata in streams

---

### Task 1.2: Add Anthropic Thinking Block Support

**Priority:** P0 - CRITICAL
**Complexity:** Medium
**Files Modified:** `vel/providers/translators.py` (AnthropicAPITranslator only)

#### Subtasks

- [ ] **1.2.1: Verify ReasoningEvents exist in vel/events.py**
  - [ ] Confirm ReasoningStartEvent exists
  - [ ] Confirm ReasoningDeltaEvent exists
  - [ ] Confirm ReasoningEndEvent exists
  - [ ] Review their `to_dict()` implementations

- [ ] **1.2.2: Handle thinking block start**
  - [ ] In `translate_event()`, handle `content_block_start` with type='thinking'
  - [ ] Generate unique block_id (UUID)
  - [ ] Store in `_content_blocks[index]`
  - [ ] Return ReasoningStartEvent

  **Code location:** `AnthropicAPITranslator.translate_event()` under `content_block_start` handler

  **Code pattern:**
  ```python
  elif block_type == 'thinking':
      block_id = str(uuid.uuid4())
      self._content_blocks[index] = {
          'type': 'thinking',
          'block_id': block_id,
          'buffer': []
      }
      return ReasoningStartEvent(block_id=block_id)
  ```

- [ ] **1.2.3: Handle thinking deltas**
  - [ ] In `translate_event()`, handle `content_block_delta` with type='thinking_delta'
  - [ ] Extract `delta.thinking` field
  - [ ] Append to buffer
  - [ ] Return ReasoningDeltaEvent

  **Code location:** Under `content_block_delta` handler

- [ ] **1.2.4: Handle thinking block end**
  - [ ] In `translate_event()`, handle `content_block_stop` for thinking blocks
  - [ ] Return ReasoningEndEvent with block_id

  **Code location:** Under `content_block_stop` handler

- [ ] **1.2.5: Update imports**
  - [ ] Ensure ReasoningStartEvent, ReasoningDeltaEvent, ReasoningEndEvent imported

- [ ] **1.2.6: Write tests for thinking blocks**
  - [ ] Test thinking_start event
  - [ ] Test thinking_delta event
  - [ ] Test thinking_stop event
  - [ ] Test mixed content (text + thinking + tool)

- [ ] **1.2.7: Manual validation**
  - [ ] Test with Claude model that supports thinking (Claude 3.7 Sonnet+)
  - [ ] Verify reasoning events appear in stream
  - [ ] Test with extended thinking mode enabled

**Acceptance Criteria:**
- ✅ Thinking blocks generate reasoning-start/delta/end events
- ✅ Tests pass for thinking block scenarios
- ✅ Manual testing confirms thinking streams work

---

### Task 1.3: Add SourceEvent for Citations

**Priority:** P0 - CRITICAL
**Complexity:** Medium
**Files Modified:** `vel/events.py`, `vel/providers/translators.py` (Gemini + OpenAI)

#### Subtasks

- [ ] **1.3.1: Add SourceEvent class to vel/events.py**
  - [ ] Add dataclass with `sources` field (list of dicts)
  - [ ] Implement `to_dict()` method
  - [ ] Add to EventType literal
  - [ ] Export in `__all__`

  **Code to add:**
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

- [ ] **1.3.2: Add deduplication to GeminiAPITranslator**
  - [ ] Add `_seen_source_urls: set[str]` to `__init__()`
  - [ ] Clear in `reset()` method

- [ ] **1.3.3: Handle Gemini grounding sources**
  - [ ] In `translate_chunk()`, check for `candidates[].grounding_metadata`
  - [ ] Extract `grounding_sources[]`
  - [ ] For each source with `web` attribute, extract uri, title
  - [ ] Deduplicate by URL
  - [ ] Return SourceEvent if sources found

  **Code location:** `GeminiAPITranslator.translate_chunk()`

  **Code pattern:**
  ```python
  if hasattr(chunk, 'candidates'):
      for candidate in chunk.candidates:
          if hasattr(candidate, 'grounding_metadata'):
              metadata = candidate.grounding_metadata
              if hasattr(metadata, 'grounding_sources'):
                  sources = []
                  for source in metadata.grounding_sources:
                      if hasattr(source, 'web'):
                          web = source.web
                          url = getattr(web, 'uri', '')

                          if url and url not in self._seen_source_urls:
                              self._seen_source_urls.add(url)
                              sources.append({
                                  'type': 'web',
                                  'url': url,
                                  'title': getattr(web, 'title', ''),
                              })

                  if sources:
                      return SourceEvent(sources=sources)
  ```

- [ ] **1.3.4: Handle OpenAI annotations**
  - [ ] In `translate_chunk()`, check for `delta.annotations`
  - [ ] For each annotation with type='url_citation', extract url, title, text
  - [ ] Return SourceEvent if citations found

  **Code location:** `OpenAIAPITranslator.translate_chunk()`

  **Note:** OpenAI annotations are less common, may need special model config

- [ ] **1.3.5: Update imports**
  - [ ] Import SourceEvent in translators.py

- [ ] **1.3.6: Write tests for SourceEvent**
  - [ ] Test Gemini grounding extraction
  - [ ] Test URL deduplication
  - [ ] Test OpenAI annotations (if available)
  - [ ] Test `to_dict()` formatting

- [ ] **1.3.7: Manual validation**
  - [ ] Test Gemini with grounding enabled
  - [ ] Verify sources appear in stream
  - [ ] Verify deduplication works

**Acceptance Criteria:**
- ✅ SourceEvent class exists in vel/events.py
- ✅ Gemini grounding sources are captured
- ✅ URL deduplication works
- ✅ Tests pass
- ✅ Manual testing confirms sources in streams

---

### Task 1.4: Enhance ErrorEvent

**Priority:** P0 - CRITICAL
**Complexity:** Low
**Files Modified:** `vel/events.py`, `vel/providers/translators.py` (Anthropic)

#### Subtasks

- [ ] **1.4.1: Add error_code and error_type to ErrorEvent**
  - [ ] Add optional fields: `error_code`, `error_type`
  - [ ] Update `to_dict()` to include new fields

  **Code modification in vel/events.py:**
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

- [ ] **1.4.2: Extract error details in AnthropicAPITranslator**
  - [ ] In error event handler, extract `error.code` and `error.type`
  - [ ] Pass to ErrorEvent constructor

  **Code location:** `AnthropicAPITranslator.translate_event()` under error handler

  **Code modification:**
  ```python
  elif event_type == 'error':
      error_data = data.get('error', {})
      return ErrorEvent(
          error=error_data.get('message', 'Unknown error'),
          error_code=error_data.get('code'),  # NEW
          error_type=error_data.get('type')   # NEW
      )
  ```

- [ ] **1.4.3: Consider error handling in other translators**
  - [ ] OpenAI errors typically come from HTTP layer (not chunks)
  - [ ] Gemini errors typically come as exceptions
  - [ ] Document this limitation

- [ ] **1.4.4: Write tests for enhanced ErrorEvent**
  - [ ] Test error with all fields
  - [ ] Test error with message only (backward compat)
  - [ ] Test `to_dict()` formatting

- [ ] **1.4.5: Manual validation**
  - [ ] Trigger Anthropic error (rate limit or invalid request)
  - [ ] Verify error details captured

**Acceptance Criteria:**
- ✅ ErrorEvent has error_code and error_type fields
- ✅ Anthropic errors include details
- ✅ Backward compatibility maintained
- ✅ Tests pass

---

## Phase 2: Feature Completeness (P1)

**Goal:** Add advanced features for production use
**Timeline:** Tasks 2.1 → 2.2 → 2.3 (sequential)
**Estimated Effort:** ~3-4 hours

---

### Task 2.1: Add FileEvent

**Priority:** P1 - IMPORTANT
**Complexity:** Medium
**Files Modified:** `vel/events.py`, `vel/providers/translators.py` (Gemini)

#### Subtasks

- [ ] **2.1.1: Add FileEvent class to vel/events.py**
  - [ ] Add dataclass with fields: content, name, mime_type
  - [ ] Implement `to_dict()` method
  - [ ] Add to EventType literal
  - [ ] Export in `__all__`

  **Code to add:**
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

- [ ] **2.1.2: Handle inline_data in GeminiAPITranslator**
  - [ ] In `translate_chunk()`, iterate through parts
  - [ ] Check for `part.inline_data`
  - [ ] Extract data (base64) and mime_type
  - [ ] Return FileEvent

  **Code location:** `GeminiAPITranslator.translate_chunk()`

  **Code pattern:**
  ```python
  if hasattr(chunk, 'parts'):
      for part in chunk.parts:
          if hasattr(part, 'inline_data'):
              inline = part.inline_data
              return FileEvent(
                  content=getattr(inline, 'data', ''),  # base64
                  mime_type=getattr(inline, 'mime_type', '')
              )
  ```

- [ ] **2.1.3: Update imports**
  - [ ] Import FileEvent in translators.py

- [ ] **2.1.4: Write tests for FileEvent**
  - [ ] Test inline_data extraction
  - [ ] Test different mime types (image/png, application/pdf)
  - [ ] Test `to_dict()` formatting

- [ ] **2.1.5: Manual validation**
  - [ ] Test Gemini with image input
  - [ ] Verify file event appears
  - [ ] Check base64 content is correct

**Acceptance Criteria:**
- ✅ FileEvent class exists
- ✅ Gemini inline_data handled
- ✅ Tests pass
- ✅ Manual testing with images works

---

### Task 2.2: Standardize Block ID Generation

**Priority:** P1 - IMPORTANT
**Complexity:** Low
**Files Modified:** `vel/providers/translators.py` (all translators)

#### Subtasks

- [ ] **2.2.1: Document block ID strategy**
  - [ ] Decision: Use indices (like V3) - `'0'`, `'1'`, `'2'`
  - [ ] Rationale: Consistency with Vercel AI SDK
  - [ ] Update CLAUDE.md with decision

- [ ] **2.2.2: Update OpenAIAPITranslator**
  - [ ] Change from UUID to index-based
  - [ ] Add `_next_block_index: int = 0` to `__init__()`
  - [ ] Use `str(self._next_block_index)` instead of `uuid.uuid4()`
  - [ ] Increment after assignment
  - [ ] Reset in `reset()`

- [ ] **2.2.3: Update AnthropicAPITranslator**
  - [ ] Already uses indices from provider
  - [ ] Verify uses `str(index)` not UUID
  - [ ] May already be correct

- [ ] **2.2.4: Update GeminiAPITranslator**
  - [ ] Change from UUID to index-based
  - [ ] Add `_next_block_index: int = 0` to `__init__()`
  - [ ] Use index for text blocks
  - [ ] Reset in `reset()`

- [ ] **2.2.5: Update OpenAIAgentsSDKTranslator**
  - [ ] Change from UUID to index-based
  - [ ] Consistent with other translators

- [ ] **2.2.6: Write tests for block IDs**
  - [ ] Verify IDs are sequential integers as strings
  - [ ] Test reset clears index
  - [ ] Test multiple blocks get '0', '1', '2'

- [ ] **2.2.7: Manual validation**
  - [ ] Check block IDs in streams
  - [ ] Verify consistency across providers

**Acceptance Criteria:**
- ✅ All translators use index-based block IDs
- ✅ IDs are consistent ('0', '1', '2', etc.)
- ✅ Tests verify correct ID generation
- ✅ Documentation updated

---

### Task 2.3: Add Gemini Code Execution Support

**Priority:** P1 - IMPORTANT (but low impact)
**Complexity:** Low
**Files Modified:** `vel/providers/translators.py` (Gemini)

#### Subtasks

- [ ] **2.3.1: Decide on handling approach**
  - [ ] Option A: Skip for now (niche feature)
  - [ ] Option B: Log and document
  - [ ] Option C: Emit as metadata
  - [ ] **Decision:** Option B - Log for visibility, skip emission

- [ ] **2.3.2: Add detection in GeminiAPITranslator**
  - [ ] In `translate_chunk()`, detect `part.executable_code`
  - [ ] In `translate_chunk()`, detect `part.code_execution_result`
  - [ ] Log when encountered (for debugging)
  - [ ] Return None (skip for now)

  **Code pattern:**
  ```python
  if hasattr(chunk, 'parts'):
      for part in chunk.parts:
          # Handle executable code
          if hasattr(part, 'executable_code'):
              # Log for debugging
              import logging
              logging.debug(f"Gemini executable_code detected: {part.executable_code}")
              # Skip for now
              continue

          # Handle code execution result
          if hasattr(part, 'code_execution_result'):
              import logging
              logging.debug(f"Gemini code_execution_result: {part.code_execution_result}")
              # Skip for now
              continue
  ```

- [ ] **2.3.3: Document limitation**
  - [ ] Add to docs/event-translators.md
  - [ ] Note that code execution is detected but not emitted

**Acceptance Criteria:**
- ✅ Code execution parts are detected
- ✅ Logged for debugging
- ✅ Documented as limitation

---

## Phase 3: Documentation & Polish (P2)

**Goal:** Complete documentation and final polish
**Timeline:** Tasks 3.2 → 3.3 (sequential)
**Estimated Effort:** ~2-3 hours

---

### Task 3.2: Review Custom Events

**Priority:** P2 - POLISH
**Complexity:** Low
**Files Modified:** Documentation only

#### Subtasks

- [ ] **3.2.1: Document start-step and finish-step**
  - [ ] Add section to docs/stream-protocol.md
  - [ ] Note these are Vel-specific extensions
  - [ ] Explain use case (custom agent steps)

- [ ] **3.2.2: Evaluate if needed**
  - [ ] Review usage in codebase
  - [ ] Consider if can be deprecated
  - [ ] Decision: Keep or deprecate?

- [ ] **3.2.3: Update CLAUDE.md**
  - [ ] Document Vel-specific extensions
  - [ ] Explain differences from V3

**Acceptance Criteria:**
- ✅ Custom events documented
- ✅ Decision made on future

---

### Task 3.3: Update Documentation

**Priority:** P2 - POLISH
**Complexity:** Medium
**Files Modified:** All docs

#### Subtasks

- [ ] **3.3.1: Update docs/stream-protocol.md**
  - [ ] Add ResponseMetadataEvent section
  - [ ] Add SourceEvent section
  - [ ] Add FileEvent section
  - [ ] Update event type table
  - [ ] Add examples for each new event

- [ ] **3.3.2: Update docs/event-translators.md**
  - [ ] Add ResponseMetadataEvent to translator capabilities
  - [ ] Document thinking block support
  - [ ] Document source/citation support
  - [ ] Document file support
  - [ ] Update examples

- [ ] **3.3.3: Update README.md**
  - [ ] Add ResponseMetadataEvent to features
  - [ ] Add thinking block support to features
  - [ ] Add source/citation support to features
  - [ ] Update parity status

- [ ] **3.3.4: Update CLAUDE.md**
  - [ ] Document new event types
  - [ ] Update architecture section
  - [ ] Add block ID generation strategy
  - [ ] Note Vel-specific extensions

- [ ] **3.3.5: Create PARITY_STATUS.md**
  - [ ] Document current parity percentage
  - [ ] List all implemented events
  - [ ] List any remaining gaps
  - [ ] Note skipped features (safety ratings, event renaming)

**Acceptance Criteria:**
- ✅ All docs updated
- ✅ Examples provided
- ✅ Parity status documented

---

## Testing Strategy

### Unit Tests

**File:** `tests/test_translators.py` (create if doesn't exist)

#### Test Structure

```python
import pytest
from vel.providers.translators import (
    OpenAIAPITranslator,
    AnthropicAPITranslator,
    GeminiAPITranslator,
    OpenAIAgentsSDKTranslator
)
from vel.events import *

class TestOpenAIAPITranslator:
    def setup_method(self):
        self.translator = OpenAIAPITranslator()

    # Existing functionality tests
    def test_text_streaming(self): pass
    def test_tool_call_streaming(self): pass

    # NEW: Metadata tests
    def test_metadata_extraction(self):
        chunk = {
            'usage': {
                'prompt_tokens': 10,
                'completion_tokens': 20,
                'total_tokens': 30
            },
            'model': 'gpt-4o'
        }
        event = self.translator.translate_chunk(chunk)
        assert isinstance(event, ResponseMetadataEvent)
        assert event.usage['promptTokens'] == 10
        assert event.usage['completionTokens'] == 20
        assert event.model_id == 'gpt-4o'

    # NEW: Source tests
    def test_annotation_extraction(self):
        chunk = {
            'choices': [{
                'delta': {
                    'annotations': [
                        {
                            'type': 'url_citation',
                            'url': 'https://example.com',
                            'title': 'Example',
                            'text': 'snippet'
                        }
                    ]
                }
            }]
        }
        event = self.translator.translate_chunk(chunk)
        assert isinstance(event, SourceEvent)
        assert len(event.sources) == 1
        assert event.sources[0]['url'] == 'https://example.com'

    # NEW: Block ID tests
    def test_block_id_sequential(self):
        chunk1 = {'choices': [{'delta': {'content': 'Hello'}}]}
        chunk2 = {'choices': [{'delta': {'content': ' world'}}]}

        event1 = self.translator.translate_chunk(chunk1)
        assert event1.block_id == '0'

        event2 = self.translator.translate_chunk(chunk2)
        assert event2.block_id == '0'  # Same block


class TestAnthropicAPITranslator:
    def setup_method(self):
        self.translator = AnthropicAPITranslator()

    # Existing tests
    def test_text_streaming(self): pass
    def test_tool_streaming(self): pass

    # NEW: Thinking block tests
    def test_thinking_block_start(self):
        data = {
            'type': 'content_block_start',
            'index': 0,
            'content_block': {
                'type': 'thinking',
                'id': 'thinking_123'
            }
        }
        event = self.translator.translate_event(data)
        assert isinstance(event, ReasoningStartEvent)
        assert event.block_id is not None

    def test_thinking_block_delta(self):
        # First start the block
        start_data = {
            'type': 'content_block_start',
            'index': 0,
            'content_block': {'type': 'thinking'}
        }
        self.translator.translate_event(start_data)

        # Then send delta
        delta_data = {
            'type': 'content_block_delta',
            'index': 0,
            'delta': {
                'type': 'thinking_delta',
                'thinking': 'Let me think...'
            }
        }
        event = self.translator.translate_event(delta_data)
        assert isinstance(event, ReasoningDeltaEvent)
        assert event.delta == 'Let me think...'

    def test_thinking_block_stop(self):
        # Setup block
        start_data = {
            'type': 'content_block_start',
            'index': 0,
            'content_block': {'type': 'thinking'}
        }
        start_event = self.translator.translate_event(start_data)

        # Stop block
        stop_data = {
            'type': 'content_block_stop',
            'index': 0
        }
        event = self.translator.translate_event(stop_data)
        assert isinstance(event, ReasoningEndEvent)
        assert event.block_id == start_event.block_id

    # NEW: Error detail tests
    def test_error_with_details(self):
        data = {
            'type': 'error',
            'error': {
                'message': 'Rate limit exceeded',
                'code': 'rate_limit_error',
                'type': 'invalid_request_error'
            }
        }
        event = self.translator.translate_event(data)
        assert isinstance(event, ErrorEvent)
        assert event.error == 'Rate limit exceeded'
        assert event.error_code == 'rate_limit_error'
        assert event.error_type == 'invalid_request_error'


class TestGeminiAPITranslator:
    def setup_method(self):
        self.translator = GeminiAPITranslator()

    # Existing tests
    def test_text_streaming(self): pass

    # NEW: Source tests
    def test_grounding_sources(self):
        # Mock chunk with grounding metadata
        # (Would need to create mock object with proper attributes)
        pass

    # NEW: File tests
    def test_inline_data(self):
        # Mock chunk with inline_data
        pass

    # NEW: Block ID tests
    def test_block_id_sequential(self):
        pass


class TestOpenAIAgentsSDKTranslator:
    # Existing tests
    pass


# Integration tests (if API keys available)
@pytest.mark.integration
class TestRealProviders:
    @pytest.mark.skipif(not os.getenv('OPENAI_API_KEY'), reason="No API key")
    async def test_openai_metadata_real(self):
        # Make real API call, verify metadata
        pass

    @pytest.mark.skipif(not os.getenv('ANTHROPIC_API_KEY'), reason="No API key")
    async def test_anthropic_thinking_real(self):
        # Make real API call with thinking, verify events
        pass
```

#### Test Checklist

- [ ] **OpenAI Translator Tests**
  - [ ] test_metadata_extraction
  - [ ] test_metadata_to_dict
  - [ ] test_annotation_extraction
  - [ ] test_source_to_dict
  - [ ] test_block_id_sequential
  - [ ] test_reset_clears_block_index

- [ ] **Anthropic Translator Tests**
  - [ ] test_thinking_block_start
  - [ ] test_thinking_block_delta
  - [ ] test_thinking_block_stop
  - [ ] test_mixed_content_text_and_thinking
  - [ ] test_error_with_details
  - [ ] test_error_backward_compat

- [ ] **Gemini Translator Tests**
  - [ ] test_grounding_source_extraction
  - [ ] test_source_deduplication
  - [ ] test_inline_data_extraction
  - [ ] test_file_to_dict
  - [ ] test_block_id_sequential

- [ ] **Event Tests**
  - [ ] test_response_metadata_to_dict
  - [ ] test_source_event_to_dict
  - [ ] test_file_event_to_dict
  - [ ] test_error_event_to_dict_with_details

---

### Integration Tests

**Goal:** Verify behavior with real API calls

#### Setup

```bash
# Ensure API keys are set
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
export GOOGLE_API_KEY="..."
```

#### Manual Test Script

Create `examples/test_parity_features.py`:

```python
import asyncio
from vel import Agent

async def test_openai_metadata():
    """Test OpenAI metadata extraction"""
    agent = Agent(
        id='test',
        model={'provider': 'openai', 'model': 'gpt-4o'},
    )

    events = []
    async for event in agent.run_stream({'message': 'Say hello'}):
        events.append(event)
        print(f"{event.type}: {event.to_dict()}")

    # Verify metadata event exists
    metadata_events = [e for e in events if e.type == 'response-metadata']
    assert len(metadata_events) > 0
    print(f"✅ OpenAI metadata: {metadata_events[0].to_dict()}")

async def test_anthropic_thinking():
    """Test Anthropic thinking blocks"""
    agent = Agent(
        id='test',
        model={'provider': 'anthropic', 'model': 'claude-sonnet-4-20250514'},
    )

    # May need extended_thinking parameter
    events = []
    async for event in agent.run_stream({'message': 'Solve a complex problem'}):
        events.append(event)
        if event.type.startswith('reasoning-'):
            print(f"✅ Reasoning event: {event.type}")

    reasoning_events = [e for e in events if 'reasoning' in e.type]
    print(f"Found {len(reasoning_events)} reasoning events")

async def test_gemini_grounding():
    """Test Gemini grounding sources"""
    agent = Agent(
        id='test',
        model={'provider': 'google', 'model': 'gemini-1.5-pro'},
    )

    # May need grounding config
    events = []
    async for event in agent.run_stream({'message': 'What is the latest news?'}):
        events.append(event)
        if event.type == 'source':
            print(f"✅ Source event: {event.to_dict()}")

    source_events = [e for e in events if e.type == 'source']
    print(f"Found {len(source_events)} source events")

if __name__ == '__main__':
    asyncio.run(test_openai_metadata())
    asyncio.run(test_anthropic_thinking())
    asyncio.run(test_gemini_grounding())
```

#### Integration Test Checklist

- [ ] **OpenAI Integration**
  - [ ] Run with real API
  - [ ] Verify metadata appears
  - [ ] Check usage counts are reasonable

- [ ] **Anthropic Integration**
  - [ ] Run with Claude 3.7 Sonnet+
  - [ ] Verify thinking blocks (if model supports)
  - [ ] Check error details on failure

- [ ] **Gemini Integration**
  - [ ] Run with grounding enabled
  - [ ] Verify sources appear
  - [ ] Test with inline data (image)

---

## Validation Checklist

### Pre-Merge Validation

Before merging to main:

- [ ] **Code Quality**
  - [ ] All code follows existing style
  - [ ] No unnecessary dependencies added
  - [ ] Type hints added where appropriate
  - [ ] Docstrings updated

- [ ] **Testing**
  - [ ] All unit tests pass: `pytest tests/test_translators.py -v`
  - [ ] All existing tests still pass: `pytest -v`
  - [ ] Integration tests run successfully
  - [ ] Code coverage acceptable

- [ ] **Functionality**
  - [ ] ResponseMetadataEvent works for all providers
  - [ ] Thinking blocks work for Anthropic
  - [ ] Sources work for Gemini (+ OpenAI if available)
  - [ ] File events work for Gemini
  - [ ] Error details captured
  - [ ] Block IDs are sequential

- [ ] **Documentation**
  - [ ] All docs updated
  - [ ] Examples provided
  - [ ] PARITY_STATUS.md created
  - [ ] CLAUDE.md updated

- [ ] **Backward Compatibility**
  - [ ] Existing agents still work
  - [ ] No breaking changes to public API
  - [ ] Event serialization unchanged for old events

- [ ] **Git Hygiene**
  - [ ] Commits are atomic and well-described
  - [ ] Branch is rebased on latest main
  - [ ] No merge conflicts

---

## Rollback Plan

### If Things Go Wrong

**Immediate Rollback:**
```bash
# Discard all changes
git reset --hard HEAD

# Or revert to specific commit
git log --oneline  # Find last good commit
git reset --hard <commit-hash>
```

**Partial Rollback:**
```bash
# Revert specific file
git checkout HEAD -- vel/events.py

# Revert specific commit
git revert <commit-hash>
```

**Recovery Strategy:**
1. Document what went wrong
2. Create bug report with reproduction steps
3. Fix in separate branch
4. Re-test before retry

---

## Success Metrics

### Final Parity Score

**Target:** 95%+ parity (24/25 events)

**Expected Coverage:**

| Event Category | Before | After | Target |
|---------------|--------|-------|--------|
| Text events | 100% | 100% | ✅ |
| Reasoning events | 67% | 100% | ✅ |
| Tool events | 83% | 83% | ✅ |
| Metadata events | 0% | 100% | ✅ |
| Source events | 0% | 100% | ✅ |
| File events | 0% | 100% | ✅ |
| Error events | 50% | 100% | ✅ |

**Skipped (as requested):**
- Safety ratings (Phase 2, Task 2.4)
- Event naming alignment (Phase 3, Task 3.1)

**Final Expected Parity:** ~96% (24/25 events)
- Missing: Safety ratings only

---

## Timeline Summary

| Phase | Tasks | Estimated Time | Cumulative |
|-------|-------|----------------|------------|
| Pre-Implementation | Setup, review | 1 hour | 1 hour |
| Phase 1 (P0) | Tasks 1.1-1.4 | 4-6 hours | 5-7 hours |
| Phase 2 (P1) | Tasks 2.1-2.3 | 3-4 hours | 8-11 hours |
| Phase 3 (P2) | Tasks 3.2-3.3 | 2-3 hours | 10-14 hours |
| Testing & Validation | Integration, docs | 2-3 hours | 12-17 hours |

**Total Estimated Time:** 12-17 hours
**Recommended Approach:** Work in 3-4 sessions over 2-3 days

---

## Final Checklist

### Before Starting
- [ ] Read this entire plan
- [ ] Understand all tasks
- [ ] Have API keys ready
- [ ] Create feature branch
- [ ] Establish baseline (run tests)

### During Implementation
- [ ] Use TodoWrite to track progress
- [ ] Test after each task
- [ ] Commit after each phase
- [ ] Document any issues immediately

### After Implementation
- [ ] Run full test suite
- [ ] Run integration tests
- [ ] Update all documentation
- [ ] Create PARITY_STATUS.md
- [ ] Review git commits
- [ ] Merge to main

---

**Document Status:** Ready for implementation
**Approval:** Required before starting
**Owner:** TBD
**Last Updated:** 2025-10-14

---

## Notes

- This plan skips safety ratings (Task 2.4) and event naming (Task 3.1) as requested
- Focus is on critical parity features that add real value
- All changes maintain backward compatibility
- Testing strategy ensures quality
- Documentation ensures discoverability

**Ready to begin? Confirm approval and we'll start with Phase 1, Task 1.1.**
