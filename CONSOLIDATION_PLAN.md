# Vel Translation Consolidation Plan

## Overview

Consolidate all event translation logic into a single, reusable module within the `providers/` directory. This eliminates duplication between provider implementations and creates a clean abstraction for external projects to import.

## Goals

1. **Single Source of Truth** - All translation logic in `vel/providers/translators.py`
2. **Eliminate Duplication** - Providers use translators instead of inline translation
3. **External Reusability** - Projects like Mesh can import translators directly
4. **Clean Separation** - Providers handle API calls, translators handle event formatting
5. **Easy Extension** - Clear pattern for adding new providers

---

## Current State

### File Structure
```
vel/
├── providers/
│   ├── __init__.py          # ProviderRegistry
│   ├── base.py              # BaseProvider interface
│   ├── openai.py            # OpenAI API calls + inline translation ❌
│   ├── anthropic.py         # Anthropic API calls + inline translation ❌
│   └── google.py            # Gemini API calls + inline translation ❌
├── sdk_translators.py       # OpenAI Agents SDK translator ✅
└── __init__.py              # Exports
```

### Problems

1. **Translation logic duplicated** - Each provider has its own translation code
2. **Not importable** - External projects can't reuse provider translation logic
3. **Inconsistent location** - SDK translator separate from providers
4. **Hard to maintain** - Update same logic in 3 places
5. **Hard to test** - Translation logic mixed with API calls

---

## Target Architecture

### New File Structure
```
vel/
├── providers/
│   ├── __init__.py          # ProviderRegistry + export translators
│   ├── base.py              # BaseProvider interface
│   ├── translators.py       # ALL translation logic ✅ NEW
│   ├── openai.py            # OpenAI API calls only (uses translator)
│   ├── anthropic.py         # Anthropic API calls only (uses translator)
│   └── google.py            # Gemini API calls only (uses translator)
└── __init__.py              # Export translators for external use
```

### Clean Separation

```
┌──────────────────────────────────────────────┐
│  vel/providers/translators.py               │
│  (Single source of truth for translation)   │
├──────────────────────────────────────────────┤
│  - OpenAIAPITranslator                       │
│  - OpenAIAgentsSDKTranslator                 │
│  - AnthropicAPITranslator                    │
│  - GeminiAPITranslator                       │
└──────────────────────────────────────────────┘
          ↑                           ↑
          │                           │
     Used by                      Imported by
          │                           │
┌─────────────────────┐   ┌──────────────────────┐
│  vel/providers/     │   │  External Projects   │
│  - openai.py        │   │  - Mesh              │
│  - anthropic.py     │   │  - Custom agents     │
│  - google.py        │   │                      │
└─────────────────────┘   └──────────────────────┘
```

---

## Implementation Phases

### Phase 1: Extract & Centralize Translators ✅
**Goal:** Create unified translators module without breaking existing code

#### Tasks

1. **Create `vel/providers/translators.py`**
   - Move `OpenAIAgentsSDKTranslator` from `vel/sdk_translators.py`
   - Extract `OpenAIAPITranslator` from `vel/providers/openai.py`
   - Extract `AnthropicAPITranslator` from `vel/providers/anthropic.py`
   - Extract `GeminiAPITranslator` from `vel/providers/google.py`

2. **Translator Interface Pattern**
   ```python
   class BaseTranslator:
       """Base class for event translators (optional, for consistency)"""
       def translate(self, native_event: Any) -> Optional[StreamEvent]:
           raise NotImplementedError

       def reset(self):
           """Reset internal state between messages"""
           pass
   ```

3. **Implement Each Translator**
   - `OpenAIAPITranslator` - Translates `ChatCompletionChunk`
   - `OpenAIAgentsSDKTranslator` - Translates OpenAI Agents SDK events (already exists)
   - `AnthropicAPITranslator` - Translates `MessageStreamEvent`
   - `GeminiAPITranslator` - Translates Gemini streaming response

4. **Convenience Functions**
   ```python
   def get_openai_api_translator() -> OpenAIAPITranslator:
       return OpenAIAPITranslator()

   def get_openai_agents_translator() -> OpenAIAgentsSDKTranslator:
       return OpenAIAgentsSDKTranslator()

   def get_anthropic_translator() -> AnthropicAPITranslator:
       return AnthropicAPITranslator()

   def get_gemini_translator() -> GeminiAPITranslator:
       return GeminiAPITranslator()
   ```

**Files Created:**
- `vel/providers/translators.py` (NEW)

**Files Modified:**
- None yet (parallel implementation)

**Testing:**
- Unit tests for each translator
- Test with sample events from each provider
- Verify translation output matches current provider output

---

### Phase 2: Refactor Providers to Use Translators 🔄
**Goal:** Update providers to delegate translation, remove inline logic

#### Tasks

1. **Update `vel/providers/openai.py`**
   ```python
   from .translators import OpenAIAPITranslator

   class OpenAIProvider(BaseProvider):
       def __init__(self):
           self.translator = OpenAIAPITranslator()
           # ... existing init code

       async def stream(self, messages, model, tools):
           # 1. Make API call (Provider's job)
           response = await client.chat.completions.create(...)

           # 2. Translate events (Translator's job)
           async for chunk in response:
               vel_event = self.translator.translate(chunk)
               if vel_event:
                   yield vel_event
   ```

2. **Update `vel/providers/anthropic.py`**
   - Similar pattern using `AnthropicAPITranslator`
   - Remove inline translation logic

3. **Update `vel/providers/google.py`**
   - Similar pattern using `GeminiAPITranslator`
   - Remove inline translation logic

4. **Update `generate()` methods**
   - Non-streaming methods should also use translators
   - Or keep simple inline translation (less critical since not streamed)

**Files Modified:**
- `vel/providers/openai.py`
- `vel/providers/anthropic.py`
- `vel/providers/google.py`

**Testing:**
- Integration tests with actual API calls
- Verify streaming works end-to-end
- Run existing test suite to ensure no regressions

---

### Phase 3: Update Exports & Delete Old File 🗑️
**Goal:** Export translators for external use, clean up old files

#### Tasks

1. **Update `vel/providers/__init__.py`**
   ```python
   from .base import BaseProvider
   from .openai import OpenAIProvider
   from .google import GeminiProvider
   from .anthropic import AnthropicProvider
   from .registry import ProviderRegistry

   # Export translators
   from .translators import (
       OpenAIAPITranslator,
       OpenAIAgentsSDKTranslator,
       AnthropicAPITranslator,
       GeminiAPITranslator,
       get_openai_api_translator,
       get_openai_agents_translator,
       get_anthropic_translator,
       get_gemini_translator,
   )

   __all__ = [
       'BaseProvider',
       'OpenAIProvider',
       'GeminiProvider',
       'AnthropicProvider',
       'ProviderRegistry',
       # Translators
       'OpenAIAPITranslator',
       'OpenAIAgentsSDKTranslator',
       'AnthropicAPITranslator',
       'GeminiAPITranslator',
       'get_openai_api_translator',
       'get_openai_agents_translator',
       'get_anthropic_translator',
       'get_gemini_translator',
   ]
   ```

2. **Update `vel/__init__.py`**
   ```python
   # Import from providers
   from .providers import (
       OpenAIAPITranslator,
       OpenAIAgentsSDKTranslator,
       AnthropicAPITranslator,
       GeminiAPITranslator,
       get_openai_api_translator,
       get_openai_agents_translator,
       get_anthropic_translator,
       get_gemini_translator,
   )

   __all__ = [
       # ... existing exports
       # Event Translators
       'OpenAIAPITranslator',
       'OpenAIAgentsSDKTranslator',
       'AnthropicAPITranslator',
       'GeminiAPITranslator',
       'get_openai_api_translator',
       'get_openai_agents_translator',
       'get_anthropic_translator',
       'get_gemini_translator',
   ]
   ```

3. **Delete old file**
   ```bash
   rm vel/sdk_translators.py
   ```

4. **Update imports in any internal code**
   - Search for `from vel.sdk_translators import`
   - Replace with `from vel.providers.translators import`
   - Or use top-level import: `from vel import`

**Files Modified:**
- `vel/providers/__init__.py`
- `vel/__init__.py`

**Files Deleted:**
- `vel/sdk_translators.py`

**Testing:**
- Verify imports work: `from vel import get_openai_api_translator`
- Verify backwards compatibility if needed
- Check all examples still run

---

### Phase 4: Documentation & Examples 📚
**Goal:** Update all documentation to reflect new architecture

#### Tasks

1. **Update `docs/sdk-translators.md`**
   - Rename to `docs/event-translators.md`
   - Update title: "Event Translators" (not "SDK Event Translators")
   - Add sections for all translators:
     - OpenAI Chat API
     - OpenAI Agents SDK
     - Anthropic API
     - Google Gemini API
   - Update import examples
   - Show usage for each translator

2. **Update `docs/providers.md`**
   - Add section explaining provider architecture
   - Document that providers use translators internally
   - Explain separation: providers = API calls, translators = events

3. **Update `README.md`**
   - Update link from "SDK Event Translators" to "Event Translators"
   - Update description to mention all providers

4. **Create example file**
   - `examples/event_translators_example.py`
   - Show usage with each provider
   - Demonstrate external integration (Mesh use case)

5. **Update `CLAUDE.md`**
   - Document new architecture
   - Update file paths in structure diagram
   - Add section on translators

6. **Update API reference**
   - Document all translator classes
   - Document convenience functions
   - Show event mapping tables

**Files Modified:**
- `docs/sdk-translators.md` → `docs/event-translators.md`
- `docs/providers.md`
- `README.md`
- `CLAUDE.md`
- `docs/api-reference.md`

**Files Created:**
- `examples/event_translators_example.py`

---

### Phase 5: Testing & Validation ✅
**Goal:** Ensure everything works end-to-end

#### Tasks

1. **Unit Tests**
   - Test each translator independently
   - Test with sample events
   - Test edge cases (empty events, errors, etc.)

2. **Integration Tests**
   - Test providers with real API calls
   - Verify streaming works correctly
   - Test tool calling
   - Test error handling

3. **External Usage Test**
   - Create minimal Mesh-like example
   - Import translators from vel
   - Verify external usage works

4. **Regression Tests**
   - Run full test suite
   - Verify all existing functionality works
   - Check examples run correctly

5. **Manual Testing**
   ```bash
   # Test imports
   python -c "from vel import get_openai_api_translator; print('✓')"
   python -c "from vel import get_openai_agents_translator; print('✓')"
   python -c "from vel import get_anthropic_translator; print('✓')"
   python -c "from vel import get_gemini_translator; print('✓')"

   # Test provider usage
   python examples/quickstart.py
   python examples/event_translators_example.py
   ```

**Test Files:**
- `tests/test_translators.py` (NEW)
- `tests/test_providers.py` (UPDATE)
- `tests/test_integration.py` (NEW)

---

## Detailed Implementation Guide

### Translator Extraction Details

#### OpenAI API Translator

**Extract from:** `vel/providers/openai.py` (lines ~50-120)

**Current inline logic:**
```python
async for chunk in response:
    if chunk.choices and len(chunk.choices) > 0:
        choice = chunk.choices[0]

        if choice.delta.content:
            yield TextDeltaEvent(
                block_id=self._current_block_id,
                delta=choice.delta.content
            )

        if choice.delta.tool_calls:
            # ... tool call logic

        if choice.finish_reason:
            # ... finish logic
```

**New translator class:**
```python
class OpenAIAPITranslator:
    """Translates OpenAI Chat Completions API events to Vel format."""

    def __init__(self):
        self._text_block_id: Optional[str] = None
        self._tool_calls: Dict[str, Dict] = {}

    def translate(self, chunk: ChatCompletionChunk) -> Optional[StreamEvent]:
        """
        Translate ChatCompletionChunk to Vel StreamEvent.

        Args:
            chunk: Native ChatCompletionChunk from OpenAI

        Returns:
            StreamEvent or None if event should be skipped
        """
        if not chunk.choices or len(chunk.choices) == 0:
            return None

        choice = chunk.choices[0]

        # Text content
        if choice.delta.content:
            if self._text_block_id is None:
                self._text_block_id = str(uuid.uuid4())

            return TextDeltaEvent(
                block_id=self._text_block_id,
                delta=choice.delta.content
            )

        # Tool calls
        if choice.delta.tool_calls:
            # ... extract tool call logic
            pass

        # Finish
        if choice.finish_reason:
            # ... extract finish logic
            pass

        return None

    def reset(self):
        """Reset state between messages."""
        self._text_block_id = None
        self._tool_calls.clear()
```

#### Anthropic API Translator

**Extract from:** `vel/providers/anthropic.py` (lines ~40-100)

**Pattern:** Similar to OpenAI, extract MessageStreamEvent translation

#### Gemini API Translator

**Extract from:** `vel/providers/google.py` (lines ~50-110)

**Pattern:** Similar to others, extract Gemini streaming response translation

---

## Breaking Changes

### None Expected ✅

This refactor is **internal only**:
- Public API remains the same
- `Agent` usage unchanged
- Provider usage unchanged
- Only adds new exports (translators)

### Deprecation Path (if needed)

If we decide to deprecate old imports:

```python
# vel/sdk_translators.py (mark as deprecated, keep for v1.x)
import warnings
from .providers.translators import *

warnings.warn(
    "vel.sdk_translators is deprecated. Import from vel.providers.translators or vel directly.",
    DeprecationWarning,
    stacklevel=2
)
```

**Removal:** v2.0

---

## Success Criteria

### Must Have ✅
1. All translation logic in `vel/providers/translators.py`
2. Providers use translators (no inline translation)
3. Translators importable: `from vel import get_openai_api_translator`
4. All existing tests pass
5. Documentation updated

### Nice to Have 🎯
1. `BaseTranslator` interface for consistency
2. Comprehensive translator unit tests
3. Example for external integration
4. Performance benchmarks (no regression)

---

## Timeline Estimate

### Phase 1: Extract Translators
- **Effort:** 4-6 hours
- **Tasks:** Create translators.py, extract logic, write tests

### Phase 2: Refactor Providers
- **Effort:** 2-3 hours
- **Tasks:** Update 3 providers, test integration

### Phase 3: Update Exports
- **Effort:** 1 hour
- **Tasks:** Update __init__.py files, delete old file

### Phase 4: Documentation
- **Effort:** 2-3 hours
- **Tasks:** Update docs, create examples

### Phase 5: Testing
- **Effort:** 2-3 hours
- **Tasks:** Run tests, manual verification

**Total:** ~12-16 hours

---

## Risk Assessment

### Low Risk 🟢
- Internal refactor only
- No public API changes
- Can implement incrementally
- Easy to rollback (keep old code during transition)

### Potential Issues & Mitigation

| Risk | Mitigation |
|------|------------|
| Translation logic missed during extraction | Thorough code review, compare outputs |
| Provider behavior changes | Comprehensive integration tests |
| Performance regression | Benchmark before/after |
| Import errors after move | Update all imports, test thoroughly |

---

## Post-Implementation

### Monitoring
- Watch for issues in production
- Monitor performance metrics
- Collect user feedback

### Future Enhancements
1. Add more translators (Cohere, Mistral, etc.)
2. Shared translation utilities
3. Event validation/schema checking
4. Translation middleware system

---

## Questions to Resolve

1. **Should we create a `BaseTranslator` abstract class?**
   - Pro: Consistency, clear interface
   - Con: More boilerplate, not strictly needed

2. **Should we keep backwards compatibility for `vel.sdk_translators`?**
   - Pro: No breaking changes
   - Con: Maintains old module

3. **Should translators be stateful or stateless?**
   - Current: Stateful (track block IDs, tool calls)
   - Alternative: Pass state explicitly

4. **Should we add type hints for native event types?**
   - Pro: Better IDE support
   - Con: Requires importing provider SDKs as dependencies

---

## Approval Checklist

Before starting implementation:
- [ ] Architecture reviewed and approved
- [ ] File structure confirmed
- [ ] Breaking changes acceptable (none expected)
- [ ] Timeline realistic
- [ ] Success criteria clear

---

**Created:** 2025-10-14
**Status:** Planning
**Next Step:** Review and approve, then begin Phase 1
