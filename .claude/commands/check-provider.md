---
description: "Validate a provider implementation"
argument-hint: "[provider-name]"
---

Validate that a Vel provider implementation is correct.

## Instructions

1. **Check file exists:**
   ```bash
   ls vel/providers/$1.py
   ```

2. **Verify BaseProvider implementation:**

   Check that the provider class:
   - Extends `BaseProvider`
   - Implements `async def stream()` method
   - Implements `async def generate()` method
   - Supports optional `api_key` parameter
   - Has proper error handling

3. **Verify stream translator:**

   Check `vel/providers/translators.py` for:
   - Translator class named `${1^}StreamTranslator`
   - `translate()` method returning `List[StreamEvent]`
   - Proper event type mapping:
     - Text chunks -> `text-delta`
     - Tool calls -> `tool-input-start`, `tool-input-available`
     - Finish -> `finish-message`
     - Errors -> `error`

4. **Check message translation:**

   Verify `vel/providers/message_translator.py` handles:
   - Role mapping for provider format
   - Tool result formatting
   - System message handling
   - Thinking/reasoning blocks (if supported)

5. **Verify registration:**

   Check `vel/providers/__init__.py`:
   - Provider is registered with soft-loading
   - Uses try/except for graceful fallback

6. **Check tests exist:**
   ```bash
   ls tests/test_providers/test_$1.py
   ```

7. **Run provider tests:**
   ```bash
   pytest tests/test_providers/test_$1.py -v
   ```

## Checklist Output

Report findings as:

| Check | Status | Notes |
|-------|--------|-------|
| File exists | ✓/✗ | |
| BaseProvider interface | ✓/✗ | |
| Stream translator | ✓/✗ | |
| Message translation | ✓/✗ | |
| Registration | ✓/✗ | |
| Tests exist | ✓/✗ | |
| Tests pass | ✓/✗ | |

## Reference

- ADR-003 for architecture requirements
- `.claude/rules/provider-development.md` for guidelines
