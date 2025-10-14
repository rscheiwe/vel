# Vercel AI SDK Parity Validation Plan

## Objective

Ensure Vel's event translators have complete parity with Vercel AI SDK's TypeScript provider implementations for:
- OpenAI (Chat Completions API)
- Google (Gemini API)
- Anthropic (Messages API)

## Scope

### Target Repository
https://github.com/vercel/ai/tree/main/packages

**Provider Packages to Examine:**
1. `packages/openai/` - OpenAI provider implementation
2. `packages/google/` - Google Gemini provider implementation
3. `packages/anthropic/` - Anthropic Claude provider implementation

### What We're Validating

For each provider, we need to verify:

1. **Event Types Coverage**
   - All native provider events are handled
   - All stream protocol events are emitted correctly
   - Event sequencing matches Vercel AI SDK

2. **Event Mapping Accuracy**
   - Native events → Stream protocol events mapping is identical
   - Event properties/fields are correctly mapped
   - Edge cases are handled (errors, partial responses, etc.)

3. **Stateful Translation Logic**
   - Block ID generation/tracking matches
   - Tool call accumulation matches
   - Message finalization matches

4. **Missing Features**
   - Features in Vercel AI SDK but not in Vel
   - Different approaches that need reconciliation

## Methodology

### Phase 1: Repository Reconnaissance
**Objective:** Locate exact files in Vercel AI SDK repo

For each provider:
- Find main provider file (e.g., `openai-chat-language-model.ts`)
- Find streaming logic (e.g., `openai-chat-stream.ts`)
- Find event conversion logic
- Identify utility functions and types

### Phase 2: Deep Code Analysis
**Objective:** Extract complete event handling logic

For each provider file:
- Document all native event types handled
- Document all stream protocol events emitted
- Extract event mapping logic (native → stream)
- Identify state tracking mechanisms
- Note error handling patterns

### Phase 3: Comparison Matrix
**Objective:** Side-by-side comparison of Vel vs Vercel AI SDK

Create comparison tables:

```
| Native Event | Vercel AI SDK Action | Vel Translator Action | Status | Notes |
|--------------|---------------------|----------------------|--------|-------|
| ... | ... | ... | ✅/❌/⚠️ | ... |
```

Status key:
- ✅ Parity achieved
- ❌ Missing in Vel
- ⚠️ Different implementation (needs review)

### Phase 4: Gap Analysis
**Objective:** Identify all discrepancies

Document:
- Missing event handlers in Vel
- Different event mappings
- Missing properties/fields
- Different state management
- Missing error handling

### Phase 5: Remediation Plan
**Objective:** Plan to achieve 100% parity

For each gap:
- Priority (P0: Critical, P1: Important, P2: Nice-to-have)
- Action required (add handler, modify mapping, etc.)
- Estimated complexity
- Implementation notes

### Phase 6: Implementation & Validation
**Objective:** Execute remediation plan

- Implement changes to translators
- Update tests
- Validate against real API responses
- Update documentation

## Execution Plan

### Step 1: OpenAI Provider Analysis

**Files to examine:**
- `packages/openai/src/openai-chat-language-model.ts`
- `packages/openai/src/openai-chat-stream.ts`
- `packages/openai/src/convert-to-openai-chat-messages.ts`
- `packages/openai/src/map-openai-chat-logprobs.ts`
- `packages/openai/src/openai-error.ts`

**Focus areas:**
- SSE chunk parsing (`data: [DONE]`, JSON chunks)
- Delta handling (text, tool_calls)
- Tool call accumulation (index-based)
- Finish reason mapping
- Error handling

**Vel comparison target:**
- `vel/providers/translators.py::OpenAIAPITranslator`

### Step 2: Google (Gemini) Provider Analysis

**Files to examine:**
- `packages/google/src/google-generative-ai-language-model.ts`
- `packages/google/src/google-generative-ai-stream.ts`
- `packages/google/src/convert-to-google-generative-ai-messages.ts`

**Focus areas:**
- Chunk structure (candidates, parts)
- Text part handling
- Function call handling
- Finish reason mapping
- Safety ratings handling
- Error handling

**Vel comparison target:**
- `vel/providers/translators.py::GeminiAPITranslator`

### Step 3: Anthropic Provider Analysis

**Files to examine:**
- `packages/anthropic/src/anthropic-messages-language-model.ts`
- `packages/anthropic/src/anthropic-messages-stream.ts`
- `packages/anthropic/src/convert-to-anthropic-messages-prompt.ts`

**Focus areas:**
- SSE event types (message_start, content_block_start, content_block_delta, content_block_stop, message_delta, message_stop)
- Content block types (text, tool_use)
- Delta types (text_delta, input_json_delta)
- Stop reason mapping
- Usage tracking
- Error handling

**Vel comparison target:**
- `vel/providers/translators.py::AnthropicAPITranslator`

## Deliverables

1. **OPENAI_PARITY_ANALYSIS.md** - OpenAI provider comparison
2. **GEMINI_PARITY_ANALYSIS.md** - Google Gemini provider comparison
3. **ANTHROPIC_PARITY_ANALYSIS.md** - Anthropic provider comparison
4. **PARITY_GAPS_SUMMARY.md** - Consolidated gap analysis
5. **REMEDIATION_PLAN.md** - Implementation plan to achieve parity
6. Updated translators (if needed)
7. Updated tests (if needed)
8. Updated documentation (if needed)

## Success Criteria

- ✅ All native events from Vercel AI SDK are handled in Vel
- ✅ All stream protocol events match Vercel AI SDK output
- ✅ Event properties/fields are identical
- ✅ Edge cases are handled identically
- ✅ Documentation confirms parity
- ✅ Tests validate behavior matches

## Timeline

**Estimated effort:**
- Phase 1 (Reconnaissance): 30 minutes
- Phase 2 (Deep Analysis): 2-3 hours
- Phase 3 (Comparison): 1-2 hours
- Phase 4 (Gap Analysis): 1 hour
- Phase 5 (Remediation Plan): 1 hour
- Phase 6 (Implementation): TBD based on gaps found

**Total planning/analysis:** ~6-8 hours
**Implementation:** TBD

## Notes

- Focus on **streaming** implementations (non-streaming is secondary)
- Prioritize **text and tool call events** (primary use case)
- Document any **Vercel AI SDK bugs** we find
- Consider **future compatibility** (new event types, etc.)

---

**Status:** Plan created, awaiting execution
**Next step:** Begin Phase 1 (OpenAI reconnaissance)
