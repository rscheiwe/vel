"""
Event Translators for Vel Stream Protocol

This module provides translators for converting native provider events to Vel's
standardized stream protocol events. These translators can be used:

1. Internally by Vel providers (composition pattern)
2. Externally by orchestration libraries (like Mesh, LangGraph)

Supported Sources:
- OpenAI Chat Completions API
- OpenAI Agents SDK
- Anthropic Messages API
- Google Gemini API

## Scope & Responsibility

**Translators handle:** Protocol conversion for a SINGLE LLM response
- text-start/delta/end
- tool-input-start/delta/available
- response-metadata
- finish-message

**Translators do NOT handle:** Multi-step orchestration
- ❌ start-step / finish-step (use Agent for multi-step)
- ❌ Tool execution (translators only detect tool calls)
- ❌ Context/memory management
- ❌ Agentic loops

## When to Use What

**Use Translator directly when:**
- Building a custom orchestrator
- Using with external frameworks (Mesh, LangGraph)
- Single-shot LLM calls (no multi-step needed)
- Protocol testing/validation

**Use Agent when:**
- You need multi-step execution
- You need tool calling with execution
- You need context/session management
- You want full agentic runtime

**⚠️ Important:** If using translator directly with AI SDK frontend components,
you must manually emit orchestration events (start, start-step, finish-step, finish).
See docs/using-translators.md for a complete guide with working examples.

## Example: Translator in Custom Orchestrator

```python
from vel.providers.translators import OpenAIAPITranslator

translator = OpenAIAPITranslator()

# Your orchestration logic
while not done:
    async for chunk in openai_stream:
        event = translator.translate_chunk(chunk)

        if event.type == 'tool-input-available':
            # Your tool execution logic
            result = await execute_tool(event.tool_name, event.input)
            # Your next step logic

        elif event.type == 'text-delta':
            # Your streaming logic
            print(event.delta)
```

## Example: Agent for Full Orchestration

```python
from vel import Agent

agent = Agent(
    id='my-agent:v1',
    model={'provider': 'openai', 'model': 'gpt-4o'},
    tools=['get_weather']
)

# Agent handles everything: multi-step, tool execution, context
async for event in agent.run_stream({'message': 'What's the weather?'}):
    print(event)  # Includes start-step, finish-step, tool-output-available
```
"""
from __future__ import annotations
from typing import Any, Dict, Optional
from collections import deque
import json
import uuid

from ..events import (
    StreamEvent,
    StartEvent,
    TextStartEvent,
    TextDeltaEvent,
    TextEndEvent,
    ReasoningStartEvent,
    ReasoningDeltaEvent,
    ReasoningEndEvent,
    ToolInputStartEvent,
    ToolInputDeltaEvent,
    ToolInputAvailableEvent,  # V5 UI Stream Protocol
    ToolOutputAvailableEvent,  # V5 UI Stream Protocol
    ResponseMetadataEvent,
    SourceEvent,
    FileEvent,
    FinishMessageEvent,
    ErrorEvent
)


class OpenAIAPITranslator:
    """
    Translates OpenAI Chat Completions API events to Vel stream protocol.

    Handles:
    - Streaming chunks from openai.chat.completions.create(stream=True)
    - Text deltas
    - Tool calls with incremental arguments
    - Finish reasons

    Usage:
        >>> translator = OpenAIAPITranslator()
        >>> # Stream from OpenAI API
        >>> async for chunk in client.chat.completions.create(stream=True, ...):
        ...     vel_event = translator.translate_chunk(chunk)
        ...     if vel_event:
        ...         yield vel_event
    """

    def __init__(self):
        self._text_block_id: Optional[str] = None
        self._reasoning_block_id: Optional[str] = None  # For reasoning content (o1/o3 models)
        self._next_block_index: int = 0  # For sequential block IDs
        self._tool_calls: Dict[int, Dict[str, Any]] = {}  # tool_index -> {id, name, args_buffer}
        self._message_id: Optional[str] = None  # OpenAI message/completion ID
        self._pending_events: deque = deque()  # Queue of events to emit

    def translate_chunk(self, chunk: Dict[str, Any]) -> Optional[StreamEvent]:
        """
        Translate a single streaming chunk from OpenAI API.

        Args:
            chunk: Parsed JSON chunk from SSE stream (e.g., {"choices": [{"delta": {...}}]})

        Returns:
            StreamEvent or None if chunk should be skipped

        Example:
            >>> chunk = {"choices": [{"delta": {"content": "Hello"}}]}
            >>> event = translator.translate_chunk(chunk)
            >>> print(event.type)  # "text-start" or "text-delta"

        Note:
            This may queue additional events in _pending_events.
            Call get_pending_event() to drain them.
        """
        # Capture message ID from first chunk (for metadata only, don't emit start event)
        if self._message_id is None and 'id' in chunk:
            self._message_id = chunk['id']

        # Safe access to choices array (may be empty in usage-only chunks)
        choices = chunk.get('choices', [])
        delta = choices[0].get('delta', {}) if choices else {}
        finish_reason = choices[0].get('finish_reason') if choices else None

        # Handle usage metadata (typically in final chunk)
        # AI SDK v5 parity: Include full response metadata
        usage = chunk.get('usage')
        if usage:
            # Build usage object with all fields
            usage_dict = {
                'inputTokens': usage.get('prompt_tokens', 0),
                'outputTokens': usage.get('completion_tokens', 0),
                'totalTokens': usage.get('total_tokens', 0)
            }
            # Add reasoning tokens if present (o1 models)
            if 'completion_tokens_details' in usage:
                details = usage['completion_tokens_details']
                if 'reasoning_tokens' in details:
                    usage_dict['reasoningTokens'] = details['reasoning_tokens']

            # Build response metadata event
            return ResponseMetadataEvent(
                id=self._message_id or chunk.get('id'),
                model_id=chunk.get('model'),
                timestamp=None,  # OpenAI doesn't provide timestamp in stream
                usage=usage_dict
            )

        # Handle text content
        if 'content' in delta and delta['content']:
            content = delta['content']
            if self._text_block_id is None:
                # First text chunk - queue both start and delta events
                self._text_block_id = str(self._next_block_index)
                self._next_block_index += 1
                self._pending_events.append(TextDeltaEvent(block_id=self._text_block_id, delta=content))
                return TextStartEvent(block_id=self._text_block_id)
            # Subsequent chunks - emit delta directly
            return TextDeltaEvent(block_id=self._text_block_id, delta=content)

        # Handle reasoning content (o1/o3 models)
        # OpenAI exposes reasoning via delta.reasoning_content field
        if 'reasoning_content' in delta and delta['reasoning_content']:
            reasoning = delta['reasoning_content']
            if self._reasoning_block_id is None:
                # First reasoning chunk - queue both start and delta events
                self._reasoning_block_id = str(self._next_block_index)
                self._next_block_index += 1
                self._pending_events.append(ReasoningDeltaEvent(block_id=self._reasoning_block_id, delta=reasoning))
                return ReasoningStartEvent(block_id=self._reasoning_block_id)
            # Subsequent chunks - emit delta directly
            return ReasoningDeltaEvent(block_id=self._reasoning_block_id, delta=reasoning)

        # Handle tool calls
        # AI SDK parity: Robust handling of malformed tool_calls
        # Some providers send tool_calls[].type as empty string after first delta
        # Reference: vercel/ai#7255 - Accept "type": "" tool calls
        if delta.get('tool_calls'):  # Check if tool_calls exists and is not None
            for tc in delta['tool_calls']:
                # Use index as primary identifier (more reliable than type field)
                idx = tc.get('index')
                if idx is None:
                    # Fallback to 0 if index missing (defensive)
                    idx = 0

                if idx not in self._tool_calls:
                    # New tool call - initialize tracking
                    # Note: Don't rely on 'type' field - may be empty/missing
                    tool_id = tc.get('id')
                    if not tool_id:
                        # Generate ID if missing (defensive)
                        tool_id = f"call_{uuid.uuid4().hex[:8]}"

                    # Extract function name defensively
                    function_data = tc.get('function', {})
                    if isinstance(function_data, dict):
                        tool_name = function_data.get('name', '')
                    else:
                        tool_name = ''

                    self._tool_calls[idx] = {
                        'id': tool_id,
                        'name': tool_name,
                        'args_buffer': '',
                        'input_available_emitted': False  # Track if we've emitted tool-input-available
                    }

                    # Only emit start if we have a tool name
                    if tool_name:
                        return ToolInputStartEvent(
                            tool_call_id=tool_id,
                            tool_name=tool_name
                        )
                else:
                    # Update existing tool call with missing fields
                    # Handle case where type was empty in first delta, fields arrive later
                    existing = self._tool_calls[idx]
                    if not existing['id'] and tc.get('id'):
                        existing['id'] = tc['id']
                    if not existing['name']:
                        function_data = tc.get('function', {})
                        if isinstance(function_data, dict):
                            name = function_data.get('name', '')
                            if name:
                                existing['name'] = name
                                # Emit start now that we have the name
                                return ToolInputStartEvent(
                                    tool_call_id=existing['id'],
                                    tool_name=name
                                )

                # Accumulate function arguments (defensive checks)
                if 'function' in tc:
                    function_data = tc['function']
                    if isinstance(function_data, dict) and 'arguments' in function_data:
                        args_delta = function_data['arguments']
                        if args_delta:  # Only process non-empty deltas
                            self._tool_calls[idx]['args_buffer'] += args_delta
                            return ToolInputDeltaEvent(
                                tool_call_id=self._tool_calls[idx]['id'],
                                input_delta=args_delta
                            )

        # Handle finish
        if finish_reason:
            # End reasoning block if active (reasoning comes before text typically)
            if self._reasoning_block_id:
                reasoning_block_id = self._reasoning_block_id
                self._reasoning_block_id = None
                # Queue text-end if there's also a text block
                if self._text_block_id:
                    self._pending_events.append(TextEndEvent(block_id=self._text_block_id))
                    self._text_block_id = None
                return ReasoningEndEvent(block_id=reasoning_block_id)

            # End text block if active
            if self._text_block_id:
                text_block_id = self._text_block_id
                self._text_block_id = None
                return TextEndEvent(block_id=text_block_id)

        return None

    def finalize_tool_calls(self) -> list[StreamEvent]:
        """
        Generate events for stream completion.
        Emits any pending events, reasoning-end, text-end, and tool-input-available events.
        Call this when the stream completes.

        AI SDK parity: Guaranteed tool-input-available emission
        - Emits even if args only appeared at .done (no streaming deltas)
        - Ensures every tool call gets tool-input-available event

        Returns:
            List of StreamEvent objects
        """
        events = []

        # Drain any pending events first
        while self._pending_events:
            events.append(self._pending_events.popleft())

        # Emit reasoning-end if reasoning block is active
        if self._reasoning_block_id:
            events.append(ReasoningEndEvent(block_id=self._reasoning_block_id))
            self._reasoning_block_id = None

        # Emit text-end if text block is active
        if self._text_block_id:
            events.append(TextEndEvent(block_id=self._text_block_id))
            self._text_block_id = None

        # Emit tool-input-available for all accumulated tool calls
        # Guaranteed emission: handles case where args only appear at .done
        for tc_data in self._tool_calls.values():
            # Only emit if we haven't emitted yet
            if not tc_data.get('input_available_emitted', False):
                try:
                    args = json.loads(tc_data['args_buffer'] or '{}')
                except json.JSONDecodeError:
                    args = {}

                events.append(ToolInputAvailableEvent(
                    tool_call_id=tc_data['id'],
                    tool_name=tc_data['name'],
                    input=args,
                    provider_metadata={'openai': {'itemId': tc_data['id']}}
                ))
                tc_data['input_available_emitted'] = True

        return events

    def get_pending_event(self) -> Optional[StreamEvent]:
        """
        Get next pending event without processing a new chunk.

        Returns:
            StreamEvent or None if no pending events
        """
        if self._pending_events:
            return self._pending_events.popleft()
        return None

    def reset(self):
        """Reset translator state between messages."""
        self._text_block_id = None
        self._reasoning_block_id = None
        self._next_block_index = 0
        self._tool_calls.clear()
        self._message_id = None
        self._pending_events.clear()


class OpenAIAgentsSDKTranslator:
    """
    Translates OpenAI Agents SDK native events to Vel stream protocol.

    Handles:
    - raw_response_event (token streaming)
    - run_item_stream_event (progress updates)
    - Tool calls and completions

    Usage:
        >>> translator = OpenAIAgentsSDKTranslator()
        >>> result = Runner.run_streamed(agent, "Hello")
        >>> async for native_event in result.stream_events():
        ...     vel_event = translator.translate(native_event)
        ...     if vel_event:
        ...         yield vel_event
    """

    def __init__(self):
        self._text_block_id: Optional[str] = None
        self._next_block_index: int = 0  # For sequential block IDs

    def translate(self, native_event: Any) -> Optional[StreamEvent]:
        """
        Translate a native OpenAI Agents SDK event to Vel format.

        Args:
            native_event: Native event from OpenAI Agents SDK

        Returns:
            StreamEvent in Vel format, or None if event should be skipped
        """
        event_type = getattr(native_event, 'type', None)

        if event_type == 'raw_response_event':
            # Token-by-token streaming from LLM
            return self._translate_raw_response(native_event)

        elif event_type == 'run_item_stream_event':
            # Higher-level progress updates
            return self._translate_run_item(native_event)

        elif event_type == 'agent_updated_stream_event':
            # Agent state changes - typically skipped
            return None

        # Unknown event type
        return None

    def _translate_raw_response(self, event: Any) -> Optional[StreamEvent]:
        """Translate raw_response_event to Vel text events."""
        data = getattr(event, 'data', None)
        if not data:
            return None

        delta = getattr(data, 'delta', '')
        if not delta:
            return None

        # Start text block if not started
        if self._text_block_id is None:
            self._text_block_id = str(self._next_block_index)
            self._next_block_index += 1

        return TextDeltaEvent(
            block_id=self._text_block_id,
            delta=delta
        )

    def _translate_run_item(self, event: Any) -> Optional[StreamEvent]:
        """Translate run_item_stream_event to Vel events."""
        item = getattr(event, 'item', None)
        if not item:
            return None

        item_type = getattr(item, 'type', '')
        status = getattr(item, 'status', '')

        # Message output completed
        if item_type == 'message_output_item' and status == 'completed':
            # End text block
            if self._text_block_id:
                text_block_id = self._text_block_id
                self._text_block_id = None
                return TextEndEvent(block_id=text_block_id)

        # Tool calls
        elif 'tool' in item_type.lower():
            tool_name = getattr(item, 'name', 'unknown')
            tool_id = getattr(item, 'id', str(uuid.uuid4()))

            if status == 'in_progress':
                return ToolInputStartEvent(
                    tool_call_id=tool_id,
                    tool_name=tool_name
                )
            elif status == 'completed':
                output = getattr(item, 'output', None)
                return ToolOutputAvailableEvent(
                    tool_call_id=tool_id,
                    output=output
                )

        return None

    def reset(self):
        """Reset translator state between messages."""
        self._text_block_id = None
        self._next_block_index = 0


class AnthropicAPITranslator:
    """
    Translates Anthropic Messages API events to Vel stream protocol.

    Handles:
    - SSE streaming from Anthropic Messages API
    - Content blocks (text and tool_use)
    - Incremental JSON for tool inputs
    - Message lifecycle events

    Usage:
        >>> translator = AnthropicAPITranslator()
        >>> # Stream from Anthropic API
        >>> async for line in response.aiter_lines():
        ...     if line.startswith('data: '):
        ...         data = json.loads(line[6:])
        ...         vel_event = translator.translate_event(data)
        ...         if vel_event:
        ...             yield vel_event
    """

    def __init__(self):
        self._content_blocks: Dict[int, Dict[str, Any]] = {}  # index -> block state
        self._finish_reason: str = 'end_turn'
        self._usage_data: Dict[str, int] = {}  # Track usage for metadata event
        self._message_id: Optional[str] = None  # Message ID for metadata
        self._model_id: Optional[str] = None  # Model ID for metadata
        self._metadata_emitted: bool = False  # Track if we've emitted early metadata

    def translate_event(self, data: Dict[str, Any]) -> Optional[StreamEvent]:
        """
        Translate a parsed SSE event from Anthropic API.

        Args:
            data: Parsed JSON event data (from SSE stream)

        Returns:
            StreamEvent or None if event should be skipped
        """
        event_type = data.get('type')

        # Handle message_start
        # AI SDK parity: Emit early metadata when id/model are known
        # Reference: packages/anthropic/src/anthropic-messages-language-model.ts
        if event_type == 'message_start':
            message = data.get('message', {})

            # Capture message/model IDs
            self._message_id = message.get('id')
            self._model_id = message.get('model')

            # Track input tokens from message start
            usage = message.get('usage')
            if usage and 'input_tokens' in usage:
                self._usage_data['promptTokens'] = usage['input_tokens']

            # Emit early metadata if we have id or model
            if self._message_id or self._model_id:
                self._metadata_emitted = True
                return ResponseMetadataEvent(
                    id=self._message_id,
                    model_id=self._model_id,
                    usage=None  # Usage will be updated later
                )

            return None

        # Handle content_block_start
        elif event_type == 'content_block_start':
            index = data.get('index', 0)
            content_block = data.get('content_block', {})
            block_type = content_block.get('type')

            if block_type == 'text':
                block_id = str(index)  # Use index as block ID
                self._content_blocks[index] = {
                    'type': 'text',
                    'block_id': block_id,
                    'buffer': []
                }
                return TextStartEvent(block_id=block_id)

            elif block_type == 'thinking':
                block_id = str(index)  # Use index as block ID
                self._content_blocks[index] = {
                    'type': 'thinking',
                    'block_id': block_id,
                    'buffer': []
                }
                return ReasoningStartEvent(block_id=block_id)

            elif block_type == 'tool_use':
                tool_id = content_block.get('id', f"call_{uuid.uuid4().hex[:8]}")
                tool_name = content_block.get('name', '')
                self._content_blocks[index] = {
                    'type': 'tool_use',
                    'tool_id': tool_id,
                    'tool_name': tool_name,
                    'input_buffer': ''
                }
                return ToolInputStartEvent(
                    tool_call_id=tool_id,
                    tool_name=tool_name
                )

        # Handle content_block_delta
        elif event_type == 'content_block_delta':
            index = data.get('index', 0)
            delta = data.get('delta', {})
            delta_type = delta.get('type')

            if index in self._content_blocks:
                block = self._content_blocks[index]

                if delta_type == 'text_delta':
                    text = delta.get('text', '')
                    block['buffer'].append(text)
                    return TextDeltaEvent(
                        block_id=block['block_id'],
                        delta=text
                    )

                elif delta_type == 'thinking_delta':
                    thinking = delta.get('thinking', '')
                    block['buffer'].append(thinking)
                    return ReasoningDeltaEvent(
                        block_id=block['block_id'],
                        delta=thinking
                    )

                elif delta_type == 'input_json_delta':
                    partial_json = delta.get('partial_json', '')
                    block['input_buffer'] += partial_json
                    return ToolInputDeltaEvent(
                        tool_call_id=block['tool_id'],
                        input_delta=partial_json
                    )

        # Handle content_block_stop
        elif event_type == 'content_block_stop':
            index = data.get('index', 0)
            if index in self._content_blocks:
                block = self._content_blocks[index]

                if block['type'] == 'text':
                    return TextEndEvent(block_id=block['block_id'])

                elif block['type'] == 'thinking':
                    return ReasoningEndEvent(block_id=block['block_id'])

                elif block['type'] == 'tool_use':
                    # Parse accumulated JSON input
                    try:
                        tool_input = json.loads(block['input_buffer'] or '{}')
                    except json.JSONDecodeError:
                        tool_input = {}

                    return ToolInputAvailableEvent(
                        tool_call_id=block['tool_id'],
                        tool_name=block['tool_name'],
                        input=tool_input
                    )

        # Handle message_delta
        elif event_type == 'message_delta':
            delta = data.get('delta', {})
            self._finish_reason = delta.get('stop_reason', 'end_turn')

            # Track usage (output tokens)
            usage = data.get('usage')
            if usage:
                if 'output_tokens' in usage:
                    self._usage_data['completionTokens'] = usage['output_tokens']

        # Handle message_stop
        elif event_type == 'message_stop':
            # Check if we have usage data to emit
            if self._usage_data:
                # Note: We can't return two events, so we'll emit metadata in message_stop
                # The provider will need to handle this or we buffer
                pass
            return FinishMessageEvent(finish_reason=self._finish_reason)

        # Handle error
        elif event_type == 'error':
            error_data = data.get('error', {})
            return ErrorEvent(
                error=error_data.get('message', 'Unknown error'),
                error_code=error_data.get('code'),
                error_type=error_data.get('type'),
                provider='anthropic'
            )

        return None

    def get_metadata_event(self) -> Optional[ResponseMetadataEvent]:
        """
        Get metadata event if usage data was collected.
        Call this after message_stop to emit usage metadata.

        AI SDK parity: If early metadata was emitted, this returns a second metadata
        event with usage data. Otherwise returns complete metadata with id/model/usage.

        Returns:
            ResponseMetadataEvent or None if no usage data
        """
        if self._usage_data:
            prompt_tokens = self._usage_data.get('promptTokens', 0)
            completion_tokens = self._usage_data.get('completionTokens', 0)

            # If we already emitted early metadata, emit usage update
            # Otherwise, emit complete metadata
            return ResponseMetadataEvent(
                id=self._message_id if not self._metadata_emitted else None,
                model_id=self._model_id if not self._metadata_emitted else None,
                usage={
                    'promptTokens': prompt_tokens,
                    'completionTokens': completion_tokens,
                    'totalTokens': prompt_tokens + completion_tokens
                }
            )
        return None

    def reset(self):
        """Reset translator state between messages."""
        self._content_blocks.clear()
        self._finish_reason = 'end_turn'
        self._usage_data.clear()
        self._message_id = None
        self._model_id = None
        self._metadata_emitted = False


class GeminiAPITranslator:
    """
    Translates Google Gemini API events to Vel stream protocol.

    Handles:
    - Streaming chunks from GenerativeModel.generate_content_async(stream=True)
    - Text content
    - Function calls
    - Grounding sources (web citations)
    - Inline data (files)

    Usage:
        >>> translator = GeminiAPITranslator()
        >>> # Stream from Gemini
        >>> response = chat.send_message_async(message, stream=True)
        >>> async for chunk in response:
        ...     vel_event = translator.translate_chunk(chunk)
        ...     if vel_event:
        ...         yield vel_event
    """

    def __init__(self):
        self._text_block_id: Optional[str] = None
        self._next_block_index: int = 0  # For sequential block IDs
        self._seen_source_urls: set[str] = set()  # Deduplicate grounding sources
        self._pending_events: deque = deque()  # Queue of events to emit

    def translate_chunk(self, chunk: Any) -> Optional[StreamEvent]:
        """
        Translate a streaming chunk from Gemini API.

        Args:
            chunk: Native chunk from Gemini streaming response

        Returns:
            StreamEvent or None if chunk should be skipped
        """
        # Drain pending events first
        if self._pending_events:
            return self._pending_events.popleft()

        # Handle usage metadata
        if hasattr(chunk, 'usage_metadata'):
            usage = chunk.usage_metadata
            if hasattr(usage, 'total_token_count') and usage.total_token_count > 0:
                return ResponseMetadataEvent(
                    usage={
                        'promptTokens': getattr(usage, 'prompt_token_count', 0),
                        'completionTokens': getattr(usage, 'candidates_token_count', 0),
                        'totalTokens': getattr(usage, 'total_token_count', 0)
                    }
                )

        # Handle grounding sources (web citations)
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

        # Handle parts (inline data, function calls, code execution, etc.)
        if hasattr(chunk, 'parts'):
            for part in chunk.parts:
                # Handle inline data (files)
                if hasattr(part, 'inline_data'):
                    inline = part.inline_data
                    return FileEvent(
                        content=getattr(inline, 'data', ''),  # base64
                        mime_type=getattr(inline, 'mime_type', '')
                    )

                # Handle code execution (detect but don't emit for now)
                if hasattr(part, 'executable_code'):
                    # Log for debugging (optional implementation in future)
                    # For now, skip - this is a niche feature
                    pass

                if hasattr(part, 'code_execution_result'):
                    # Log for debugging (optional implementation in future)
                    # For now, skip - this is a niche feature
                    pass

                # Handle function calls
                # AI SDK parity: Gemini emits complete function calls (not streaming)
                # Emit both tool-input-start and tool-input-available immediately
                # Reference: packages/google/src/google-generative-ai-language-model.ts
                if hasattr(part, 'function_call'):
                    fc = part.function_call
                    tool_call_id = f"call_{uuid.uuid4().hex[:8]}"
                    tool_name = fc.name if hasattr(fc, 'name') else 'unknown'

                    # Convert args to dict (defensive)
                    args = {}
                    if hasattr(fc, 'args'):
                        try:
                            args = dict(fc.args)
                        except (TypeError, ValueError):
                            args = {}

                    # Queue tool-input-available (will be emitted after start)
                    self._pending_events.append(ToolInputAvailableEvent(
                        tool_call_id=tool_call_id,
                        tool_name=tool_name,
                        input=args
                    ))

                    # Return start first
                    return ToolInputStartEvent(
                        tool_call_id=tool_call_id,
                        tool_name=tool_name
                    )

        # Handle text content
        if hasattr(chunk, 'text') and chunk.text:
            if self._text_block_id is None:
                self._text_block_id = str(self._next_block_index)
                self._next_block_index += 1
                return TextStartEvent(block_id=self._text_block_id)
            return TextDeltaEvent(block_id=self._text_block_id, delta=chunk.text)

        return None

    def finalize_text_block(self) -> Optional[TextEndEvent]:
        """
        End the current text block if active.
        Call this when stream completes.

        Returns:
            TextEndEvent or None
        """
        if self._text_block_id:
            text_block_id = self._text_block_id
            self._text_block_id = None
            return TextEndEvent(block_id=text_block_id)
        return None

    def get_pending_event(self) -> Optional[StreamEvent]:
        """
        Get next pending event without processing a new chunk.

        Returns:
            StreamEvent or None if no pending events
        """
        if self._pending_events:
            return self._pending_events.popleft()
        return None

    def reset(self):
        """Reset translator state between messages."""
        self._text_block_id = None
        self._next_block_index = 0
        self._seen_source_urls.clear()
        self._pending_events.clear()


# Convenience functions for easy instantiation

def get_openai_api_translator() -> OpenAIAPITranslator:
    """
    Get a translator for OpenAI Chat Completions API.

    Returns:
        OpenAIAPITranslator instance

    Example:
        >>> from vel.providers.translators import get_openai_api_translator
        >>> translator = get_openai_api_translator()
    """
    return OpenAIAPITranslator()


def get_openai_agents_translator() -> OpenAIAgentsSDKTranslator:
    """
    Get a translator for OpenAI Agents SDK events.

    Returns:
        OpenAIAgentsSDKTranslator instance

    Example:
        >>> from vel.providers.translators import get_openai_agents_translator
        >>> translator = get_openai_agents_translator()
    """
    return OpenAIAgentsSDKTranslator()


def get_anthropic_translator() -> AnthropicAPITranslator:
    """
    Get a translator for Anthropic Messages API.

    Returns:
        AnthropicAPITranslator instance

    Example:
        >>> from vel.providers.translators import get_anthropic_translator
        >>> translator = get_anthropic_translator()
    """
    return AnthropicAPITranslator()


def get_gemini_translator() -> GeminiAPITranslator:
    """
    Get a translator for Google Gemini API.

    Returns:
        GeminiAPITranslator instance

    Example:
        >>> from vel.providers.translators import get_gemini_translator
        >>> translator = get_gemini_translator()
    """
    return GeminiAPITranslator()


class OpenAIResponsesAPITranslator:
    """
    Translates OpenAI Responses API events to Vel stream protocol.

    Handles the structured event format from /v1/responses endpoint:
    - response.created, response.done, response.error
    - response.text.delta, response.output_text.delta
    - response.reasoning.delta (and all variants)
    - response.output_item.added (for synthesis)
    - response.function_call_arguments.delta

    Based on EXTENDED_PARITY.md mapping and AI SDK V5 parity requirements.

    Key AI SDK Parity Features:
    - Normalizes ALL reasoning variants to single event type
    - Deduplicates reasoning-start events
    - Maps provider-executed tools (web_search_call, computer_call)
    - Extracts and emits sources/citations
    - Early metadata emission (id/model) with later usage updates
    """

    def __init__(self):
        self._text_block_id: Optional[str] = None
        self._reasoning_block_id: Optional[str] = None
        self._next_block_index: int = 0
        self._tool_calls: Dict[str, Dict[str, Any]] = {}  # tool_call_id -> {name, args_buffer}
        self._pending_events: deque = deque()
        self._active_output_items: Dict[str, str] = {}  # item_id -> type (message, function_call, etc.)
        self._seen_reasoning_ids: set = set()  # Track reasoning block IDs to prevent duplicate starts
        self._response_id: Optional[str] = None  # Response ID for metadata
        self._model_id: Optional[str] = None  # Model ID for metadata
        self._metadata_emitted: bool = False  # Track if we've emitted early metadata

    def translate_event(self, event: Dict[str, Any]) -> Optional[StreamEvent]:
        """
        Translate a Responses API event to Vel stream protocol.

        Args:
            event: Parsed event from Responses API SSE stream

        Returns:
            StreamEvent or None if event should be skipped
        """
        # Drain pending events first
        if self._pending_events:
            return self._pending_events.popleft()

        event_type = event.get('type', '')

        # === Lifecycle events ===
        if event_type == 'response.created':
            # Emit early metadata (AI SDK parity: emit when id/model known)
            response_data = event.get('response', {})
            self._response_id = response_data.get('id')
            self._model_id = response_data.get('model')

            if self._response_id or self._model_id:
                self._metadata_emitted = True
                return ResponseMetadataEvent(
                    id=self._response_id,
                    model_id=self._model_id,
                    usage=None  # Usage comes later
                )
            return None

        if event_type == 'response.in_progress':
            return None  # Handled at Agent level

        if event_type in ('response.completed', 'response.done'):
            # Emit usage metadata if not yet emitted
            response_data = event.get('response', {})
            usage_data = response_data.get('usage')
            if usage_data and not self._metadata_emitted:
                # Emit metadata with usage if we haven't emitted yet
                self._pending_events.append(ResponseMetadataEvent(
                    id=self._response_id,
                    model_id=self._model_id,
                    usage={
                        'promptTokens': usage_data.get('input_tokens', 0),
                        'completionTokens': usage_data.get('output_tokens', 0),
                        'totalTokens': usage_data.get('total_tokens', 0)
                    }
                ))
            elif usage_data and self._metadata_emitted:
                # Update with usage (second metadata event)
                self._pending_events.append(ResponseMetadataEvent(
                    id=self._response_id,
                    model_id=self._model_id,
                    usage={
                        'promptTokens': usage_data.get('input_tokens', 0),
                        'completionTokens': usage_data.get('output_tokens', 0),
                        'totalTokens': usage_data.get('total_tokens', 0)
                    }
                ))

            # Return finish handled at Agent level, but emit any pending end events
            return self._finalize()

        if event_type == 'response.error':
            error_data = event.get('error', {})
            return ErrorEvent(
                error=error_data.get('message', 'Unknown error'),
                error_code=error_data.get('code'),
                error_type=error_data.get('type'),
                provider='openai-responses'
            )

        # === Text events ===
        # Normalize both response.text.delta and response.output_text.delta
        if event_type in ('response.text.delta', 'response.output_text.delta'):
            text = event.get('delta', '') or event.get('text', '')
            if text:
                if self._text_block_id is None:
                    self._text_block_id = str(self._next_block_index)
                    self._next_block_index += 1
                    self._pending_events.append(TextDeltaEvent(block_id=self._text_block_id, delta=text))
                    return TextStartEvent(block_id=self._text_block_id)
                return TextDeltaEvent(block_id=self._text_block_id, delta=text)

        if event_type in ('response.text.done', 'response.output_text.done'):
            if self._text_block_id:
                block_id = self._text_block_id
                self._text_block_id = None
                return TextEndEvent(block_id=block_id)

        # === Reasoning events (all variants) ===
        # AI SDK parity: Normalize ALL reasoning variants to single event type
        # Variants: response.reasoning.delta, response.reasoning_summary.delta, response.reasoning_summary_text.delta
        # Reference: packages/openai/src/responses/openai-responses-language-model.ts
        reasoning_delta_variants = [
            'response.reasoning.delta',
            'response.reasoning_summary.delta',
            'response.reasoning_summary_text.delta'
        ]
        if event_type in reasoning_delta_variants:
            # Extract reasoning text from various field names
            reasoning = event.get('delta', '') or event.get('reasoning', '') or event.get('summary', '') or event.get('text', '')

            if reasoning:
                # Get or create reasoning block ID
                if self._reasoning_block_id is None:
                    # Use item ID from event if available for stable IDs
                    item_id = event.get('item_id')
                    if item_id:
                        self._reasoning_block_id = item_id
                    else:
                        self._reasoning_block_id = str(self._next_block_index)
                        self._next_block_index += 1

                    # Deduplicate: only emit start if we haven't seen this ID before
                    if self._reasoning_block_id not in self._seen_reasoning_ids:
                        self._seen_reasoning_ids.add(self._reasoning_block_id)
                        self._pending_events.append(ReasoningDeltaEvent(block_id=self._reasoning_block_id, delta=reasoning))
                        return ReasoningStartEvent(block_id=self._reasoning_block_id)
                    # If already seen, just emit delta

                return ReasoningDeltaEvent(block_id=self._reasoning_block_id, delta=reasoning)

        # Normalize: reasoning.done, reasoning_summary.done, reasoning_summary_text.done
        reasoning_done_variants = [
            'response.reasoning.done',
            'response.reasoning_summary.done',
            'response.reasoning_summary_text.done'
        ]
        if event_type in reasoning_done_variants:
            if self._reasoning_block_id:
                block_id = self._reasoning_block_id
                self._reasoning_block_id = None
                return ReasoningEndEvent(block_id=block_id)

        # === Output item events (used for synthesis) ===
        if event_type == 'response.output_item.added':
            item = event.get('item', {})
            item_id = item.get('id')
            item_type = item.get('type')  # message, function_call, web_search_call, computer_call, reasoning

            if item_id and item_type:
                self._active_output_items[item_id] = item_type

                # Synthesize reasoning-start for reasoning items (o1/o3 models)
                if item_type in ('reasoning', 'thinking'):
                    if self._reasoning_block_id is None:
                        self._reasoning_block_id = item_id  # Use OpenAI's ID
                        return ReasoningStartEvent(block_id=self._reasoning_block_id)

                # Synthesize tool-input-start for tool calls
                if item_type in ('function_call', 'web_search_call', 'computer_call'):
                    tool_name = item.get('name', item_type)
                    self._tool_calls[item_id] = {
                        'name': tool_name,
                        'args_buffer': ''
                    }
                    return ToolInputStartEvent(tool_call_id=item_id, tool_name=tool_name)
            return None

        if event_type == 'response.content_part.added':
            # Used to synthesize text-start when first text part arrives
            # Already handled by response.text.delta logic above
            return None

        # === Tool call arguments ===
        if event_type == 'response.function_call_arguments.delta':
            call_id = event.get('call_id') or event.get('id')
            args_delta = event.get('delta', '') or event.get('arguments', '')

            if call_id and args_delta:
                if call_id in self._tool_calls:
                    self._tool_calls[call_id]['args_buffer'] += args_delta
                    return ToolInputDeltaEvent(tool_call_id=call_id, input_delta=args_delta)

        if event_type == 'response.function_call_arguments.done':
            call_id = event.get('call_id') or event.get('id')
            if call_id and call_id in self._tool_calls:
                tc = self._tool_calls[call_id]
                try:
                    args = json.loads(tc['args_buffer'] or '{}')
                except json.JSONDecodeError:
                    args = {}
                return ToolInputAvailableEvent(
                    tool_call_id=call_id,
                    tool_name=tc['name'],
                    input=args,
                    provider_metadata={'openai': {'itemId': call_id}}
                )

        # === Output item done (for provider-executed tools and reasoning) ===
        if event_type == 'response.output_item.done':
            item = event.get('item', {})
            item_id = item.get('id')
            item_type = self._active_output_items.get(item_id)

            # Reasoning items (o1/o3 models) - emit reasoning-end
            if item_type in ('reasoning', 'thinking'):
                if self._reasoning_block_id == item_id:
                    self._reasoning_block_id = None
                    return ReasoningEndEvent(block_id=item_id)

            # Provider-executed tools (web_search, computer)
            # AI SDK parity: Map web_search_call and computer_call to tool-output-available
            # Reference: packages/openai/src/responses/openai-responses-language-model.ts
            if item_type in ('web_search_call', 'computer_call'):
                # Guaranteed tool-input-available: emit even if args only appear at .done
                if item_id in self._tool_calls:
                    tc = self._tool_calls[item_id]
                    # Check if we haven't emitted tool-input-available yet
                    # This handles case where args only appear at .done (no deltas)
                    if tc.get('args_buffer') is not None:
                        try:
                            args = json.loads(tc['args_buffer'] or '{}')
                        except json.JSONDecodeError:
                            args = {}
                        self._pending_events.append(ToolInputAvailableEvent(
                            tool_call_id=item_id,
                            tool_name=tc['name'],
                            input=args,
                            provider_metadata={'openai': {'itemId': item_id}}
                        ))
                else:
                    # Args never streamed, but we still need tool-input-available
                    # Extract args from item if present
                    args = item.get('arguments', {})
                    if args:
                        self._pending_events.append(ToolInputAvailableEvent(
                            tool_call_id=item_id,
                            tool_name=item.get('name', item_type),
                            input=args,
                            provider_metadata={'openai': {'itemId': item_id}}
                        ))

                # Extract sources from web_search results (AI SDK parity)
                # web_search_call results contain sources array
                if item_type == 'web_search_call':
                    sources_data = []
                    result_data = item.get('result', {})

                    # Extract sources from result.sources or result.action.sources
                    sources_list = result_data.get('sources') or result_data.get('action', {}).get('sources', [])

                    for source in sources_list:
                        source_entry = {
                            'type': 'web',
                            'url': source.get('url', '') or source.get('uri', ''),
                            'title': source.get('title', ''),
                        }
                        # Include snippet if available
                        if 'snippet' in source:
                            source_entry['snippet'] = source['snippet']
                        # Preserve provider ID (sourceId from OpenAI)
                        if 'id' in source:
                            source_entry['sourceId'] = source['id']

                        sources_data.append(source_entry)

                    if sources_data:
                        self._pending_events.append(SourceEvent(sources=sources_data))

                # Emit tool-output-available with result
                output = item.get('result') or item.get('output', {})

                # NOTE: providerMetadata removed to strictly match AI SDK v5 spec
                return ToolOutputAvailableEvent(
                    tool_call_id=item_id,
                    output=output
                )

        return None

    def _finalize(self) -> Optional[StreamEvent]:
        """Finalize any open blocks."""
        # Drain pending first
        if self._pending_events:
            return self._pending_events.popleft()

        # Close reasoning block
        if self._reasoning_block_id:
            block_id = self._reasoning_block_id
            self._reasoning_block_id = None
            if self._text_block_id:
                self._pending_events.append(TextEndEvent(block_id=self._text_block_id))
                self._text_block_id = None
            return ReasoningEndEvent(block_id=block_id)

        # Close text block
        if self._text_block_id:
            block_id = self._text_block_id
            self._text_block_id = None
            return TextEndEvent(block_id=block_id)

        # Emit any remaining tool-input-available
        for call_id, tc in self._tool_calls.items():
            if tc['args_buffer']:
                try:
                    args = json.loads(tc['args_buffer'] or '{}')
                except json.JSONDecodeError:
                    args = {}
                self._pending_events.append(ToolInputAvailableEvent(
                    tool_call_id=call_id,
                    tool_name=tc['name'],
                    input=args,
                    provider_metadata={'openai': {'itemId': call_id}}
                ))

        if self._pending_events:
            return self._pending_events.popleft()

        return None

    def get_pending_event(self) -> Optional[StreamEvent]:
        """Get next pending event without processing new event."""
        if self._pending_events:
            return self._pending_events.popleft()
        return None

    def reset(self):
        """Reset translator state."""
        self._text_block_id = None
        self._reasoning_block_id = None
        self._next_block_index = 0
        self._tool_calls.clear()
        self._pending_events.clear()
        self._active_output_items.clear()
        self._seen_reasoning_ids.clear()
        self._response_id = None
        self._model_id = None
        self._metadata_emitted = False


def get_openai_responses_translator() -> OpenAIResponsesAPITranslator:
    """
    Get a translator for OpenAI Responses API (/v1/responses).

    Use this for:
    - OpenAI o1/o3 models with reasoning
    - Provider-executed tools (web_search, computer use)
    - Output synthesis with citations

    Returns:
        OpenAIResponsesAPITranslator instance
    """
    return OpenAIResponsesAPITranslator()


__all__ = [
    'OpenAIAPITranslator',
    'OpenAIResponsesAPITranslator',
    'OpenAIAgentsSDKTranslator',
    'AnthropicAPITranslator',
    'GeminiAPITranslator',
    'get_openai_api_translator',
    'get_openai_responses_translator',
    'get_openai_agents_translator',
    'get_anthropic_translator',
    'get_gemini_translator',
]
