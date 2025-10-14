"""
Event Translators for Vel Stream Protocol

This module provides translators for converting native provider events to Vel's
standardized stream protocol events. These translators can be used:

1. Internally by Vel providers (composition pattern)
2. Externally by orchestration libraries (like Mesh)

Supported Sources:
- OpenAI Chat Completions API
- OpenAI Agents SDK
- Anthropic Messages API
- Google Gemini API
"""
from __future__ import annotations
from typing import Any, Dict, Optional
import json
import uuid

from ..events import (
    StreamEvent,
    TextStartEvent,
    TextDeltaEvent,
    TextEndEvent,
    ToolInputStartEvent,
    ToolInputDeltaEvent,
    ToolInputAvailableEvent,
    ToolOutputAvailableEvent,
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
        self._tool_calls: Dict[int, Dict[str, Any]] = {}  # tool_index -> {id, name, args_buffer}

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
            >>> print(event.type)  # "text-delta"
        """
        delta = chunk.get('choices', [{}])[0].get('delta', {})
        finish_reason = chunk.get('choices', [{}])[0].get('finish_reason')

        # Handle text content
        if 'content' in delta and delta['content']:
            content = delta['content']
            if self._text_block_id is None:
                self._text_block_id = str(uuid.uuid4())
                # Return TextStartEvent first
                return TextStartEvent(block_id=self._text_block_id)
            return TextDeltaEvent(block_id=self._text_block_id, delta=content)

        # Handle tool calls
        if 'tool_calls' in delta:
            for tc in delta['tool_calls']:
                idx = tc.get('index', 0)
                if idx not in self._tool_calls:
                    # New tool call
                    tool_id = tc.get('id', f"call_{uuid.uuid4().hex[:8]}")
                    tool_name = tc.get('function', {}).get('name', '')
                    self._tool_calls[idx] = {
                        'id': tool_id,
                        'name': tool_name,
                        'args_buffer': ''
                    }
                    if tool_name:
                        return ToolInputStartEvent(
                            tool_call_id=tool_id,
                            tool_name=tool_name
                        )

                # Accumulate function arguments
                if 'function' in tc and 'arguments' in tc['function']:
                    args_delta = tc['function']['arguments']
                    self._tool_calls[idx]['args_buffer'] += args_delta
                    return ToolInputDeltaEvent(
                        tool_call_id=self._tool_calls[idx]['id'],
                        input_delta=args_delta
                    )

        # Handle finish
        if finish_reason:
            # End text block if active
            if self._text_block_id:
                text_block_id = self._text_block_id
                self._text_block_id = None
                return TextEndEvent(block_id=text_block_id)

        return None

    def finalize_tool_calls(self) -> list[StreamEvent]:
        """
        Generate ToolInputAvailableEvent for all accumulated tool calls.
        Call this when the stream completes.

        Returns:
            List of ToolInputAvailableEvent events
        """
        events = []
        for tc_data in self._tool_calls.values():
            try:
                args = json.loads(tc_data['args_buffer'] or '{}')
            except json.JSONDecodeError:
                args = {}
            events.append(ToolInputAvailableEvent(
                tool_call_id=tc_data['id'],
                tool_name=tc_data['name'],
                input=args
            ))
        return events

    def reset(self):
        """Reset translator state between messages."""
        self._text_block_id = None
        self._tool_calls.clear()


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
            self._text_block_id = str(uuid.uuid4())

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
        if event_type == 'message_start':
            return None  # Could emit StartEvent if needed

        # Handle content_block_start
        elif event_type == 'content_block_start':
            index = data.get('index', 0)
            content_block = data.get('content_block', {})
            block_type = content_block.get('type')

            if block_type == 'text':
                block_id = str(uuid.uuid4())
                self._content_blocks[index] = {
                    'type': 'text',
                    'block_id': block_id,
                    'buffer': []
                }
                return TextStartEvent(block_id=block_id)

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

        # Handle message_stop
        elif event_type == 'message_stop':
            return FinishMessageEvent(finish_reason=self._finish_reason)

        # Handle error
        elif event_type == 'error':
            error_msg = data.get('error', {}).get('message', 'Unknown error')
            return ErrorEvent(error=error_msg)

        return None

    def reset(self):
        """Reset translator state between messages."""
        self._content_blocks.clear()
        self._finish_reason = 'end_turn'


class GeminiAPITranslator:
    """
    Translates Google Gemini API events to Vel stream protocol.

    Handles:
    - Streaming chunks from GenerativeModel.generate_content_async(stream=True)
    - Text content
    - Function calls

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

    def translate_chunk(self, chunk: Any) -> Optional[StreamEvent]:
        """
        Translate a streaming chunk from Gemini API.

        Args:
            chunk: Native chunk from Gemini streaming response

        Returns:
            StreamEvent or None if chunk should be skipped
        """
        # Handle text content
        if hasattr(chunk, 'text') and chunk.text:
            if self._text_block_id is None:
                self._text_block_id = str(uuid.uuid4())
                return TextStartEvent(block_id=self._text_block_id)
            return TextDeltaEvent(block_id=self._text_block_id, delta=chunk.text)

        # Handle function calls
        if hasattr(chunk, 'parts'):
            for part in chunk.parts:
                if hasattr(part, 'function_call'):
                    fc = part.function_call
                    tool_call_id = f"call_{uuid.uuid4().hex[:8]}"
                    tool_name = fc.name

                    # Convert args to dict
                    args = dict(fc.args) if hasattr(fc, 'args') else {}

                    # Gemini emits complete function calls, so we emit both start and available
                    # Return start first, caller needs to handle available separately
                    return ToolInputStartEvent(
                        tool_call_id=tool_call_id,
                        tool_name=tool_name
                    )

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

    def reset(self):
        """Reset translator state between messages."""
        self._text_block_id = None


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


__all__ = [
    'OpenAIAPITranslator',
    'OpenAIAgentsSDKTranslator',
    'AnthropicAPITranslator',
    'GeminiAPITranslator',
    'get_openai_api_translator',
    'get_openai_agents_translator',
    'get_anthropic_translator',
    'get_gemini_translator',
]
