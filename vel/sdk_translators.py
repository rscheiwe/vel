"""
SDK Event Translators

Translates native SDK events from various agent frameworks to Vel's
standardized stream protocol events.

These translators are meant to be used by orchestration libraries (like Mesh)
that want to use the actual SDK/agent but get consistent event formatting.
"""
from __future__ import annotations
from typing import Any, Dict, Optional
from .events import (
    StreamEvent, TextStartEvent, TextDeltaEvent, TextEndEvent,
    ToolInputStartEvent, ToolInputDeltaEvent, ToolInputAvailableEvent,
    ToolOutputAvailableEvent, FinishMessageEvent, ErrorEvent
)
import uuid


class OpenAIAgentsSDKTranslator:
    """
    Translates OpenAI Agents SDK native events to Vel stream protocol events.

    Usage:
        >>> translator = OpenAIAgentsSDKTranslator()
        >>> # Get native event from OpenAI Agents SDK
        >>> vel_event = translator.translate(native_event)
        >>> print(vel_event.to_dict())
    """

    def __init__(self):
        self._text_block_id: Optional[str] = None

    def translate(self, native_event: Any) -> Optional[StreamEvent]:
        """
        Translate a native OpenAI Agents SDK event to Vel format.

        Args:
            native_event: Native event from OpenAI Agents SDK
                         (from Runner.run_streamed().stream_events())

        Returns:
            StreamEvent in Vel format, or None if event should be skipped

        Example:
            >>> result = Runner.run_streamed(agent, "Hello")
            >>> async for native_event in result.stream_events():
            ...     vel_event = translator.translate(native_event)
            ...     if vel_event:
            ...         print(vel_event.to_dict())
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
            # Note: We emit TextStartEvent separately, not here
            # This is just for the delta

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


def get_openai_agents_translator() -> OpenAIAgentsSDKTranslator:
    """
    Get a translator for OpenAI Agents SDK events.

    Returns:
        OpenAIAgentsSDKTranslator instance

    Example:
        >>> from vel import get_openai_agents_translator
        >>> translator = get_openai_agents_translator()
        >>> vel_event = translator.translate(native_event)
    """
    return OpenAIAgentsSDKTranslator()


__all__ = [
    'OpenAIAgentsSDKTranslator',
    'get_openai_agents_translator',
]
