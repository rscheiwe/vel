"""
Message Reducer - Aggregate streaming events into Vercel AI SDK message format

This module provides a stateful reducer that transforms Vel's streaming events
into Vercel AI SDK compatible message structures for storage and frontend use.
"""
from __future__ import annotations
import uuid
from typing import Any, Dict, List, Optional


class MessageReducer:
    """
    Stateful reducer that aggregates streaming events into Vercel AI SDK message format.

    Handles one user-assistant exchange at a time. Call reset() between exchanges.

    Example:
        reducer = MessageReducer()

        # Add user message
        user_msg = reducer.add_user_message("What's the weather?")

        # Process streaming events
        async for event in agent.run_stream({'message': 'What\'s the weather?'}):
            reducer.process_event(event)

        # Get complete messages
        messages = reducer.get_messages()  # [user_msg, assistant_msg]

        # Reset for next exchange
        reducer.reset()
    """

    def __init__(self):
        """Initialize the message reducer"""
        self.reset()

    def reset(self):
        """Reset state for a new user-assistant exchange"""
        self._user_message: Optional[Dict[str, Any]] = None
        self._assistant_id = self._generate_id()
        self._parts: List[Dict[str, Any]] = []

        # Track text blocks (aggregated from text-delta events)
        self._text_blocks: Dict[str, List[str]] = {}  # block_id -> list of deltas
        self._accumulated_text: List[str] = []  # All text chunks across all blocks

        # Track tool calls (merge input + output)
        self._tool_calls: Dict[str, Dict[str, Any]] = {}  # toolCallId -> {tool_name, input, output}

        # Track provider metadata for text and response
        self._response_metadata: Optional[Dict[str, Any]] = None
        self._message_id: Optional[str] = None  # OpenAI message ID for providerMetadata

    def _generate_id(self) -> str:
        """Generate a unique message ID"""
        # Use first 16 chars of UUID (similar to Vercel AI SDK format)
        return str(uuid.uuid4()).replace('-', '')[:16]

    def add_user_message(
        self,
        text: str,
        message_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Create a user message.

        Args:
            text: User's message text
            message_id: Optional custom message ID (generated if not provided)
            metadata: Optional metadata dict

        Returns:
            User message dict in Vercel AI SDK format
        """
        self._user_message = {
            'id': message_id or self._generate_id(),
            'role': 'user',
            'parts': [
                {
                    'type': 'text',
                    'text': text
                }
            ],
            'metadata': metadata
        }
        return self._user_message

    def process_event(self, event: Dict[str, Any]) -> None:
        """
        Process a single streaming event and update assistant message state.

        Args:
            event: Stream event dict from agent.run_stream()
        """
        event_type = event.get('type')

        if event_type == 'start':
            self._handle_start(event)

        elif event_type == 'step-start':
            self._handle_step_start(event)

        elif event_type == 'step-finish':
            self._handle_step_finish(event)

        elif event_type == 'tool-input-available':
            self._handle_tool_input_available(event)

        elif event_type == 'tool-output-available':
            self._handle_tool_output_available(event)

        elif event_type == 'text-start':
            self._handle_text_start(event)

        elif event_type == 'text-delta':
            self._handle_text_delta(event)

        elif event_type == 'text-end':
            self._handle_text_end(event)

        elif event_type == 'response-metadata':
            self._handle_response_metadata(event)

        elif event_type == 'finish-message':
            self._handle_finish_message(event)

        elif event_type == 'error':
            self._handle_error(event)

        # Other event types (reasoning-start, etc.) can be added as needed

    def _handle_start(self, event: Dict[str, Any]) -> None:
        """Handle start event - capture message ID if available"""
        message_id = event.get('messageId') or event.get('message_id')
        if message_id:
            self._message_id = message_id

    def _handle_response_metadata(self, event: Dict[str, Any]) -> None:
        """Handle response-metadata event - capture for provider metadata"""
        self._response_metadata = {
            'id': event.get('id'),
            'modelId': event.get('modelId') or event.get('model_id'),
            'timestamp': event.get('timestamp'),
            'usage': event.get('usage')
        }
        # Also capture message ID if present
        if event.get('id'):
            self._message_id = event.get('id')

    def _handle_step_start(self, event: Dict[str, Any]) -> None:
        """Handle step-start event"""
        self._parts.append({
            'type': 'step-start'
        })

    def _handle_step_finish(self, event: Dict[str, Any]) -> None:
        """Handle step-finish event (optional, not always included in parts)"""
        # Note: Based on the example, step-finish events are not included in parts
        # Only step-start events appear. If needed, uncomment below:
        # self._parts.append({'type': 'step-finish'})
        pass

    def _flush_accumulated_text(self) -> None:
        """Flush accumulated text as a single text part with provider metadata"""
        if self._accumulated_text:
            full_text = ''.join(self._accumulated_text)
            if full_text.strip():  # Only add if non-empty after stripping
                text_part = {
                    'type': 'text',
                    'text': full_text,
                    'state': 'done'
                }

                # Add providerMetadata if we have a message ID
                if self._message_id:
                    text_part['providerMetadata'] = {
                        'openai': {
                            'itemId': self._message_id
                        }
                    }

                self._parts.append(text_part)
            self._accumulated_text.clear()

    def _handle_tool_input_available(self, event: Dict[str, Any]) -> None:
        """Handle tool-input-available event - store input, wait for output"""
        # Flush any accumulated text before the tool call
        self._flush_accumulated_text()

        tool_call_id = event.get('toolCallId', '')
        tool_name = event.get('toolName', '')
        tool_input = event.get('input', {})

        # Store tool call input
        self._tool_calls[tool_call_id] = {
            'tool_name': tool_name,
            'input': tool_input,
            'output': None  # Will be filled by tool-output-available
        }

    def _handle_tool_output_available(self, event: Dict[str, Any]) -> None:
        """Handle tool-output-available event - merge with input and create part"""
        tool_call_id = event.get('toolCallId', '')
        tool_output = event.get('output', {})

        # Get stored tool call
        tool_call = self._tool_calls.get(tool_call_id)
        if not tool_call:
            # Output arrived before input (shouldn't happen, but handle gracefully)
            tool_call = {
                'tool_name': 'unknown',
                'input': {},
                'output': tool_output
            }
            self._tool_calls[tool_call_id] = tool_call
        else:
            # Merge output with stored input
            tool_call['output'] = tool_output

        # Create tool part (format: tool-{snake_case_name})
        tool_name = tool_call['tool_name']
        part = {
            'type': f"tool-{tool_name}",  # e.g., "tool-get_weather"
            'toolCallId': tool_call_id,
            'state': 'output-available',
            'input': tool_call['input'],
            'output': tool_output
        }

        # Add provider metadata if available
        if 'callProviderMetadata' in event:
            part['callProviderMetadata'] = event['callProviderMetadata']

        self._parts.append(part)

    def _handle_text_start(self, event: Dict[str, Any]) -> None:
        """Handle text-start event - initialize text block"""
        block_id = event.get('id', 'default')
        self._text_blocks[block_id] = []

    def _handle_text_delta(self, event: Dict[str, Any]) -> None:
        """Handle text-delta event - aggregate text chunks"""
        delta = event.get('delta', '')
        # Accumulate ALL text chunks into a single list
        self._accumulated_text.append(delta)

    def _handle_text_end(self, event: Dict[str, Any]) -> None:
        """Handle text-end event - just marks end of text block, don't flush yet"""
        # Don't create parts here - let text accumulate across multiple blocks
        # Text will be flushed when we hit a tool call or finish-message
        pass

    def _handle_finish_message(self, event: Dict[str, Any]) -> None:
        """Handle finish-message event - mark completion and flush remaining text"""
        # Flush any remaining accumulated text
        self._flush_accumulated_text()

    def _handle_error(self, event: Dict[str, Any]) -> None:
        """Handle error event"""
        error_msg = event.get('error', 'Unknown error')

        # Add error as a part
        self._parts.append({
            'type': 'error',
            'error': error_msg
        })

    def get_assistant_message(
        self,
        message_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Get the aggregated assistant message.

        Args:
            message_id: Optional custom message ID (uses generated ID if not provided)
            metadata: Optional metadata dict

        Returns:
            Assistant message dict in Vercel AI SDK format
        """
        return {
            'id': message_id or self._assistant_id,
            'role': 'assistant',
            'parts': self._parts.copy(),
            'metadata': metadata
        }

    def get_messages(
        self,
        user_metadata: Optional[Dict[str, Any]] = None,
        assistant_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Get both user and assistant messages.

        Args:
            user_metadata: Optional metadata for user message
            assistant_metadata: Optional metadata for assistant message

        Returns:
            List containing [user_message, assistant_message]
        """
        messages = []

        if self._user_message:
            # Update metadata if provided
            if user_metadata is not None:
                self._user_message['metadata'] = user_metadata
            messages.append(self._user_message)

        messages.append(self.get_assistant_message(metadata=assistant_metadata))

        return messages

    def get_user_message(self) -> Optional[Dict[str, Any]]:
        """Get the user message (if set)"""
        return self._user_message

    def has_user_message(self) -> bool:
        """Check if user message has been set"""
        return self._user_message is not None

    def get_parts_count(self) -> int:
        """Get the number of parts in the assistant message"""
        return len(self._parts)
