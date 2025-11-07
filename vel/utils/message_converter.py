"""
Python implementation of Vercel AI SDK's convertToModelMessages().

Converts UIMessage format (from useChat hook) to ModelMessage format (for LLMs).

UIMessage contains UI state with executed tools (input + output in same message).
ModelMessage separates tool calls from results into distinct messages.

Supports:
- Text content
- Reasoning content (OpenAI o1/o3, Anthropic extended thinking)
- Images and files
- Tool executions (splits into tool-call and tool-result messages)
- UI-only elements (filters them out)
"""
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger('vel.utils.message_converter')


# UI-only part types that should be filtered out
UI_ONLY_PARTS = {
    'step-start',
    'step-finish',
    'step-result',
}


def convert_to_model_messages(ui_messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Convert UIMessage array to ModelMessage array.

    UIMessage format (what useChat produces):
    {
        'id': 'msg-1',
        'role': 'assistant',
        'parts': [
            {
                'type': 'tool-websearch',
                'toolCallId': 'call_123',
                'state': 'output-available',
                'input': {...},
                'output': {...}  # Both input and output present
            }
        ]
    }

    ModelMessage format (what LLMs expect):
    [
        {
            'role': 'assistant',
            'content': [
                {'type': 'tool-call', 'toolCallId': 'call_123', 'toolName': 'tool-websearch', 'input': {...}}
            ]
        },
        {
            'role': 'tool',
            'content': [
                {'type': 'tool-result', 'toolCallId': 'call_123', 'toolName': 'tool-websearch', 'output': {...}}
            ]
        }
    ]

    Args:
        ui_messages: List of UIMessage objects (from useChat or stored in DB)

    Returns:
        List of ModelMessage objects (ready for LLM APIs)

    Raises:
        ValueError: If message format is invalid
    """
    if not isinstance(ui_messages, list):
        raise ValueError(f"ui_messages must be a list, got {type(ui_messages)}")

    model_messages = []

    for idx, ui_msg in enumerate(ui_messages):
        role = ui_msg.get('role')

        if not role:
            raise ValueError(f"Message at index {idx} missing 'role' field")

        # System messages - always simple string content
        if role == 'system':
            content = ui_msg.get('content', ui_msg.get('parts', [{}])[0].get('text', ''))
            model_messages.append({
                'role': 'system',
                'content': content
            })
            continue

        # Get message parts
        parts = ui_msg.get('parts', [])

        # Handle legacy format: content field instead of parts
        if not parts and 'content' in ui_msg:
            content = ui_msg['content']
            # If content is already a string, wrap it as a text part
            if isinstance(content, str):
                parts = [{'type': 'text', 'text': content}]
            # If content is already an array, use it as parts
            elif isinstance(content, list):
                parts = content
            else:
                parts = [{'type': 'text', 'text': str(content)}]

        # Process parts based on message role
        if role == 'user':
            model_messages.extend(_process_user_message(parts))
        elif role == 'assistant':
            model_messages.extend(_process_assistant_message(parts))
        elif role == 'tool':
            # Tool messages in UIMessage are rare but possible
            model_messages.extend(_process_tool_message(parts))
        else:
            logger.warning(f"Unknown message role '{role}' at index {idx}, skipping")

    return model_messages


def _process_user_message(parts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Process user message parts.

    User messages can contain:
    - text
    - images
    - files
    """
    if not parts:
        return [{'role': 'user', 'content': ''}]

    content_parts = []

    for part in parts:
        part_type = part.get('type', 'text')

        # Filter UI-only parts
        if part_type in UI_ONLY_PARTS:
            continue

        if part_type == 'text':
            text = part.get('text', '')
            if text:  # Only add non-empty text
                content_parts.append({
                    'type': 'text',
                    'text': text
                })

        elif part_type == 'image':
            # Image part
            image_data = part.get('image', part.get('data'))
            mime_type = part.get('mimeType', 'image/png')
            if image_data:
                content_parts.append({
                    'type': 'image',
                    'image': image_data,
                    'mimeType': mime_type
                })

        elif part_type == 'file':
            # File part (PDF, etc.)
            file_data = part.get('data')
            mime_type = part.get('mimeType', 'application/octet-stream')
            file_uri = part.get('fileUri', part.get('url'))

            file_part = {'type': 'file', 'mimeType': mime_type}
            if file_data:
                file_part['data'] = file_data
            if file_uri:
                file_part['fileUri'] = file_uri

            content_parts.append(file_part)

        else:
            logger.warning(f"Unknown user content part type: {part_type}")

    # If only one text part, use string content
    if len(content_parts) == 1 and content_parts[0].get('type') == 'text':
        return [{'role': 'user', 'content': content_parts[0]['text']}]

    # Otherwise use array content
    return [{'role': 'user', 'content': content_parts}] if content_parts else []


def _process_assistant_message(parts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Process assistant message parts.

    Assistant messages can contain:
    - text
    - reasoning (OpenAI o1/o3, Anthropic thinking)
    - tool calls (with optional output → split into separate messages)
    """
    if not parts:
        return [{'role': 'assistant', 'content': ''}]

    assistant_content = []
    tool_results = []

    for part in parts:
        part_type = part.get('type', 'text')

        # Filter UI-only parts
        if part_type in UI_ONLY_PARTS:
            continue

        if part_type == 'text':
            text = part.get('text', '')
            if text:
                assistant_content.append({
                    'type': 'text',
                    'text': text
                })

        elif part_type == 'reasoning':
            # Reasoning content (OpenAI o1/o3, Anthropic thinking blocks)
            reasoning_text = part.get('text', '')
            if reasoning_text:
                assistant_content.append({
                    'type': 'reasoning',
                    'text': reasoning_text
                })

        elif part_type == 'tool-call':
            # Already in ModelMessage format
            assistant_content.append({
                'type': 'tool-call',
                'toolCallId': part.get('toolCallId'),
                'toolName': part.get('toolName'),
                'input': part.get('input', {})
            })

        elif _is_tool_execution(part_type):
            # Tool execution with both input and output (UIMessage format)
            # Split into tool-call and tool-result
            tool_call_id = part.get('toolCallId')
            tool_name = _extract_tool_name(part_type)
            tool_input = part.get('input', {})
            tool_output = part.get('output')

            if tool_call_id and tool_name:
                # Add tool call
                assistant_content.append({
                    'type': 'tool-call',
                    'toolCallId': tool_call_id,
                    'toolName': tool_name,
                    'input': tool_input
                })

                # Add tool result if output is available
                if tool_output is not None and part.get('state') == 'output-available':
                    tool_results.append({
                        'type': 'tool-result',
                        'toolCallId': tool_call_id,
                        'toolName': tool_name,
                        'output': tool_output
                    })

        else:
            logger.warning(f"Unknown assistant content part type: {part_type}")

    # Build messages
    messages = []

    if assistant_content:
        # If only text and no other content, use string content
        if len(assistant_content) == 1 and assistant_content[0].get('type') == 'text':
            messages.append({
                'role': 'assistant',
                'content': assistant_content[0]['text']
            })
        else:
            messages.append({
                'role': 'assistant',
                'content': assistant_content
            })

    # Tool results as separate tool message
    if tool_results:
        messages.append({
            'role': 'tool',
            'content': tool_results
        })

    return messages


def _process_tool_message(parts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Process tool message parts.

    Tool messages contain tool results.
    """
    if not parts:
        return []

    tool_results = []

    for part in parts:
        part_type = part.get('type', 'tool-result')

        if part_type == 'tool-result':
            tool_results.append({
                'type': 'tool-result',
                'toolCallId': part.get('toolCallId'),
                'toolName': part.get('toolName'),
                'output': part.get('output')
            })

    if tool_results:
        return [{'role': 'tool', 'content': tool_results}]

    return []


def _is_tool_execution(part_type: str) -> bool:
    """
    Check if part type is a tool execution.

    UIMessage uses types like:
    - tool-websearch
    - tool-news
    - tool-provideAnswer
    - tool-calculator
    etc.
    """
    return part_type.startswith('tool-') and part_type not in {'tool-call', 'tool-result'}


def _extract_tool_name(part_type: str) -> str:
    """
    Extract tool name from part type.

    'tool-websearch' → 'tool-websearch'
    'tool-news' → 'tool-news'

    Keep the full type as the tool name since that's what the agent expects.
    """
    return part_type


def convert_from_legacy_format(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Convert legacy message format to UIMessage format.

    Legacy format uses 'content' field instead of 'parts'.
    This helper converts it to UIMessage format so convert_to_model_messages() can process it.

    Args:
        messages: List of messages with 'content' field

    Returns:
        List of messages with 'parts' field
    """
    converted = []

    for msg in messages:
        if 'parts' in msg:
            # Already in UIMessage format
            converted.append(msg)
            continue

        role = msg.get('role')
        content = msg.get('content')

        if isinstance(content, str):
            # String content → text part
            converted.append({
                'role': role,
                'parts': [{'type': 'text', 'text': content}]
            })
        elif isinstance(content, list):
            # Array content → use as parts
            converted.append({
                'role': role,
                'parts': content
            })
        else:
            # Unknown format, keep as-is
            converted.append(msg)

    return converted
