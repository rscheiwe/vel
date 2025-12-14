"""
Message translation layer for Vel.

Converts Vercel AI SDK ModelMessage format to provider-specific formats.

ModelMessage is the unified format that comes from `convertToModelMessages()` in the
Vercel AI SDK. Each LLM provider (OpenAI, Anthropic, Gemini) expects a different
message structure, so we translate here.

Supported content types:
- Text
- Reasoning (OpenAI o1/o3, Anthropic extended thinking)
- Images (base64 or URL)
- Files (PDF, documents)
- Tool calls
- Tool results

Based on Vercel AI SDK implementation:
- packages/openai/src/chat/convert-to-openai-chat-messages.ts
- packages/anthropic/src/convert-to-anthropic-messages-prompt.ts
- packages/google/src/convert-to-google-generative-ai-messages.ts
"""
from typing import List, Dict, Any, Optional, Tuple
import json
import base64
import logging

logger = logging.getLogger('vel.message_translator')


class MessageTranslationError(Exception):
    """Raised when message translation fails."""
    def __init__(self, message: str, provider: str, original_message: Optional[Dict] = None):
        self.provider = provider
        self.original_message = original_message
        super().__init__(f"[{provider}] {message}")


def _is_string_content(content: Any) -> bool:
    """Check if content is a simple string."""
    return isinstance(content, str)


def _normalize_content(content: Any) -> List[Dict[str, Any]]:
    """
    Normalize content to array of parts format.

    Handles both:
    - content: "string"
    - content: [{type: 'text', text: '...'}, ...]
    """
    if _is_string_content(content):
        return [{'type': 'text', 'text': content}]
    elif isinstance(content, list):
        return content
    else:
        raise ValueError(f"Invalid content format: {type(content)}")


# =============================================================================
# OpenAI Message Translation
# =============================================================================

def translate_to_openai(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Convert ModelMessage format to OpenAI Chat Completions format.

    ModelMessage format:
    {
      role: 'assistant',
      content: [
        {type: 'text', text: '...'},
        {type: 'tool-call', toolCallId: '...', toolName: '...', input: {...}}
      ]
    }

    OpenAI format:
    {
      role: 'assistant',
      content: '...',
      tool_calls: [
        {id: '...', type: 'function', function: {name: '...', arguments: '{...}'}}
      ]
    }

    Args:
        messages: List of ModelMessage objects

    Returns:
        List of OpenAI-formatted messages

    Raises:
        MessageTranslationError: If translation fails
    """
    try:
        openai_messages = []

        for idx, msg in enumerate(messages):
            role = msg.get('role')
            content = msg.get('content', '')

            if not role:
                raise MessageTranslationError(
                    f"Message at index {idx} missing 'role' field",
                    'openai',
                    msg
                )

            # System message - simple passthrough
            if role == 'system':
                if not _is_string_content(content):
                    raise MessageTranslationError(
                        f"System message at index {idx} must have string content",
                        'openai',
                        msg
                    )
                openai_messages.append({
                    'role': 'system',
                    'content': content
                })
                continue

            # User message - can be string or multimodal
            if role == 'user':
                parts = _normalize_content(content)

                # Simple text-only message
                if len(parts) == 1 and parts[0].get('type') == 'text':
                    openai_messages.append({
                        'role': 'user',
                        'content': parts[0].get('text', '')
                    })
                else:
                    # Multimodal message (text + images/files)
                    openai_content = []
                    for part in parts:
                        part_type = part.get('type')

                        if part_type == 'text':
                            openai_content.append({
                                'type': 'text',
                                'text': part.get('text', '')
                            })
                        elif part_type == 'image':
                            # Image can be URL or base64
                            image_data = part.get('image', part.get('data'))
                            if isinstance(image_data, str) and image_data.startswith('http'):
                                openai_content.append({
                                    'type': 'image_url',
                                    'image_url': {'url': image_data}
                                })
                            else:
                                # Base64 encoded
                                mime_type = part.get('mimeType', 'image/png')
                                openai_content.append({
                                    'type': 'image_url',
                                    'image_url': {
                                        'url': f"data:{mime_type};base64,{image_data}"
                                    }
                                })
                        elif part_type == 'file':
                            # AI SDK v5: FilePart can contain images (type: 'file', mediaType: 'image/*')
                            # Check mediaType to determine if this is an image file
                            mime_type = part.get('mediaType') or part.get('mimeType', '')

                            if mime_type.startswith('image/'):
                                # Image file - treat as image for OpenAI
                                file_data = part.get('data', '')

                                # Handle different data formats
                                if file_data.startswith('data:'):
                                    # Already a data URL
                                    openai_content.append({
                                        'type': 'image_url',
                                        'image_url': {'url': file_data}
                                    })
                                elif file_data.startswith('http'):
                                    # HTTP(S) URL
                                    openai_content.append({
                                        'type': 'image_url',
                                        'image_url': {'url': file_data}
                                    })
                                else:
                                    # Raw base64 - construct data URL
                                    openai_content.append({
                                        'type': 'image_url',
                                        'image_url': {
                                            'url': f"data:{mime_type};base64,{file_data}"
                                        }
                                    })
                            else:
                                # Non-image file - OpenAI doesn't support generic files
                                logger.warning(
                                    f"OpenAI provider does not support file type '{mime_type}' "
                                    f"in chat messages. Skipping file part."
                                )
                        else:
                            logger.warning(
                                f"Unknown user content part type '{part_type}' at message index {idx}. "
                                f"Skipping."
                            )

                    openai_messages.append({
                        'role': 'user',
                        'content': openai_content
                    })
                continue

            # Assistant message - can have text and/or tool calls
            if role == 'assistant':
                # Check if message is already in OpenAI format (has tool_calls at top level)
                if 'tool_calls' in msg:
                    # Already in OpenAI format - passthrough
                    assistant_msg = {'role': 'assistant'}
                    if content is not None:
                        assistant_msg['content'] = content if isinstance(content, str) else ''
                    else:
                        assistant_msg['content'] = ''
                    assistant_msg['tool_calls'] = msg['tool_calls']
                    openai_messages.append(assistant_msg)
                    continue

                # Handle None content (no text, only tool calls in original format)
                if content is None:
                    content = []

                parts = _normalize_content(content)

                text_parts = []
                tool_calls = []

                for part in parts:
                    part_type = part.get('type')

                    if part_type == 'text':
                        text_parts.append(part.get('text', ''))
                    elif part_type == 'reasoning':
                        # Reasoning content (OpenAI o1/o3, Anthropic thinking)
                        # Include in text content
                        text_parts.append(part.get('text', ''))
                    elif part_type == 'tool-call':
                        tool_call_id = part.get('toolCallId')
                        tool_name = part.get('toolName')
                        tool_input = part.get('input', {})

                        if not tool_call_id or not tool_name:
                            raise MessageTranslationError(
                                f"Tool call at message index {idx} missing toolCallId or toolName",
                                'openai',
                                part
                            )

                        tool_calls.append({
                            'id': tool_call_id,
                            'type': 'function',
                            'function': {
                                'name': tool_name,
                                'arguments': json.dumps(tool_input)
                            }
                        })
                    else:
                        logger.warning(
                            f"Unknown assistant content part type '{part_type}' at message index {idx}"
                        )

                # Build OpenAI assistant message
                assistant_msg = {'role': 'assistant'}

                if text_parts:
                    assistant_msg['content'] = ' '.join(text_parts)
                else:
                    # OpenAI requires content field even if empty when there are tool calls
                    assistant_msg['content'] = ''

                if tool_calls:
                    assistant_msg['tool_calls'] = tool_calls

                openai_messages.append(assistant_msg)
                continue

            # Tool message - tool execution results
            if role == 'tool':
                # Check if message is already in OpenAI format (has tool_call_id at top level)
                if 'tool_call_id' in msg:
                    # Already in OpenAI format - passthrough
                    openai_messages.append({
                        'role': 'tool',
                        'tool_call_id': msg['tool_call_id'],
                        'content': content if isinstance(content, str) else str(content)
                    })
                    continue

                # Handle None content
                if content is None:
                    content = []

                parts = _normalize_content(content)

                for part in parts:
                    part_type = part.get('type')

                    if part_type == 'tool-result':
                        tool_call_id = part.get('toolCallId')
                        tool_name = part.get('toolName')
                        tool_output = part.get('output')

                        if not tool_call_id:
                            raise MessageTranslationError(
                                f"Tool result at message index {idx} missing toolCallId",
                                'openai',
                                part
                            )

                        # Convert output to string
                        if isinstance(tool_output, str):
                            content_str = tool_output
                        elif isinstance(tool_output, dict):
                            content_str = json.dumps(tool_output)
                        else:
                            content_str = str(tool_output)

                        openai_messages.append({
                            'role': 'tool',
                            'tool_call_id': tool_call_id,
                            'content': content_str
                        })
                    else:
                        logger.warning(
                            f"Unknown tool content part type '{part_type}' at message index {idx}"
                        )
                continue

            # Unknown role
            raise MessageTranslationError(
                f"Unknown message role '{role}' at index {idx}",
                'openai',
                msg
            )

        return openai_messages

    except MessageTranslationError:
        raise
    except Exception as e:
        raise MessageTranslationError(
            f"Unexpected error during translation: {str(e)}",
            'openai'
        ) from e


# =============================================================================
# Anthropic Message Translation
# =============================================================================

def translate_to_anthropic(messages: List[Dict[str, Any]]) -> Tuple[Optional[str], List[Dict[str, Any]]]:
    """
    Convert ModelMessage format to Anthropic Messages format.

    Anthropic format:
    - System messages are extracted and returned separately
    - Tool calls use 'tool_use' type with id, name, input
    - Tool results use 'tool_result' type with tool_use_id, content

    Args:
        messages: List of ModelMessage objects

    Returns:
        Tuple of (system_prompt, anthropic_messages)

    Raises:
        MessageTranslationError: If translation fails
    """
    try:
        system_parts = []
        anthropic_messages = []

        for idx, msg in enumerate(messages):
            role = msg.get('role')
            content = msg.get('content', '')

            if not role:
                raise MessageTranslationError(
                    f"Message at index {idx} missing 'role' field",
                    'anthropic',
                    msg
                )

            # System message - extract to separate system parameter
            if role == 'system':
                if not _is_string_content(content):
                    raise MessageTranslationError(
                        f"System message at index {idx} must have string content",
                        'anthropic',
                        msg
                    )
                system_parts.append(content)
                continue

            # User message
            if role == 'user':
                parts = _normalize_content(content)

                anthropic_content = []
                for part in parts:
                    part_type = part.get('type')

                    if part_type == 'text':
                        anthropic_content.append({
                            'type': 'text',
                            'text': part.get('text', '')
                        })
                    elif part_type == 'image':
                        # Anthropic expects base64 images
                        image_data = part.get('image', part.get('data'))
                        mime_type = part.get('mimeType', 'image/png')

                        # If URL, log warning (Anthropic doesn't support URLs)
                        if isinstance(image_data, str) and image_data.startswith('http'):
                            logger.warning(
                                f"Anthropic does not support image URLs. "
                                f"Please convert to base64. Skipping image at message index {idx}."
                            )
                        else:
                            anthropic_content.append({
                                'type': 'image',
                                'source': {
                                    'type': 'base64',
                                    'media_type': mime_type,
                                    'data': image_data
                                }
                            })
                    elif part_type == 'file':
                        # AI SDK v5: FilePart can contain images or documents
                        mime_type = part.get('mediaType') or part.get('mimeType', '')
                        file_data = part.get('data', '')

                        if mime_type.startswith('image/'):
                            # Image file - treat as image
                            # Remove data URL prefix if present
                            if file_data.startswith('data:'):
                                file_data = file_data.split(',', 1)[1] if ',' in file_data else file_data

                            # Anthropic doesn't support URLs
                            if file_data.startswith('http'):
                                logger.warning(
                                    f"Anthropic does not support image URLs. "
                                    f"Please convert to base64. Skipping image at message index {idx}."
                                )
                            else:
                                anthropic_content.append({
                                    'type': 'image',
                                    'source': {
                                        'type': 'base64',
                                        'media_type': mime_type,
                                        'data': file_data
                                    }
                                })
                        elif mime_type == 'application/pdf':
                            # PDF document
                            anthropic_content.append({
                                'type': 'document',
                                'source': {
                                    'type': 'base64',
                                    'media_type': mime_type,
                                    'data': file_data
                                }
                            })
                        else:
                            logger.warning(
                                f"Anthropic only supports images and PDF documents. "
                                f"Received '{mime_type}'. Skipping file at message index {idx}."
                            )
                    else:
                        logger.warning(
                            f"Unknown user content part type '{part_type}' at message index {idx}"
                        )

                anthropic_messages.append({
                    'role': 'user',
                    'content': anthropic_content
                })
                continue

            # Assistant message
            if role == 'assistant':
                parts = _normalize_content(content)

                anthropic_content = []

                for part in parts:
                    part_type = part.get('type')

                    if part_type == 'text':
                        anthropic_content.append({
                            'type': 'text',
                            'text': part.get('text', '')
                        })
                    elif part_type == 'reasoning':
                        # Reasoning content (thinking blocks)
                        # Anthropic uses 'thinking' type for extended thinking
                        anthropic_content.append({
                            'type': 'text',
                            'text': part.get('text', '')
                        })
                    elif part_type == 'tool-call':
                        tool_call_id = part.get('toolCallId')
                        tool_name = part.get('toolName')
                        tool_input = part.get('input', {})

                        if not tool_call_id or not tool_name:
                            raise MessageTranslationError(
                                f"Tool call at message index {idx} missing toolCallId or toolName",
                                'anthropic',
                                part
                            )

                        anthropic_content.append({
                            'type': 'tool_use',
                            'id': tool_call_id,
                            'name': tool_name,
                            'input': tool_input
                        })
                    else:
                        logger.warning(
                            f"Unknown assistant content part type '{part_type}' at message index {idx}"
                        )

                anthropic_messages.append({
                    'role': 'assistant',
                    'content': anthropic_content
                })
                continue

            # Tool message - convert to user message with tool_result
            if role == 'tool':
                parts = _normalize_content(content)

                tool_results = []

                for part in parts:
                    part_type = part.get('type')

                    if part_type == 'tool-result':
                        tool_call_id = part.get('toolCallId')
                        tool_output = part.get('output')

                        if not tool_call_id:
                            raise MessageTranslationError(
                                f"Tool result at message index {idx} missing toolCallId",
                                'anthropic',
                                part
                            )

                        # Convert output to string or structured content
                        if isinstance(tool_output, str):
                            result_content = tool_output
                        elif isinstance(tool_output, dict):
                            result_content = json.dumps(tool_output)
                        else:
                            result_content = str(tool_output)

                        tool_results.append({
                            'type': 'tool_result',
                            'tool_use_id': tool_call_id,
                            'content': result_content
                        })
                    else:
                        logger.warning(
                            f"Unknown tool content part type '{part_type}' at message index {idx}"
                        )

                # Anthropic requires tool results to be in a user message
                anthropic_messages.append({
                    'role': 'user',
                    'content': tool_results
                })
                continue

            # Unknown role
            raise MessageTranslationError(
                f"Unknown message role '{role}' at index {idx}",
                'anthropic',
                msg
            )

        # Combine system messages
        system_prompt = '\n\n'.join(system_parts) if system_parts else None

        return system_prompt, anthropic_messages

    except MessageTranslationError:
        raise
    except Exception as e:
        raise MessageTranslationError(
            f"Unexpected error during translation: {str(e)}",
            'anthropic'
        ) from e


# =============================================================================
# Gemini Message Translation
# =============================================================================

def translate_to_gemini(messages: List[Dict[str, Any]]) -> Tuple[Optional[Dict], List[Dict[str, Any]]]:
    """
    Convert ModelMessage format to Google Gemini format.

    Gemini format:
    - System messages become systemInstruction
    - Uses 'user' and 'model' roles (not 'assistant')
    - Tool calls use functionCall with name and args
    - Tool results use functionResponse

    Args:
        messages: List of ModelMessage objects

    Returns:
        Tuple of (system_instruction, gemini_messages)

    Raises:
        MessageTranslationError: If translation fails
    """
    try:
        system_parts = []
        gemini_messages = []

        for idx, msg in enumerate(messages):
            role = msg.get('role')
            content = msg.get('content', '')

            if not role:
                raise MessageTranslationError(
                    f"Message at index {idx} missing 'role' field",
                    'gemini',
                    msg
                )

            # System message - extract to systemInstruction
            if role == 'system':
                if not _is_string_content(content):
                    raise MessageTranslationError(
                        f"System message at index {idx} must have string content",
                        'gemini',
                        msg
                    )
                system_parts.append({'text': content})
                continue

            # User message
            if role == 'user':
                parts = _normalize_content(content)

                gemini_parts = []
                for part in parts:
                    part_type = part.get('type')

                    if part_type == 'text':
                        gemini_parts.append({
                            'text': part.get('text', '')
                        })
                    elif part_type == 'image':
                        # Gemini supports inline images
                        image_data = part.get('image', part.get('data'))
                        mime_type = part.get('mimeType', 'image/png')

                        if isinstance(image_data, str) and image_data.startswith('http'):
                            # URL-based image (not supported in inline_data, would need fileData)
                            logger.warning(
                                f"Gemini inline images require base64 encoding. "
                                f"Use File API for URL-based images. Skipping image at message index {idx}."
                            )
                        else:
                            gemini_parts.append({
                                'inline_data': {
                                    'mime_type': mime_type,
                                    'data': image_data
                                }
                            })
                    elif part_type == 'file':
                        # AI SDK v5: FilePart can contain images or other files
                        mime_type = part.get('mediaType') or part.get('mimeType', '')
                        file_data = part.get('data', '')

                        if mime_type.startswith('image/'):
                            # Image file - treat as inline image
                            # Remove data URL prefix if present
                            if file_data.startswith('data:'):
                                file_data = file_data.split(',', 1)[1] if ',' in file_data else file_data

                            if file_data.startswith('http'):
                                logger.warning(
                                    f"Gemini inline images require base64 encoding. "
                                    f"Use File API for URL-based images. Skipping image at message index {idx}."
                                )
                            else:
                                gemini_parts.append({
                                    'inline_data': {
                                        'mime_type': mime_type,
                                        'data': file_data
                                    }
                                })
                        else:
                            # Non-image file - requires File API with fileUri
                            file_uri = part.get('fileUri', part.get('url'))
                            if file_uri:
                                gemini_parts.append({
                                    'file_data': {
                                        'mime_type': mime_type,
                                        'file_uri': file_uri
                                    }
                                })
                            else:
                                logger.warning(
                                    f"Gemini non-image file upload requires fileUri. "
                                    f"Skipping file at message index {idx}."
                                )
                    else:
                        logger.warning(
                            f"Unknown user content part type '{part_type}' at message index {idx}"
                        )

                gemini_messages.append({
                    'role': 'user',
                    'parts': gemini_parts
                })
                continue

            # Assistant message (converted to 'model' role)
            if role == 'assistant':
                parts = _normalize_content(content)

                gemini_parts = []

                for part in parts:
                    part_type = part.get('type')

                    if part_type == 'text':
                        gemini_parts.append({
                            'text': part.get('text', '')
                        })
                    elif part_type == 'reasoning':
                        # Reasoning content - include as text
                        gemini_parts.append({
                            'text': part.get('text', '')
                        })
                    elif part_type == 'tool-call':
                        tool_name = part.get('toolName')
                        tool_input = part.get('input', {})

                        if not tool_name:
                            raise MessageTranslationError(
                                f"Tool call at message index {idx} missing toolName",
                                'gemini',
                                part
                            )

                        gemini_parts.append({
                            'function_call': {
                                'name': tool_name,
                                'args': tool_input
                            }
                        })
                    else:
                        logger.warning(
                            f"Unknown assistant content part type '{part_type}' at message index {idx}"
                        )

                gemini_messages.append({
                    'role': 'model',  # Gemini uses 'model' instead of 'assistant'
                    'parts': gemini_parts
                })
                continue

            # Tool message - convert to user message with functionResponse
            if role == 'tool':
                parts = _normalize_content(content)

                function_responses = []

                for part in parts:
                    part_type = part.get('type')

                    if part_type == 'tool-result':
                        tool_name = part.get('toolName')
                        tool_output = part.get('output')

                        if not tool_name:
                            raise MessageTranslationError(
                                f"Tool result at message index {idx} missing toolName",
                                'gemini',
                                part
                            )

                        # Gemini expects structured response
                        if not isinstance(tool_output, dict):
                            if isinstance(tool_output, str):
                                try:
                                    tool_output = json.loads(tool_output)
                                except json.JSONDecodeError:
                                    tool_output = {'result': tool_output}
                            else:
                                tool_output = {'result': str(tool_output)}

                        function_responses.append({
                            'function_response': {
                                'name': tool_name,
                                'response': tool_output
                            }
                        })
                    else:
                        logger.warning(
                            f"Unknown tool content part type '{part_type}' at message index {idx}"
                        )

                gemini_messages.append({
                    'role': 'user',
                    'parts': function_responses
                })
                continue

            # Unknown role
            raise MessageTranslationError(
                f"Unknown message role '{role}' at index {idx}",
                'gemini',
                msg
            )

        # Build systemInstruction if present
        system_instruction = None
        if system_parts:
            system_instruction = {'parts': system_parts}

        return system_instruction, gemini_messages

    except MessageTranslationError:
        raise
    except Exception as e:
        raise MessageTranslationError(
            f"Unexpected error during translation: {str(e)}",
            'gemini'
        ) from e
