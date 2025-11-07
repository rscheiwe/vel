"""Anthropic Claude provider with stream protocol support"""
from __future__ import annotations
import os, httpx, json
from typing import Any, AsyncGenerator, Dict, List, Optional
from .base import BaseProvider, LLMMessage
from .translators import AnthropicAPITranslator
from .message_translator import translate_to_anthropic, MessageTranslationError
from ..events import StreamEvent, ErrorEvent
import logging

logger = logging.getLogger('vel.providers.anthropic')

class AnthropicProvider(BaseProvider):
    """Anthropic Claude provider implementing stream protocol"""
    name = 'anthropic'

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Anthropic provider.

        Args:
            api_key: Optional API key. If not provided, falls back to ANTHROPIC_API_KEY environment variable.
        """
        self.base = os.getenv('ANTHROPIC_API_BASE', 'https://api.anthropic.com')
        self.translator = AnthropicAPITranslator()

        # Use provided API key or fall back to environment variable
        self.api_key = api_key or os.getenv('ANTHROPIC_API_KEY', '')
        if not self.api_key:
            raise ValueError("Anthropic API key not provided. Either pass api_key parameter or set ANTHROPIC_API_KEY environment variable.")

    def _headers(self):
        """Get headers with API key"""
        return {
            'x-api-key': self.api_key,
            'anthropic-version': '2023-06-01',
            'Content-Type': 'application/json'
        }

    def _convert_messages(self, messages: List[LLMMessage]) -> tuple[str, List[Dict[str, Any]]]:
        """Convert messages to Anthropic format, extracting system message"""
        system_message = ""
        anthropic_messages = []

        for msg in messages:
            role = msg.get('role', 'user')
            content = msg.get('content', '')

            if role == 'system':
                system_message = content
            else:
                anthropic_messages.append({
                    'role': role,
                    'content': content
                })

        return system_message, anthropic_messages

    def _convert_tools(self, tools: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Convert tool schemas to Anthropic tool format"""
        if not tools:
            return []

        anthropic_tools = []
        for name, schema in tools.items():
            anthropic_tools.append({
                'name': name,
                'description': schema.get('description', f'Tool: {name}'),
                'input_schema': schema['input']
            })
        return anthropic_tools

    async def stream(
        self,
        messages: List[LLMMessage],
        model: str,
        tools: Dict[str, Any],
        generation_config: Optional[Dict[str, Any]] = None
    ) -> AsyncGenerator[StreamEvent, None]:
        """Stream Anthropic response as stream protocol events"""
        # Reset translator state
        self.translator.reset()

        # Translate ModelMessage format to Anthropic format
        try:
            system_message, anthropic_messages = translate_to_anthropic(messages)
            logger.debug(f"Translated {len(messages)} messages to Anthropic format: {len(anthropic_messages)} messages")
        except MessageTranslationError as e:
            logger.error(f"Message translation failed: {e}")
            yield ErrorEvent(error=str(e))
            return

        anthropic_tools = self._convert_tools(tools)

        # Start with default max_tokens
        payload = {
            'model': model,
            'messages': anthropic_messages,
            'max_tokens': 4096,  # Default, can be overridden
            'stream': True
        }

        if system_message:
            payload['system'] = system_message

        if anthropic_tools:
            payload['tools'] = anthropic_tools

        # Add generation config parameters
        config = generation_config or {}
        if 'temperature' in config:
            payload['temperature'] = config['temperature']
        if 'max_tokens' in config:
            payload['max_tokens'] = config['max_tokens']
        if 'top_p' in config:
            payload['top_p'] = config['top_p']
        if 'top_k' in config:
            payload['top_k'] = config['top_k']
        if 'stop_sequences' in config:
            payload['stop_sequences'] = config['stop_sequences']
        if 'stop' in config:  # Alias for stop_sequences
            payload['stop_sequences'] = config['stop']

        # Enable extended thinking (for reasoning models)
        # Can be: {'type': 'enabled', 'budget_tokens': 10000} or {'type': 'disabled'}
        if 'thinking' in config:
            payload['thinking'] = config['thinking']

        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                async with client.stream(
                    'POST',
                    f"{self.base}/v1/messages",
                    headers=self._headers(),
                    json=payload
                ) as response:
                    response.raise_for_status()

                    async for line in response.aiter_lines():
                        if not line.strip():
                            continue

                        # Parse SSE format: "data: <json>"
                        if line.startswith('data: '):
                            data_str = line[6:].strip()
                            try:
                                data = json.loads(data_str)
                            except json.JSONDecodeError:
                                continue

                            # Translate event to Vel format
                            vel_event = self.translator.translate_event(data)
                            if vel_event:
                                yield vel_event

                                # Check if this is a stop event
                                if vel_event.type == 'finish-message':
                                    return
                                elif vel_event.type == 'error':
                                    return

        except Exception as e:
            yield self._create_error_event(e)

    def _create_error_event(self, exception: Exception) -> ErrorEvent:
        """Create detailed ErrorEvent from exception

        Extracts HTTP status codes, error codes, and provider-specific error details.
        """
        import httpx

        # Handle httpx HTTP errors
        if isinstance(exception, httpx.HTTPStatusError):
            response = exception.response
            status_code = response.status_code

            # Try to parse error body as JSON
            error_details = {}
            error_message = str(exception)
            error_code = None
            error_type = None

            try:
                error_body = response.json()
                if 'error' in error_body:
                    error_data = error_body['error']
                    error_message = error_data.get('message', str(exception))
                    error_code = error_data.get('code')
                    error_type = error_data.get('type')
                    error_details = error_data
            except Exception:
                # If JSON parsing fails, use response text
                error_message = f"HTTP {status_code}: {response.text[:200]}"

            return ErrorEvent(
                error=error_message,
                error_code=error_code,
                error_type=error_type,
                status_code=status_code,
                provider='anthropic',
                details=error_details if error_details else None
            )

        # Handle other httpx errors
        elif isinstance(exception, httpx.RequestError):
            return ErrorEvent(
                error=f"Request failed: {str(exception)}",
                error_type='request_error',
                provider='anthropic'
            )

        # Generic exception handling
        return ErrorEvent(
            error=str(exception),
            error_type=type(exception).__name__,
            provider='anthropic'
        )

    async def generate(
        self,
        messages: List[LLMMessage],
        model: str,
        tools: Dict[str, Any],
        generation_config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Non-streaming generation"""
        # Translate ModelMessage format to Anthropic format
        try:
            system_message, anthropic_messages = translate_to_anthropic(messages)
            logger.debug(f"Translated {len(messages)} messages to Anthropic format in generate()")
        except MessageTranslationError as e:
            logger.error(f"Message translation failed in generate(): {e}")
            raise ValueError(f"Failed to translate messages to Anthropic format: {e}") from e

        anthropic_tools = self._convert_tools(tools)

        payload = {
            'model': model,
            'messages': anthropic_messages,
            'max_tokens': 4096  # Default, can be overridden
        }

        if system_message:
            payload['system'] = system_message

        if anthropic_tools:
            payload['tools'] = anthropic_tools

        # Add generation config parameters
        config = generation_config or {}
        if 'temperature' in config:
            payload['temperature'] = config['temperature']
        if 'max_tokens' in config:
            payload['max_tokens'] = config['max_tokens']
        if 'top_p' in config:
            payload['top_p'] = config['top_p']
        if 'top_k' in config:
            payload['top_k'] = config['top_k']
        if 'stop_sequences' in config:
            payload['stop_sequences'] = config['stop_sequences']
        if 'stop' in config:  # Alias for stop_sequences
            payload['stop_sequences'] = config['stop']

        # Enable extended thinking (for reasoning models)
        # Can be: {'type': 'enabled', 'budget_tokens': 10000} or {'type': 'disabled'}
        if 'thinking' in config:
            payload['thinking'] = config['thinking']

        async with httpx.AsyncClient(timeout=30.0) as client:
            r = await client.post(
                f"{self.base}/v1/messages",
                headers=self._headers(),
                json=payload
            )
            r.raise_for_status()
            data = r.json()

        # Parse response
        content = data.get('content', [])
        usage_data = data.get('usage', {})
        usage = {
            'input_tokens': usage_data.get('input_tokens', 0),
            'output_tokens': usage_data.get('output_tokens', 0)
        }

        # Check for tool use
        for block in content:
            if block.get('type') == 'tool_use':
                return {
                    'tool': block.get('name'),
                    'args': block.get('input', {}),
                    'usage': usage
                }

        # Extract text content
        text_content = ''.join([
            block.get('text', '')
            for block in content
            if block.get('type') == 'text'
        ])

        return {'done': True, 'answer': text_content, 'usage': usage}
