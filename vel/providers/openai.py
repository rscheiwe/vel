"""OpenAI provider with stream protocol support"""
from __future__ import annotations
import os, httpx, json
from typing import Any, AsyncGenerator, Dict, List, Optional
from .base import BaseProvider, LLMMessage
from .translators import OpenAIAPITranslator, OpenAIResponsesAPITranslator
from .message_translator import translate_to_openai, MessageTranslationError
from ..events import StreamEvent, FinishMessageEvent, ErrorEvent
import logging

logger = logging.getLogger('vel.providers.openai')

class OpenAIProvider(BaseProvider):
    """OpenAI provider implementing stream protocol"""
    name = 'openai'

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize OpenAI provider.

        Args:
            api_key: Optional API key. If not provided, falls back to OPENAI_API_KEY environment variable.
        """
        self.base = os.getenv('OPENAI_API_BASE', 'https://api.openai.com/v1')
        self.translator = OpenAIAPITranslator()

        # Use provided API key or fall back to environment variable
        self.api_key = api_key or os.getenv('OPENAI_API_KEY', '')
        if not self.api_key:
            raise ValueError("OpenAI API key not provided. Either pass api_key parameter or set OPENAI_API_KEY environment variable.")

    def _headers(self):
        """Get headers with API key"""
        return {
            'Authorization': f"Bearer {self.api_key}",
            'Content-Type': 'application/json'
        }

    async def stream(
        self,
        messages: List[LLMMessage],
        model: str,
        tools: Dict[str, Any],
        generation_config: Optional[Dict[str, Any]] = None
    ) -> AsyncGenerator[StreamEvent, None]:
        """Stream OpenAI response as stream protocol events"""
        # Reset translator state
        self.translator.reset()

        # Translate ModelMessage format to OpenAI format
        try:
            msgs = translate_to_openai(messages)
            logger.debug(f"Translated {len(messages)} messages to OpenAI format: {len(msgs)} messages")
        except MessageTranslationError as e:
            logger.error(f"Message translation failed: {e}")
            yield ErrorEvent(error=str(e))
            return

        oaitools = [
            {'type': 'function', 'function': {'name': n, 'description': s.get('description', f'Tool: {n}'), 'parameters': s['input']}}
            for n, s in tools.items()
        ] if tools else []

        # Add generation config parameters
        config = generation_config or {}

        payload = {
            'model': model,
            'messages': msgs,
            'stream': True,
            'stream_options': {'include_usage': True}  # AI SDK v5 parity: include usage in stream
        }
        if oaitools:
            payload['tools'] = oaitools
            payload['tool_choice'] = 'auto'
            # Disable parallel tool calls if specified (prevents duplicate calls in same response)
            if config.get('parallel_tool_calls') is False:
                payload['parallel_tool_calls'] = False

        if 'temperature' in config:
            payload['temperature'] = config['temperature']
        if 'max_tokens' in config:
            payload['max_tokens'] = config['max_tokens']
        if 'top_p' in config:
            payload['top_p'] = config['top_p']
        if 'presence_penalty' in config:
            payload['presence_penalty'] = config['presence_penalty']
        if 'frequency_penalty' in config:
            payload['frequency_penalty'] = config['frequency_penalty']
        if 'stop' in config:
            payload['stop'] = config['stop']
        if 'seed' in config:
            payload['seed'] = config['seed']
        if 'logit_bias' in config:
            payload['logit_bias'] = config['logit_bias']
        if 'user' in config:
            payload['user'] = config['user']

        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                async with client.stream(
                    'POST',
                    f"{self.base}/chat/completions",
                    headers=self._headers(),
                    json=payload
                ) as response:
                    response.raise_for_status()

                    finish_reason = None

                    async for line in response.aiter_lines():
                        if not line.strip() or line.strip() == 'data: [DONE]':
                            continue
                        if line.startswith('data: '):
                            data_str = line[6:]
                            try:
                                chunk = json.loads(data_str)
                            except json.JSONDecodeError:
                                continue

                            # Translate chunk to Vel event(s)
                            vel_event = self.translator.translate_chunk(chunk)
                            if vel_event:
                                yield vel_event

                            # Drain any pending events from the translator
                            while True:
                                pending_event = self.translator.get_pending_event()
                                if pending_event is None:
                                    break
                                yield pending_event

                            # Track finish_reason but don't return yet (usage comes in next chunk)
                            choices = chunk.get('choices', [])
                            if choices and choices[0].get('finish_reason'):
                                finish_reason = choices[0]['finish_reason']

                    # Stream ended - emit finish-message and finalize
                    for tool_event in self.translator.finalize_tool_calls():
                        yield tool_event
                    yield FinishMessageEvent(finish_reason=finish_reason if finish_reason else 'stop')

        except Exception as e:
            yield self._create_error_event(e, 'openai')

    def _create_error_event(self, exception: Exception, provider: str) -> ErrorEvent:
        """Create detailed ErrorEvent from exception

        Extracts HTTP status codes, error codes, and provider-specific error details.
        """
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
                # For streaming responses, body may not have been read yet
                # Check if response content is available before accessing
                if hasattr(response, 'is_stream_consumed') and not response.is_stream_consumed:
                    # Streaming response - body not available
                    error_message = f"OpenAI API error: HTTP {status_code}"
                elif hasattr(response, '_content') and response._content:
                    # Non-streaming response with content available
                    error_body = response.json()
                    if 'error' in error_body:
                        error_data = error_body['error']
                        error_message = error_data.get('message', str(exception))
                        error_code = error_data.get('code')
                        error_type = error_data.get('type')
                        error_details = error_data
                else:
                    # Fallback - try to read but catch if not available
                    error_body = response.json()
                    if 'error' in error_body:
                        error_data = error_body['error']
                        error_message = error_data.get('message', str(exception))
                        error_code = error_data.get('code')
                        error_type = error_data.get('type')
                        error_details = error_data
            except (httpx.ResponseNotRead, Exception):
                # If response body not available or JSON parsing fails
                error_message = f"OpenAI API error: HTTP {status_code}"

            return ErrorEvent(
                error=error_message,
                error_code=error_code,
                error_type=error_type,
                status_code=status_code,
                provider=provider,
                details=error_details if error_details else None
            )

        # Handle other httpx errors
        elif isinstance(exception, httpx.RequestError):
            return ErrorEvent(
                error=f"Request failed: {str(exception)}",
                error_type='request_error',
                provider=provider
            )

        # Generic exception handling
        return ErrorEvent(
            error=str(exception),
            error_type=type(exception).__name__,
            provider=provider
        )

    async def generate(
        self,
        messages: List[LLMMessage],
        model: str,
        tools: Dict[str, Any],
        generation_config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Non-streaming generation"""
        # Translate ModelMessage format to OpenAI format
        try:
            msgs = translate_to_openai(messages)
            logger.debug(f"Translated {len(messages)} messages to OpenAI format: {len(msgs)} messages")
        except MessageTranslationError as e:
            logger.error(f"Message translation failed in generate(): {e}")
            raise ValueError(f"Failed to translate messages to OpenAI format: {e}") from e

        oaitools = [
            {'type': 'function', 'function': {'name': n, 'description': s.get('description', f'Tool: {n}'), 'parameters': s['input']}}
            for n, s in tools.items()
        ] if tools else []

        # Add generation config parameters
        config = generation_config or {}

        payload = {'model': model, 'messages': msgs}
        if oaitools:
            payload['tools'] = oaitools
            payload['tool_choice'] = 'auto'
            # Disable parallel tool calls if specified (prevents duplicate calls in same response)
            if config.get('parallel_tool_calls') is False:
                payload['parallel_tool_calls'] = False

        if 'temperature' in config:
            payload['temperature'] = config['temperature']
        if 'max_tokens' in config:
            payload['max_tokens'] = config['max_tokens']
        if 'top_p' in config:
            payload['top_p'] = config['top_p']
        if 'presence_penalty' in config:
            payload['presence_penalty'] = config['presence_penalty']
        if 'frequency_penalty' in config:
            payload['frequency_penalty'] = config['frequency_penalty']
        if 'stop' in config:
            payload['stop'] = config['stop']
        if 'seed' in config:
            payload['seed'] = config['seed']
        if 'logit_bias' in config:
            payload['logit_bias'] = config['logit_bias']
        if 'user' in config:
            payload['user'] = config['user']

        async with httpx.AsyncClient(timeout=60.0) as client:
            r = await client.post(
                f"{self.base}/chat/completions",
                headers=self._headers(),
                json=payload
            )
            r.raise_for_status()
            data = r.json()

        msg = data['choices'][0].get('message', {})
        usage = data.get('usage', {})

        tc = (msg.get('tool_calls') or [None])[0]
        if tc:
            return {
                'tool': tc['function']['name'],
                'args': json.loads(tc['function'].get('arguments') or '{}'),
                'usage': usage
            }
        return {'done': True, 'answer': msg.get('content', ''), 'usage': usage}


class OpenAIResponsesProvider(BaseProvider):
    """
    OpenAI Responses API provider implementing stream protocol.

    Uses the /v1/responses endpoint which provides structured events:
    - response.text.delta, response.reasoning.delta
    - response.output_item.added (for synthesis)
    - response.function_call_arguments.delta
    - Provider-executed tools (web_search, computer)

    Requires OpenAI API key with Responses API access.
    """
    name = 'openai-responses'

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize OpenAI Responses API provider.

        Args:
            api_key: Optional API key. If not provided, falls back to OPENAI_API_KEY environment variable.
        """
        self.base = os.getenv('OPENAI_API_BASE', 'https://api.openai.com/v1')
        self.translator = OpenAIResponsesAPITranslator()

        # Use provided API key or fall back to environment variable
        self.api_key = api_key or os.getenv('OPENAI_API_KEY', '')
        if not self.api_key:
            raise ValueError("OpenAI API key not provided. Either pass api_key parameter or set OPENAI_API_KEY environment variable.")

    def _headers(self):
        """Get headers with API key"""
        return {
            'Authorization': f"Bearer {self.api_key}",
            'Content-Type': 'application/json'
        }

    async def stream(
        self,
        messages: List[LLMMessage],
        model: str,
        tools: Dict[str, Any],
        generation_config: Optional[Dict[str, Any]] = None
    ) -> AsyncGenerator[StreamEvent, None]:
        """Stream OpenAI Responses API events as stream protocol events"""
        # Reset translator state
        self.translator.reset()

        # Translate ModelMessage format to OpenAI format first
        try:
            openai_messages = translate_to_openai(messages)
            logger.debug(f"Translated {len(messages)} messages to OpenAI format for Responses API")
        except MessageTranslationError as e:
            logger.error(f"Message translation failed: {e}")
            yield ErrorEvent(error=str(e))
            return

        # Convert OpenAI format messages to Responses API format
        # Responses API uses 'input' field with special input_text type
        input_messages = []
        for m in openai_messages:
            # Extract text content from OpenAI message
            content = m.get('content', '')
            if isinstance(content, list):
                # Multimodal content - extract text parts
                text_parts = [p.get('text', '') for p in content if p.get('type') == 'text']
                content_text = ' '.join(text_parts)
            else:
                content_text = content

            input_messages.append({
                'role': m.get('role', 'user'),
                'content': [
                    {
                        'type': 'input_text',
                        'text': content_text
                    }
                ]
            })

        # Convert tools to Responses API format
        response_tools = []
        if tools:
            for name, schema in tools.items():
                response_tools.append({
                    'type': 'function',
                    'function': {
                        'name': name,
                        'description': schema.get('description', f'Tool: {name}'),
                        'parameters': schema['input']
                    }
                })

        payload = {
            'model': model,
            'input': input_messages,  # 'input' not 'messages'
            'stream': True
        }

        # Only include tools if there are any (reasoning models don't support tools)
        if response_tools:
            payload['tools'] = response_tools
            payload['tool_choice'] = 'auto'

        # Add generation config parameters
        config = generation_config or {}
        if 'temperature' in config:
            payload['temperature'] = config['temperature']
        if 'max_tokens' in config:
            # Responses API uses 'max_output_tokens' instead of 'max_tokens'
            payload['max_output_tokens'] = config['max_tokens']
        if 'top_p' in config:
            payload['top_p'] = config['top_p']

        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                async with client.stream(
                    'POST',
                    f"{self.base}/responses",
                    headers=self._headers(),
                    json=payload
                ) as response:
                    response.raise_for_status()

                    async for line in response.aiter_lines():
                        if not line.strip() or line.strip() == 'data: [DONE]':
                            continue
                        if line.startswith('data: '):
                            data_str = line[6:]
                            try:
                                event = json.loads(data_str)
                            except json.JSONDecodeError:
                                continue

                            # Translate event to Vel stream protocol
                            vel_event = self.translator.translate_event(event)
                            if vel_event:
                                yield vel_event

                            # Drain any pending events
                            while True:
                                pending = self.translator.get_pending_event()
                                if pending is None:
                                    break
                                yield pending

                            # Check for completion
                            event_type = event.get('type', '')
                            if event_type in ('response.completed', 'response.done'):
                                # Finalize and emit FinishMessageEvent
                                while True:
                                    final_event = self.translator._finalize()
                                    if final_event is None:
                                        break
                                    yield final_event
                                yield FinishMessageEvent(finish_reason='stop')
                                return

        except Exception as e:
            yield self._create_error_event(e, 'openai-responses')

    def _create_error_event(self, exception: Exception, provider: str) -> ErrorEvent:
        """Create detailed ErrorEvent from exception

        Extracts HTTP status codes, error codes, and provider-specific error details.
        """
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
                # For streaming responses, body may not have been read yet
                # Check if response content is available before accessing
                if hasattr(response, 'is_stream_consumed') and not response.is_stream_consumed:
                    # Streaming response - body not available
                    error_message = f"OpenAI API error: HTTP {status_code}"
                elif hasattr(response, '_content') and response._content:
                    # Non-streaming response with content available
                    error_body = response.json()
                    if 'error' in error_body:
                        error_data = error_body['error']
                        error_message = error_data.get('message', str(exception))
                        error_code = error_data.get('code')
                        error_type = error_data.get('type')
                        error_details = error_data
                else:
                    # Fallback - try to read but catch if not available
                    error_body = response.json()
                    if 'error' in error_body:
                        error_data = error_body['error']
                        error_message = error_data.get('message', str(exception))
                        error_code = error_data.get('code')
                        error_type = error_data.get('type')
                        error_details = error_data
            except (httpx.ResponseNotRead, Exception):
                # If response body not available or JSON parsing fails
                error_message = f"OpenAI API error: HTTP {status_code}"

            return ErrorEvent(
                error=error_message,
                error_code=error_code,
                error_type=error_type,
                status_code=status_code,
                provider=provider,
                details=error_details if error_details else None
            )

        # Handle other httpx errors
        elif isinstance(exception, httpx.RequestError):
            return ErrorEvent(
                error=f"Request failed: {str(exception)}",
                error_type='request_error',
                provider=provider
            )

        # Generic exception handling
        return ErrorEvent(
            error=str(exception),
            error_type=type(exception).__name__,
            provider=provider
        )

    async def generate(
        self,
        messages: List[LLMMessage],
        model: str,
        tools: Dict[str, Any],
        generation_config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Non-streaming generation for Responses API.

        Note: Responses API is primarily designed for streaming.
        This method aggregates the stream into a single response.
        """
        # Translate ModelMessage format to OpenAI format
        try:
            msgs = translate_to_openai(messages)
            logger.debug(f"Translated {len(messages)} messages to OpenAI format for Responses API generate()")
        except MessageTranslationError as e:
            logger.error(f"Message translation failed in generate(): {e}")
            raise ValueError(f"Failed to translate messages to OpenAI format: {e}") from e

        response_tools = []
        if tools:
            for name, schema in tools.items():
                response_tools.append({
                    'type': 'function',
                    'function': {
                        'name': name,
                        'description': schema.get('description', f'Tool: {name}'),
                        'parameters': schema['input']
                    }
                })

        payload = {'model': model, 'messages': msgs}
        if response_tools:
            payload['tools'] = response_tools

        config = generation_config or {}
        if 'temperature' in config:
            payload['temperature'] = config['temperature']
        if 'max_tokens' in config:
            payload['max_tokens'] = config['max_tokens']
        if 'top_p' in config:
            payload['top_p'] = config['top_p']

        # Aggregate streaming response
        full_text = []
        tool_call = None
        usage = {}

        async for event in self.stream(messages, model, tools, generation_config):
            if event.type == 'text-delta':
                full_text.append(event.delta)
            elif event.type == 'tool-input-available':
                tool_call = {'name': event.tool_name, 'input': event.input}
            elif event.type == 'response-metadata' and hasattr(event, 'usage'):
                usage = event.usage

        if tool_call:
            return {
                'tool': tool_call['name'],
                'args': tool_call['input'],
                'usage': usage
            }
        return {'done': True, 'answer': ''.join(full_text), 'usage': usage}
