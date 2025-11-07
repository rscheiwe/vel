"""Google Gemini provider with stream protocol support"""
from __future__ import annotations
import os, json
from typing import Any, AsyncGenerator, Dict, List, Optional
from .base import BaseProvider, LLMMessage
from .translators import GeminiAPITranslator
from .message_translator import translate_to_gemini, MessageTranslationError
from ..events import StreamEvent, FinishMessageEvent, ErrorEvent, ToolInputAvailableEvent
import logging

logger = logging.getLogger('vel.providers.google')

try:
    import google.generativeai as genai
except ImportError:
    genai = None

class GeminiProvider(BaseProvider):
    """Google Gemini provider implementing stream protocol"""
    name = 'google'

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Gemini provider.

        Args:
            api_key: Optional API key. If not provided, falls back to GOOGLE_API_KEY environment variable.
        """
        if genai is None:
            raise ImportError("google-generativeai not installed. Install with: pip install google-generativeai")

        # Use provided API key or fall back to environment variable
        self.api_key = api_key or os.getenv('GOOGLE_API_KEY', '')
        if not self.api_key:
            raise ValueError("Google API key not provided. Either pass api_key parameter or set GOOGLE_API_KEY environment variable.")

        genai.configure(api_key=self.api_key)
        self.translator = GeminiAPITranslator()

    def _convert_messages(self, messages: List[LLMMessage]) -> List[Dict[str, Any]]:
        """Convert messages to Gemini format"""
        gemini_messages = []
        for msg in messages:
            role = msg.get('role', 'user')
            # Gemini uses 'user' and 'model' roles
            if role == 'assistant':
                role = 'model'
            gemini_messages.append({
                'role': role,
                'parts': [msg.get('content', '')]
            })
        return gemini_messages

    def _convert_tools(self, tools: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Convert tool schemas to Gemini function declarations"""
        if not tools:
            return []

        declarations = []
        for name, schema in tools.items():
            declarations.append({
                'name': name,
                'description': schema.get('description', f'Tool: {name}'),
                'parameters': schema['input']
            })
        return declarations

    async def stream(
        self,
        messages: List[LLMMessage],
        model: str,
        tools: Dict[str, Any],
        generation_config: Optional[Dict[str, Any]] = None
    ) -> AsyncGenerator[StreamEvent, None]:
        """Stream Gemini response as stream protocol events"""
        # Reset translator state
        self.translator.reset()

        try:
            # Translate ModelMessage format to Gemini format
            try:
                system_instruction, gemini_messages = translate_to_gemini(messages)
                logger.debug(f"Translated {len(messages)} messages to Gemini format: {len(gemini_messages)} messages")
            except MessageTranslationError as e:
                logger.error(f"Message translation failed: {e}")
                yield ErrorEvent(error=str(e))
                return

            # Create model with optional system instruction
            if system_instruction:
                gemini_model = genai.GenerativeModel(
                    model,
                    system_instruction=system_instruction
                )
            else:
                gemini_model = genai.GenerativeModel(model)

            # Build generation config
            config = generation_config or {}
            gen_config = {}

            # Map common parameters to Gemini's GenerationConfig
            if 'temperature' in config:
                gen_config['temperature'] = config['temperature']
            if 'max_tokens' in config:
                gen_config['max_output_tokens'] = config['max_tokens']
            if 'max_output_tokens' in config:  # Direct Gemini parameter
                gen_config['max_output_tokens'] = config['max_output_tokens']
            if 'top_p' in config:
                gen_config['top_p'] = config['top_p']
            if 'top_k' in config:
                gen_config['top_k'] = config['top_k']
            if 'stop_sequences' in config:
                gen_config['stop_sequences'] = config['stop_sequences']
            if 'stop' in config:  # Alias
                gen_config['stop_sequences'] = config['stop']

            tool_config = None
            if tools:
                tool_declarations = self._convert_tools(tools)
                tool_config = {'function_declarations': tool_declarations}

            # Gemini chat requires history + current message split
            history = gemini_messages[:-1] if len(gemini_messages) > 1 else []
            current_message = gemini_messages[-1]['parts'][0] if gemini_messages else "Hello"

            chat = gemini_model.start_chat(history=history)

            response = await chat.send_message_async(
                current_message,
                generation_config=gen_config,
                tools=tool_config if tool_config else None,
                stream=True
            )

            async for chunk in response:
                # Translate chunk to Vel event
                vel_event = self.translator.translate_chunk(chunk)
                if vel_event:
                    yield vel_event

                # Drain any pending events from translator
                # (Gemini emits complete function calls, translator queues tool-input-available)
                while True:
                    pending = self.translator.get_pending_event()
                    if pending is None:
                        break
                    yield pending

            # End text block if active
            text_end_event = self.translator.finalize_text_block()
            if text_end_event:
                yield text_end_event

            yield FinishMessageEvent(finish_reason='stop')

        except Exception as e:
            yield self._create_error_event(e)

    def _create_error_event(self, exception: Exception) -> ErrorEvent:
        """Create detailed ErrorEvent from exception

        Extracts error codes and provider-specific error details.
        """
        # Handle Google API errors
        if hasattr(exception, 'reason'):
            # google.api_core.exceptions have 'reason' attribute
            return ErrorEvent(
                error=str(exception),
                error_code=getattr(exception, 'code', None),
                error_type=type(exception).__name__,
                provider='google'
            )

        # Generic exception handling
        return ErrorEvent(
            error=str(exception),
            error_type=type(exception).__name__,
            provider='google'
        )

    async def generate(
        self,
        messages: List[LLMMessage],
        model: str,
        tools: Dict[str, Any],
        generation_config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Non-streaming generation"""
        try:
            # Translate ModelMessage format to Gemini format
            try:
                system_instruction, gemini_messages = translate_to_gemini(messages)
                logger.debug(f"Translated {len(messages)} messages to Gemini format in generate()")
            except MessageTranslationError as e:
                logger.error(f"Message translation failed in generate(): {e}")
                raise ValueError(f"Failed to translate messages to Gemini format: {e}") from e

            # Create model with optional system instruction
            if system_instruction:
                gemini_model = genai.GenerativeModel(
                    model,
                    system_instruction=system_instruction
                )
            else:
                gemini_model = genai.GenerativeModel(model)

            # Build generation config
            config = generation_config or {}
            gen_config = {}

            # Map common parameters to Gemini's GenerationConfig
            if 'temperature' in config:
                gen_config['temperature'] = config['temperature']
            if 'max_tokens' in config:
                gen_config['max_output_tokens'] = config['max_tokens']
            if 'max_output_tokens' in config:  # Direct Gemini parameter
                gen_config['max_output_tokens'] = config['max_output_tokens']
            if 'top_p' in config:
                gen_config['top_p'] = config['top_p']
            if 'top_k' in config:
                gen_config['top_k'] = config['top_k']
            if 'stop_sequences' in config:
                gen_config['stop_sequences'] = config['stop_sequences']
            if 'stop' in config:  # Alias
                gen_config['stop_sequences'] = config['stop']

            tool_config = None
            if tools:
                tool_declarations = self._convert_tools(tools)
                tool_config = {'function_declarations': tool_declarations}

            # Gemini chat requires history + current message split
            history = gemini_messages[:-1] if len(gemini_messages) > 1 else []
            current_message = gemini_messages[-1]['parts'][0] if gemini_messages else "Hello"

            chat = gemini_model.start_chat(history=history)

            response = await chat.send_message_async(
                current_message,
                generation_config=gen_config,
                tools=tool_config if tool_config else None
            )

            # Extract usage if available
            usage = {}
            if hasattr(response, 'usage_metadata'):
                um = response.usage_metadata
                usage = {
                    'prompt_tokens': getattr(um, 'prompt_token_count', 0),
                    'completion_tokens': getattr(um, 'candidates_token_count', 0),
                    'total_tokens': getattr(um, 'total_token_count', 0)
                }

            # Check for function calls
            if hasattr(response, 'parts'):
                for part in response.parts:
                    if hasattr(part, 'function_call'):
                        fc = part.function_call
                        args = dict(fc.args) if hasattr(fc, 'args') else {}
                        return {
                            'tool': fc.name,
                            'args': args,
                            'usage': usage
                        }

            # Return text response
            return {'done': True, 'answer': response.text if hasattr(response, 'text') else '', 'usage': usage}

        except Exception as e:
            # Raise with enhanced error context
            error_msg = str(e)
            error_type = type(e).__name__
            if hasattr(e, 'code'):
                raise RuntimeError(f"Gemini generation failed ({error_type}, code={e.code}): {error_msg}") from e
            raise RuntimeError(f"Gemini generation failed ({error_type}): {error_msg}") from e
