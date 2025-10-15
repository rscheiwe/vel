"""Google Gemini provider with stream protocol support"""
from __future__ import annotations
import os, json
from typing import Any, AsyncGenerator, Dict, List, Optional
from .base import BaseProvider, LLMMessage
from .translators import GeminiAPITranslator
from ..events import StreamEvent, FinishMessageEvent, ErrorEvent, ToolInputAvailableEvent

try:
    import google.generativeai as genai
except ImportError:
    genai = None

class GeminiProvider(BaseProvider):
    """Google Gemini provider implementing stream protocol"""
    name = 'google'

    def __init__(self):
        if genai is None:
            raise ImportError("google-generativeai not installed. Install with: pip install google-generativeai")

        api_key = os.getenv('GOOGLE_API_KEY')
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable is not set. Set it in your .env file or export it.")
        genai.configure(api_key=api_key)
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
                'description': schema.get('description', f"Function {name}"),
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
            gemini_model = genai.GenerativeModel(model)
            gemini_messages = self._convert_messages(messages)

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

            tool_calls_seen = []  # Track tool calls to emit ToolInputAvailable after start

            async for chunk in response:
                # Translate chunk to Vel event
                vel_event = self.translator.translate_chunk(chunk)
                if vel_event:
                    yield vel_event

                # Handle function calls (Gemini emits complete calls, need to emit available)
                if hasattr(chunk, 'parts'):
                    for part in chunk.parts:
                        if hasattr(part, 'function_call'):
                            fc = part.function_call
                            tool_call_id = f"call_{fc.name}_{len(tool_calls_seen)}"
                            tool_name = fc.name
                            args = dict(fc.args) if hasattr(fc, 'args') else {}

                            # Emit ToolInputAvailableEvent for V5 UI Stream Protocol
                            yield ToolInputAvailableEvent(
                                tool_call_id=tool_call_id,
                                tool_name=tool_name,
                                input=args
                            )
                            tool_calls_seen.append(tool_call_id)

            # End text block if active
            text_end_event = self.translator.finalize_text_block()
            if text_end_event:
                yield text_end_event

            yield FinishMessageEvent(finish_reason='stop')

        except Exception as e:
            yield ErrorEvent(error=str(e))

    async def generate(
        self,
        messages: List[LLMMessage],
        model: str,
        tools: Dict[str, Any],
        generation_config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Non-streaming generation"""
        try:
            gemini_model = genai.GenerativeModel(model)
            gemini_messages = self._convert_messages(messages)

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

            # Check for function calls
            if hasattr(response, 'parts'):
                for part in response.parts:
                    if hasattr(part, 'function_call'):
                        fc = part.function_call
                        args = dict(fc.args) if hasattr(fc, 'args') else {}
                        return {
                            'tool': fc.name,
                            'args': args
                        }

            # Return text response
            return {'done': True, 'answer': response.text if hasattr(response, 'text') else ''}

        except Exception as e:
            raise RuntimeError(f"Gemini generation failed: {e}")
