"""Google Gemini provider with stream protocol support"""
from __future__ import annotations
import os, json, uuid
from typing import Any, AsyncGenerator, Dict, List
from .base import BaseProvider, LLMMessage
from ..events import (
    StreamEvent, StartEvent, TextStartEvent, TextDeltaEvent, TextEndEvent,
    ToolInputStartEvent, ToolInputAvailableEvent,
    FinishMessageEvent, ErrorEvent
)

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
        tools: Dict[str, Any]
    ) -> AsyncGenerator[StreamEvent, None]:
        """Stream Gemini response as stream protocol events"""
        try:
            gemini_model = genai.GenerativeModel(model)
            gemini_messages = self._convert_messages(messages)

            # Build generation config
            gen_config = {}
            tool_config = None
            if tools:
                tool_declarations = self._convert_tools(tools)
                tool_config = {'function_declarations': tool_declarations}

            # Start streaming
            text_block_id = None

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
                # Handle text content
                if hasattr(chunk, 'text') and chunk.text:
                    if text_block_id is None:
                        text_block_id = str(uuid.uuid4())
                        yield TextStartEvent(block_id=text_block_id)
                    yield TextDeltaEvent(block_id=text_block_id, delta=chunk.text)

                # Handle function calls
                if hasattr(chunk, 'parts'):
                    for part in chunk.parts:
                        if hasattr(part, 'function_call'):
                            fc = part.function_call
                            tool_call_id = f"call_{uuid.uuid4().hex[:8]}"
                            tool_name = fc.name

                            # Convert args to dict
                            args = dict(fc.args) if hasattr(fc, 'args') else {}

                            yield ToolInputStartEvent(
                                tool_call_id=tool_call_id,
                                tool_name=tool_name
                            )
                            yield ToolInputAvailableEvent(
                                tool_call_id=tool_call_id,
                                tool_name=tool_name,
                                input=args
                            )

            # End text block if active
            if text_block_id:
                yield TextEndEvent(block_id=text_block_id)

            yield FinishMessageEvent(finish_reason='stop')

        except Exception as e:
            yield ErrorEvent(error=str(e))

    async def generate(
        self,
        messages: List[LLMMessage],
        model: str,
        tools: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Non-streaming generation"""
        try:
            gemini_model = genai.GenerativeModel(model)
            gemini_messages = self._convert_messages(messages)

            gen_config = {}
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
