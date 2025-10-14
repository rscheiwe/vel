"""Anthropic Claude provider with stream protocol support"""
from __future__ import annotations
import os, httpx, json
from typing import Any, AsyncGenerator, Dict, List
from .base import BaseProvider, LLMMessage
from .translators import AnthropicAPITranslator
from ..events import StreamEvent, ErrorEvent

def _headers():
    api_key = os.getenv('ANTHROPIC_API_KEY', '')
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY environment variable is not set")
    return {
        'x-api-key': api_key,
        'anthropic-version': '2023-06-01',
        'Content-Type': 'application/json'
    }

class AnthropicProvider(BaseProvider):
    """Anthropic Claude provider implementing stream protocol"""
    name = 'anthropic'

    def __init__(self):
        self.base = os.getenv('ANTHROPIC_API_BASE', 'https://api.anthropic.com')
        self.translator = AnthropicAPITranslator()
        # Validate API key is set
        api_key = os.getenv('ANTHROPIC_API_KEY', '')
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable is not set. Set it in your .env file or export it.")

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
                'description': schema.get('description', f"Function {name}"),
                'input_schema': schema['input']
            })
        return anthropic_tools

    async def stream(
        self,
        messages: List[LLMMessage],
        model: str,
        tools: Dict[str, Any]
    ) -> AsyncGenerator[StreamEvent, None]:
        """Stream Anthropic response as stream protocol events"""
        # Reset translator state
        self.translator.reset()

        system_message, anthropic_messages = self._convert_messages(messages)
        anthropic_tools = self._convert_tools(tools)

        payload = {
            'model': model,
            'messages': anthropic_messages,
            'max_tokens': 4096,
            'stream': True
        }

        if system_message:
            payload['system'] = system_message

        if anthropic_tools:
            payload['tools'] = anthropic_tools

        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                async with client.stream(
                    'POST',
                    f"{self.base}/v1/messages",
                    headers=_headers(),
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
            yield ErrorEvent(error=str(e))

    async def generate(
        self,
        messages: List[LLMMessage],
        model: str,
        tools: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Non-streaming generation"""
        system_message, anthropic_messages = self._convert_messages(messages)
        anthropic_tools = self._convert_tools(tools)

        payload = {
            'model': model,
            'messages': anthropic_messages,
            'max_tokens': 4096
        }

        if system_message:
            payload['system'] = system_message

        if anthropic_tools:
            payload['tools'] = anthropic_tools

        async with httpx.AsyncClient(timeout=30.0) as client:
            r = await client.post(
                f"{self.base}/v1/messages",
                headers=_headers(),
                json=payload
            )
            r.raise_for_status()
            data = r.json()

        # Parse response
        content = data.get('content', [])

        # Check for tool use
        for block in content:
            if block.get('type') == 'tool_use':
                return {
                    'tool': block.get('name'),
                    'args': block.get('input', {})
                }

        # Extract text content
        text_content = ''.join([
            block.get('text', '')
            for block in content
            if block.get('type') == 'text'
        ])

        return {'done': True, 'answer': text_content}
