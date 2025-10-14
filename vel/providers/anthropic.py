"""Anthropic Claude provider with stream protocol support"""
from __future__ import annotations
import os, httpx, json, uuid
from typing import Any, AsyncGenerator, Dict, List
from .base import BaseProvider, LLMMessage
from ..events import (
    StreamEvent, StartEvent, TextStartEvent, TextDeltaEvent, TextEndEvent,
    ToolInputStartEvent, ToolInputDeltaEvent, ToolInputAvailableEvent,
    FinishMessageEvent, ErrorEvent
)

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

                    # Track state for event emission
                    content_blocks = {}  # index -> {type, block_id, buffer}
                    finish_reason = 'end_turn'

                    async for line in response.aiter_lines():
                        if not line.strip():
                            continue

                        # Parse SSE format: "event: <type>\ndata: <json>"
                        if line.startswith('event: '):
                            event_type = line[7:].strip()
                            continue
                        elif line.startswith('data: '):
                            data_str = line[6:].strip()
                            try:
                                data = json.loads(data_str)
                            except json.JSONDecodeError:
                                continue

                            event_type = data.get('type')

                            # Handle message_start
                            if event_type == 'message_start':
                                pass  # Could emit StartEvent here if needed

                            # Handle content_block_start
                            elif event_type == 'content_block_start':
                                index = data.get('index', 0)
                                content_block = data.get('content_block', {})
                                block_type = content_block.get('type')

                                if block_type == 'text':
                                    block_id = str(uuid.uuid4())
                                    content_blocks[index] = {
                                        'type': 'text',
                                        'block_id': block_id,
                                        'buffer': []
                                    }
                                    yield TextStartEvent(block_id=block_id)

                                elif block_type == 'tool_use':
                                    tool_id = content_block.get('id', f"call_{uuid.uuid4().hex[:8]}")
                                    tool_name = content_block.get('name', '')
                                    content_blocks[index] = {
                                        'type': 'tool_use',
                                        'tool_id': tool_id,
                                        'tool_name': tool_name,
                                        'input_buffer': ''
                                    }
                                    yield ToolInputStartEvent(
                                        tool_call_id=tool_id,
                                        tool_name=tool_name
                                    )

                            # Handle content_block_delta
                            elif event_type == 'content_block_delta':
                                index = data.get('index', 0)
                                delta = data.get('delta', {})
                                delta_type = delta.get('type')

                                if index in content_blocks:
                                    block = content_blocks[index]

                                    if delta_type == 'text_delta':
                                        text = delta.get('text', '')
                                        block['buffer'].append(text)
                                        yield TextDeltaEvent(
                                            block_id=block['block_id'],
                                            delta=text
                                        )

                                    elif delta_type == 'input_json_delta':
                                        partial_json = delta.get('partial_json', '')
                                        block['input_buffer'] += partial_json
                                        yield ToolInputDeltaEvent(
                                            tool_call_id=block['tool_id'],
                                            input_delta=partial_json
                                        )

                            # Handle content_block_stop
                            elif event_type == 'content_block_stop':
                                index = data.get('index', 0)
                                if index in content_blocks:
                                    block = content_blocks[index]

                                    if block['type'] == 'text':
                                        yield TextEndEvent(block_id=block['block_id'])

                                    elif block['type'] == 'tool_use':
                                        # Parse accumulated JSON input
                                        try:
                                            tool_input = json.loads(block['input_buffer'] or '{}')
                                        except json.JSONDecodeError:
                                            tool_input = {}

                                        yield ToolInputAvailableEvent(
                                            tool_call_id=block['tool_id'],
                                            tool_name=block['tool_name'],
                                            input=tool_input
                                        )

                            # Handle message_delta
                            elif event_type == 'message_delta':
                                delta = data.get('delta', {})
                                finish_reason = delta.get('stop_reason', 'end_turn')

                            # Handle message_stop
                            elif event_type == 'message_stop':
                                yield FinishMessageEvent(finish_reason=finish_reason)
                                return

                            # Handle error
                            elif event_type == 'error':
                                error_msg = data.get('error', {}).get('message', 'Unknown error')
                                yield ErrorEvent(error=error_msg)
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
