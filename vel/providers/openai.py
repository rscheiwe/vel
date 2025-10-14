"""OpenAI provider with stream protocol support"""
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
    api_key = os.getenv('OPENAI_API_KEY', '')
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable is not set")
    return {
        'Authorization': f"Bearer {api_key}",
        'Content-Type': 'application/json'
    }

class OpenAIProvider(BaseProvider):
    """OpenAI provider implementing stream protocol"""
    name = 'openai'

    def __init__(self):
        self.base = os.getenv('OPENAI_API_BASE', 'https://api.openai.com/v1')
        # Validate API key is set
        api_key = os.getenv('OPENAI_API_KEY', '')
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set. Set it in your .env file or export it.")

    async def stream(
        self,
        messages: List[LLMMessage],
        model: str,
        tools: Dict[str, Any]
    ) -> AsyncGenerator[StreamEvent, None]:
        """Stream OpenAI response as stream protocol events"""
        msgs = [{'role': m.get('role', 'user'), 'content': m.get('content', '')} for m in messages]
        oaitools = [
            {'type': 'function', 'function': {'name': n, 'parameters': s['input']}}
            for n, s in tools.items()
        ] if tools else []

        payload = {
            'model': model,
            'messages': msgs,
            'stream': True
        }
        if oaitools:
            payload['tools'] = oaitools
            payload['tool_choice'] = 'auto'

        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                async with client.stream(
                    'POST',
                    f"{self.base}/chat/completions",
                    headers=_headers(),
                    json=payload
                ) as response:
                    response.raise_for_status()

                    # Track state for event emission
                    text_block_id = None
                    tool_calls = {}  # tool_index -> {id, name, args_buffer}

                    async for line in response.aiter_lines():
                        if not line.strip() or line.strip() == 'data: [DONE]':
                            continue
                        if line.startswith('data: '):
                            data_str = line[6:]
                            try:
                                chunk = json.loads(data_str)
                            except json.JSONDecodeError:
                                continue

                            delta = chunk.get('choices', [{}])[0].get('delta', {})
                            finish_reason = chunk.get('choices', [{}])[0].get('finish_reason')

                            # Handle text content
                            if 'content' in delta and delta['content']:
                                content = delta['content']
                                if text_block_id is None:
                                    text_block_id = str(uuid.uuid4())
                                    yield TextStartEvent(block_id=text_block_id)
                                yield TextDeltaEvent(block_id=text_block_id, delta=content)

                            # Handle tool calls
                            if 'tool_calls' in delta:
                                for tc in delta['tool_calls']:
                                    idx = tc.get('index', 0)
                                    if idx not in tool_calls:
                                        # New tool call
                                        tool_id = tc.get('id', f"call_{uuid.uuid4().hex[:8]}")
                                        tool_name = tc.get('function', {}).get('name', '')
                                        tool_calls[idx] = {
                                            'id': tool_id,
                                            'name': tool_name,
                                            'args_buffer': ''
                                        }
                                        if tool_name:
                                            yield ToolInputStartEvent(
                                                tool_call_id=tool_id,
                                                tool_name=tool_name
                                            )

                                    # Accumulate function arguments
                                    if 'function' in tc and 'arguments' in tc['function']:
                                        args_delta = tc['function']['arguments']
                                        tool_calls[idx]['args_buffer'] += args_delta
                                        yield ToolInputDeltaEvent(
                                            tool_call_id=tool_calls[idx]['id'],
                                            input_delta=args_delta
                                        )

                            # Handle finish
                            if finish_reason:
                                # End text block if active
                                if text_block_id:
                                    yield TextEndEvent(block_id=text_block_id)
                                    text_block_id = None

                                # Emit tool input available for each tool call
                                for tc_data in tool_calls.values():
                                    try:
                                        args = json.loads(tc_data['args_buffer'] or '{}')
                                    except json.JSONDecodeError:
                                        args = {}
                                    yield ToolInputAvailableEvent(
                                        tool_call_id=tc_data['id'],
                                        tool_name=tc_data['name'],
                                        input=args
                                    )

                                yield FinishMessageEvent(finish_reason=finish_reason)
                                return

                    # Fallback: end text block if stream ended without finish_reason
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
        msgs = [{'role': m.get('role', 'user'), 'content': m.get('content', '')} for m in messages]
        oaitools = [
            {'type': 'function', 'function': {'name': n, 'parameters': s['input']}}
            for n, s in tools.items()
        ] if tools else []

        payload = {'model': model, 'messages': msgs}
        if oaitools:
            payload['tools'] = oaitools
            payload['tool_choice'] = 'auto'

        async with httpx.AsyncClient(timeout=30.0) as client:
            r = await client.post(
                f"{self.base}/chat/completions",
                headers=_headers(),
                json=payload
            )
            r.raise_for_status()
            data = r.json()

        msg = data['choices'][0].get('message', {})
        tc = (msg.get('tool_calls') or [None])[0]
        if tc:
            return {
                'tool': tc['function']['name'],
                'args': json.loads(tc['function'].get('arguments') or '{}')
            }
        return {'done': True, 'answer': msg.get('content', '')}
