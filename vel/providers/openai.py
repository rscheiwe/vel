"""OpenAI provider with stream protocol support"""
from __future__ import annotations
import os, httpx, json
from typing import Any, AsyncGenerator, Dict, List
from .base import BaseProvider, LLMMessage
from .translators import OpenAIAPITranslator
from ..events import StreamEvent, FinishMessageEvent, ErrorEvent

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
        self.translator = OpenAIAPITranslator()
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
        # Reset translator state
        self.translator.reset()

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

                            # Translate chunk to Vel event
                            vel_event = self.translator.translate_chunk(chunk)
                            if vel_event:
                                yield vel_event

                            # Check for finish
                            finish_reason = chunk.get('choices', [{}])[0].get('finish_reason')
                            if finish_reason:
                                # Finalize tool calls
                                for tool_event in self.translator.finalize_tool_calls():
                                    yield tool_event

                                yield FinishMessageEvent(finish_reason=finish_reason)
                                return

                    # Fallback if stream ended without finish_reason
                    for tool_event in self.translator.finalize_tool_calls():
                        yield tool_event
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
