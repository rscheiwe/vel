from __future__ import annotations
import os, httpx, json
from typing import Any, Dict, List

LLMMessage = Dict[str, Any]

def _headers():
    return {'Authorization': f"Bearer {os.getenv('OPENAI_API_KEY','')}", 'Content-Type':'application/json'}

class Provider:
    name: str
    async def plan(self, messages: List[LLMMessage], model: str, tools: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError

class OpenAIProvider(Provider):
    name = 'openai'
    def __init__(self):
        self.base = os.getenv('OPENAI_API_BASE','https://api.openai.com/v1')

    async def plan(self, messages: List[LLMMessage], model: str, tools: Dict[str, Any]) -> Dict[str, Any]:
        msgs = [{'role': m.get('role','user'), 'content': m.get('content','')} for m in messages]
        oaitools = [{'type':'function','function':{'name':n,'parameters':s['input']}} for n,s in tools.items()]
        async with httpx.AsyncClient(timeout=30.0) as client:
            r = await client.post(f"{self.base}/chat/completions", headers=_headers(), json={'model':model,'messages':msgs,'tools':oaitools,'tool_choice':'auto'})
            r.raise_for_status()
            data = r.json()
        msg = data['choices'][0].get('message',{})
        tc = (msg.get('tool_calls') or [None])[0]
        if tc:
            return {'tool': tc['function']['name'], 'args': json.loads(tc['function'].get('arguments') or '{}')}
        return {'done': True, 'answer': msg.get('content','')}

class ProviderRegistry:
    def __init__(self):
        self._providers = {'openai': OpenAIProvider()}

    @classmethod
    def default(cls) -> 'ProviderRegistry':
        return cls()

    def get(self, name: str) -> Provider:
        return self._providers[name]
