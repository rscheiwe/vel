from __future__ import annotations
import os
from typing import Any, Dict, List

LLMMessage = Dict[str, Any]

class Provider:
    name: str
    async def plan(self, messages: List[LLMMessage], model: str, tools: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError

class OpenAIProvider(Provider):
    name = "openai"
    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY","")

    async def plan(self, messages: List[LLMMessage], model: str, tools: Dict[str, Any]) -> Dict[str, Any]:
        # Skeleton planner: if user asks for weather, call tool, else finish.
        if any(m.get("content","").lower().startswith("weather") for m in messages if m.get("role")=="user"):
            return {"tool":"get_weather","args":{"city":"San Francisco"}}
        return {"done": True, "answer": "Hello from Vel (skeleton planner)."}

class ProviderRegistry:
    def __init__(self):
        self._providers = {"openai": OpenAIProvider()}

    @classmethod
    def default(cls) -> "ProviderRegistry":
        return cls()

    def get(self, name: str) -> Provider:
        return self._providers[name]
