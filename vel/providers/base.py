"""Base provider interface for LLM providers"""
from __future__ import annotations
from typing import Any, AsyncGenerator, Dict, List
from abc import ABC, abstractmethod
from ..events import StreamEvent

LLMMessage = Dict[str, Any]

class BaseProvider(ABC):
    """Base interface for LLM providers"""

    name: str

    @abstractmethod
    async def stream(
        self,
        messages: List[LLMMessage],
        model: str,
        tools: Dict[str, Any]
    ) -> AsyncGenerator[StreamEvent, None]:
        """
        Stream LLM response as stream protocol events.
        Yields StreamEvent objects that follow the Vercel AI stream protocol.
        """
        raise NotImplementedError

    @abstractmethod
    async def generate(
        self,
        messages: List[LLMMessage],
        model: str,
        tools: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Non-streaming generation.
        Returns a dict with structure:
        {
            'done': bool,
            'answer': str (if done),
            'tool': str (if tool call),
            'args': dict (if tool call)
        }
        """
        raise NotImplementedError
