"""Base provider interface for LLM providers"""
from __future__ import annotations
from typing import Any, AsyncGenerator, Dict, List, Optional
from abc import ABC, abstractmethod
from ..events import StreamEvent

LLMMessage = Dict[str, Any]
GenerationConfig = Dict[str, Any]

class BaseProvider(ABC):
    """Base interface for LLM providers"""

    name: str

    @abstractmethod
    async def stream(
        self,
        messages: List[LLMMessage],
        model: str,
        tools: Dict[str, Any],
        generation_config: Optional[GenerationConfig] = None
    ) -> AsyncGenerator[StreamEvent, None]:
        """
        Stream LLM response as stream protocol events.
        Yields StreamEvent objects that follow the Vercel AI stream protocol.

        Args:
            messages: Conversation history
            model: Model name/identifier
            tools: Tool schemas
            generation_config: Model generation parameters (temperature, max_tokens, etc.)
                Common parameters:
                - temperature: float (0-2) - Sampling temperature
                - max_tokens: int - Maximum output tokens
                - top_p: float (0-1) - Nucleus sampling
                - top_k: int - Top-K sampling (Gemini, Anthropic)
                - presence_penalty: float (-2 to 2) - Penalize new tokens (OpenAI)
                - frequency_penalty: float (-2 to 2) - Penalize repeated tokens (OpenAI)
                - stop: List[str] - Stop sequences
                - seed: int - Reproducibility seed (OpenAI, Anthropic)
        """
        raise NotImplementedError

    @abstractmethod
    async def generate(
        self,
        messages: List[LLMMessage],
        model: str,
        tools: Dict[str, Any],
        generation_config: Optional[GenerationConfig] = None
    ) -> Dict[str, Any]:
        """
        Non-streaming generation.

        Args:
            messages: Conversation history
            model: Model name/identifier
            tools: Tool schemas
            generation_config: Model generation parameters (see stream() for details)

        Returns:
            Dict with structure:
            {
                'done': bool,
                'answer': str (if done),
                'tool': str (if tool call),
                'args': dict (if tool call)
            }
        """
        raise NotImplementedError
