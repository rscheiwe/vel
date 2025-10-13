"""Provider registry and exports"""
from __future__ import annotations
from typing import Dict
from .base import BaseProvider
from .openai import OpenAIProvider
from .google import GeminiProvider

class ProviderRegistry:
    """Registry for LLM providers"""

    def __init__(self):
        self._providers: Dict[str, BaseProvider] = {}
        # Register default providers
        self._providers['openai'] = OpenAIProvider()
        try:
            self._providers['google'] = GeminiProvider()
        except ImportError:
            # Gemini not available, skip
            pass

    @classmethod
    def default(cls) -> 'ProviderRegistry':
        return cls()

    def register(self, provider: BaseProvider):
        """Register a custom provider"""
        self._providers[provider.name] = provider

    def get(self, name: str) -> BaseProvider:
        """Get a provider by name"""
        if name not in self._providers:
            raise ValueError(f"Provider '{name}' not found. Available: {list(self._providers.keys())}")
        return self._providers[name]

    def available(self) -> list[str]:
        """List available provider names"""
        return list(self._providers.keys())

__all__ = ['BaseProvider', 'OpenAIProvider', 'GeminiProvider', 'ProviderRegistry']
