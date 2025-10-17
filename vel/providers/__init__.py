"""Provider registry and exports"""
from __future__ import annotations
from typing import Dict
from .base import BaseProvider
from .openai import OpenAIProvider, OpenAIResponsesProvider
from .google import GeminiProvider
from .anthropic import AnthropicProvider

# Export translators
from .translators import (
    OpenAIAPITranslator,
    OpenAIResponsesAPITranslator,
    OpenAIAgentsSDKTranslator,
    AnthropicAPITranslator,
    GeminiAPITranslator,
    get_openai_api_translator,
    get_openai_responses_translator,
    get_openai_agents_translator,
    get_anthropic_translator,
    get_gemini_translator,
)

class ProviderRegistry:
    """Registry for LLM providers"""

    def __init__(self):
        self._providers: Dict[str, BaseProvider] = {}
        # Register default providers
        try:
            self._providers['openai'] = OpenAIProvider()
        except (ImportError, ValueError):
            # OpenAI not available or API key not set, skip
            pass
        try:
            self._providers['google'] = GeminiProvider()
        except (ImportError, ValueError):
            # Gemini not available or API key not set, skip
            pass
        try:
            self._providers['anthropic'] = AnthropicProvider()
        except (ImportError, ValueError):
            # Anthropic not available or API key not set, skip
            pass
        try:
            self._providers['openai-responses'] = OpenAIResponsesProvider()
        except (ImportError, ValueError):
            # OpenAI Responses API not available or API key not set, skip
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

__all__ = [
    'BaseProvider',
    'OpenAIProvider',
    'OpenAIResponsesProvider',
    'GeminiProvider',
    'AnthropicProvider',
    'ProviderRegistry',
    # Translators
    'OpenAIAPITranslator',
    'OpenAIResponsesAPITranslator',
    'OpenAIAgentsSDKTranslator',
    'AnthropicAPITranslator',
    'GeminiAPITranslator',
    'get_openai_api_translator',
    'get_openai_responses_translator',
    'get_openai_agents_translator',
    'get_anthropic_translator',
    'get_gemini_translator',
]
