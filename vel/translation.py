"""
Event Translation API for External Integrations

This module provides a clean API for external libraries (like Mesh) to use
Vel's event translation without reimplementing provider-specific logic.
"""
from __future__ import annotations
from typing import AsyncGenerator, Dict, Any, Optional
from .events import StreamEvent
from .providers import ProviderRegistry, BaseProvider


class EventTranslator:
    """
    Translates provider-native events to Vel's stream protocol events.

    This class provides a convenient interface for external libraries to:
    1. Get the appropriate provider translator
    2. Stream events in Vel's standardized format
    3. Access event types and structures

    Example:
        >>> translator = EventTranslator.for_provider("openai")
        >>> async for event in translator.translate_stream(messages, model, tools):
        ...     print(event.to_dict())
    """

    def __init__(self, provider: BaseProvider):
        """Initialize translator with a provider.

        Args:
            provider: The provider instance to use for translation
        """
        self.provider = provider
        self.provider_name = provider.name

    @classmethod
    def for_provider(cls, provider_name: str) -> "EventTranslator":
        """Get an event translator for a specific provider.

        Args:
            provider_name: Name of the provider (e.g., "openai", "anthropic", "google")

        Returns:
            EventTranslator instance configured for that provider

        Raises:
            ValueError: If provider is not available

        Example:
            >>> translator = EventTranslator.for_provider("openai")
        """
        registry = ProviderRegistry.default()
        provider = registry.get(provider_name)
        return cls(provider)

    @classmethod
    def available_providers(cls) -> list[str]:
        """Get list of available provider names.

        Returns:
            List of provider names that can be used with for_provider()

        Example:
            >>> EventTranslator.available_providers()
            ['openai', 'anthropic', 'google']
        """
        registry = ProviderRegistry.default()
        return registry.available()

    async def translate_stream(
        self,
        messages: list[Dict[str, Any]],
        model: str,
        tools: Optional[Dict[str, Any]] = None
    ) -> AsyncGenerator[StreamEvent, None]:
        """
        Translate provider's native stream to Vel stream protocol events.

        This method streams events from the provider and yields them as
        standardized StreamEvent objects following Vel's stream protocol.

        Args:
            messages: Chat messages in format [{"role": "user", "content": "..."}]
            model: Model identifier (e.g., "gpt-4", "claude-3-opus")
            tools: Optional tool definitions (provider-agnostic format)

        Yields:
            StreamEvent objects (TextStartEvent, TextDeltaEvent, etc.)

        Example:
            >>> translator = EventTranslator.for_provider("openai")
            >>> messages = [{"role": "user", "content": "Hello!"}]
            >>> async for event in translator.translate_stream(messages, "gpt-4"):
            ...     if event.type == "text-delta":
            ...         print(event.delta, end="", flush=True)
        """
        async for event in self.provider.stream(messages, model, tools or {}):
            yield event

    async def translate_stream_to_dicts(
        self,
        messages: list[Dict[str, Any]],
        model: str,
        tools: Optional[Dict[str, Any]] = None
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Translate provider stream to dictionary format.

        Same as translate_stream() but yields dictionaries instead of
        StreamEvent objects for easier integration.

        Args:
            messages: Chat messages
            model: Model identifier
            tools: Optional tool definitions

        Yields:
            Event dictionaries in Vel stream protocol format

        Example:
            >>> translator = EventTranslator.for_provider("openai")
            >>> messages = [{"role": "user", "content": "Hello!"}]
            >>> async for event_dict in translator.translate_stream_to_dicts(messages, "gpt-4"):
            ...     print(event_dict)
            {'type': 'text-start', 'id': '...'}
            {'type': 'text-delta', 'id': '...', 'delta': 'Hello'}
        """
        async for event in self.translate_stream(messages, model, tools):
            yield event.to_dict()


def get_translator(provider_name: str) -> EventTranslator:
    """
    Convenience function to get an event translator.

    Args:
        provider_name: Name of the provider (e.g., "openai", "anthropic", "google")

    Returns:
        EventTranslator instance

    Raises:
        ValueError: If provider is not available

    Example:
        >>> from vel.translation import get_translator
        >>> translator = get_translator("openai")
        >>> async for event in translator.translate_stream(messages, "gpt-4"):
        ...     print(event.to_dict())
    """
    return EventTranslator.for_provider(provider_name)


def available_providers() -> list[str]:
    """
    Get list of available provider names.

    Returns:
        List of provider names

    Example:
        >>> from vel.translation import available_providers
        >>> available_providers()
        ['openai', 'anthropic', 'google']
    """
    return EventTranslator.available_providers()


__all__ = [
    'EventTranslator',
    'get_translator',
    'available_providers',
]
