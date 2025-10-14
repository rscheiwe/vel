"""
Prompt registry for managing prompts across environments and versions.

Follows 12-Factor Agent principles:
- Centralized prompt management
- Environment-based configuration
- Version control support
- Stateless design
"""
from __future__ import annotations
from typing import Dict, List, Optional
from .template import PromptTemplate


class PromptRegistry:
    """
    Registry for managing prompt templates.

    Supports:
    - Prompt registration and lookup by ID
    - Version management
    - Environment filtering
    - Singleton pattern for global access

    Example:
        >>> registry = PromptRegistry.default()
        >>> registry.register(template)
        >>> prompt = registry.get("chat-agent:v1")
    """

    _instance: Optional['PromptRegistry'] = None

    def __init__(self):
        """Initialize an empty registry"""
        self._prompts: Dict[str, PromptTemplate] = {}

    @classmethod
    def default(cls) -> 'PromptRegistry':
        """Get the default singleton instance"""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Reset the singleton instance (useful for testing)"""
        cls._instance = None

    def register(self, template: PromptTemplate) -> None:
        """
        Register a prompt template.

        Args:
            template: PromptTemplate to register

        Raises:
            ValueError: If prompt ID already exists
        """
        if template.id in self._prompts:
            raise ValueError(
                f"Prompt '{template.id}' already registered. "
                f"Use update() to modify existing prompts."
            )
        self._prompts[template.id] = template

    def register_or_update(self, template: PromptTemplate) -> None:
        """
        Register a prompt template, or update if it already exists.

        Args:
            template: PromptTemplate to register or update
        """
        self._prompts[template.id] = template

    def update(self, template: PromptTemplate) -> None:
        """
        Update an existing prompt template.

        Args:
            template: PromptTemplate to update

        Raises:
            KeyError: If prompt ID doesn't exist
        """
        if template.id not in self._prompts:
            raise KeyError(
                f"Prompt '{template.id}' not found. "
                f"Use register() to add new prompts."
            )
        self._prompts[template.id] = template

    def get(self, prompt_id: str) -> PromptTemplate:
        """
        Get a prompt template by ID.

        Args:
            prompt_id: Prompt identifier (e.g., "chat-agent:v1")

        Returns:
            PromptTemplate instance

        Raises:
            KeyError: If prompt ID not found
        """
        if prompt_id not in self._prompts:
            available = ', '.join(self.list_ids())
            raise KeyError(
                f"Prompt '{prompt_id}' not found. Available prompts: {available}"
            )
        return self._prompts[prompt_id]

    def get_or_none(self, prompt_id: str) -> Optional[PromptTemplate]:
        """
        Get a prompt template by ID, or None if not found.

        Args:
            prompt_id: Prompt identifier

        Returns:
            PromptTemplate instance or None
        """
        return self._prompts.get(prompt_id)

    def has(self, prompt_id: str) -> bool:
        """Check if a prompt is registered"""
        return prompt_id in self._prompts

    def remove(self, prompt_id: str) -> None:
        """
        Remove a prompt from the registry.

        Args:
            prompt_id: Prompt identifier

        Raises:
            KeyError: If prompt ID not found
        """
        if prompt_id not in self._prompts:
            raise KeyError(f"Prompt '{prompt_id}' not found")
        del self._prompts[prompt_id]

    def list_ids(self) -> List[str]:
        """Get list of all registered prompt IDs"""
        return list(self._prompts.keys())

    def list_by_prefix(self, prefix: str) -> List[PromptTemplate]:
        """
        Get prompts matching a prefix (useful for versioning).

        Args:
            prefix: Prefix to match (e.g., "chat-agent" matches "chat-agent:v1", "chat-agent:v2")

        Returns:
            List of matching PromptTemplate instances
        """
        return [
            template
            for prompt_id, template in self._prompts.items()
            if prompt_id.startswith(prefix)
        ]

    def list_by_version(self, base_id: str) -> Dict[str, PromptTemplate]:
        """
        Get all versions of a prompt.

        Args:
            base_id: Base prompt ID without version (e.g., "chat-agent")

        Returns:
            Dict mapping version -> PromptTemplate
        """
        versions = {}
        for prompt_id, template in self._prompts.items():
            if ':' in prompt_id:
                base, version = prompt_id.rsplit(':', 1)
                if base == base_id:
                    versions[version] = template
        return versions

    def clear(self) -> None:
        """Clear all registered prompts"""
        self._prompts.clear()

    def count(self) -> int:
        """Get count of registered prompts"""
        return len(self._prompts)

    def __len__(self) -> int:
        return len(self._prompts)

    def __contains__(self, prompt_id: str) -> bool:
        return prompt_id in self._prompts

    def __repr__(self) -> str:
        return f"PromptRegistry(count={len(self._prompts)})"


# Convenience functions for global registry

def register_prompt(template: PromptTemplate) -> None:
    """Register a prompt in the global registry"""
    PromptRegistry.default().register(template)


def get_prompt(prompt_id: str) -> PromptTemplate:
    """Get a prompt from the global registry"""
    return PromptRegistry.default().get(prompt_id)


def has_prompt(prompt_id: str) -> bool:
    """Check if a prompt exists in the global registry"""
    return PromptRegistry.default().has(prompt_id)


def list_prompts() -> List[str]:
    """List all prompt IDs in the global registry"""
    return PromptRegistry.default().list_ids()
