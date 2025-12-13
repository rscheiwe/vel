"""
Prompt manager for integrating prompt templates with Agent system.

Handles:
- Prompt rendering with environment-based selection
- Variable interpolation
- System message injection
- Integration with ContextManager
"""
from __future__ import annotations
from typing import Any, Dict, Optional
from .template import PromptTemplate
from .registry import PromptRegistry


class PromptManager:
    """
    Manager for prompt templates in Agent execution.

    Bridges PromptTemplate system with Agent's message flow.

    Supports two approaches for providing prompts:
    1. Dynamic prompts: Pass a PromptTemplate instance directly (recommended)
    2. Registry prompts: Pass a prompt_id to look up in the global registry

    Example (dynamic):
        >>> template = PromptTemplate(id='my-prompt', system='You are {{role}}')
        >>> manager = PromptManager(template=template, prompt_vars={'role': 'assistant'})

    Example (registry):
        >>> register_prompt(template)
        >>> manager = PromptManager(prompt_id='my-prompt', prompt_vars={'role': 'assistant'})
    """

    def __init__(
        self,
        prompt_id: Optional[str] = None,
        prompt_vars: Optional[Dict[str, Any]] = None,
        environment: str = 'prod',
        registry: Optional[PromptRegistry] = None,
        template: Optional[PromptTemplate] = None
    ):
        """
        Initialize prompt manager.

        Args:
            prompt_id: ID of prompt template to look up in registry (legacy approach)
            prompt_vars: Variables for prompt rendering
            environment: Environment to use (dev/staging/prod)
            registry: PromptRegistry instance (defaults to singleton)
            template: PromptTemplate instance to use directly (preferred approach)

        Note:
            If both `template` and `prompt_id` are provided, `template` takes precedence.
        """
        self.prompt_vars = prompt_vars or {}
        self.environment = environment
        self.registry = registry or PromptRegistry.default()
        self._template: Optional[PromptTemplate] = None

        # Prefer direct template over registry lookup
        if template is not None:
            # Dynamic prompt - use directly (no registration needed!)
            self._template = template
            self.prompt_id = template.id
        elif prompt_id:
            # Registry lookup
            self.prompt_id = prompt_id
            self._load_template()
        else:
            self.prompt_id = None

    def _load_template(self) -> None:
        """Load template from registry"""
        if not self.prompt_id:
            return

        self._template = self.registry.get_or_none(self.prompt_id)
        if self._template is None:
            raise ValueError(
                f"Prompt template '{self.prompt_id}' not found in registry. "
                f"Available prompts: {', '.join(self.registry.list_ids())}"
            )

    def has_template(self) -> bool:
        """Check if a template is loaded"""
        return self._template is not None

    def render_system_prompt(self, **additional_vars) -> Optional[str]:
        """
        Render the system prompt with variables.

        Args:
            **additional_vars: Additional variables to merge with prompt_vars

        Returns:
            Rendered system prompt, or None if no template loaded
        """
        if not self._template:
            return None

        # Merge variables (additional_vars take precedence)
        render_vars = {**self.prompt_vars, **additional_vars}

        try:
            return self._template.render(
                environment=self.environment,
                **render_vars
            )
        except Exception as e:
            raise ValueError(
                f"Failed to render prompt '{self.prompt_id}' for environment '{self.environment}': {e}"
            )

    def create_system_message(self, **additional_vars) -> Optional[Dict[str, str]]:
        """
        Create a system message dict from the template.

        Args:
            **additional_vars: Additional variables for rendering

        Returns:
            Message dict with role='system', or None if no template
        """
        content = self.render_system_prompt(**additional_vars)
        if content is None:
            return None

        return {'role': 'system', 'content': content}

    def inject_system_prompt(
        self,
        messages: list[Dict[str, Any]],
        position: int = 0,
        **additional_vars
    ) -> list[Dict[str, Any]]:
        """
        Inject system prompt into message list.

        Args:
            messages: Existing message list
            position: Position to insert system message (default: 0 = beginning)
            **additional_vars: Additional variables for rendering

        Returns:
            New message list with system prompt injected
        """
        system_msg = self.create_system_message(**additional_vars)
        if system_msg is None:
            return messages

        # Create a copy to avoid mutation
        messages_copy = list(messages)

        # Check if there's already a system message at the position
        if position < len(messages_copy) and messages_copy[position].get('role') == 'system':
            # Replace existing system message
            messages_copy[position] = system_msg
        else:
            # Insert new system message
            messages_copy.insert(position, system_msg)

        return messages_copy

    def update_vars(self, **new_vars) -> None:
        """
        Update prompt variables.

        Args:
            **new_vars: New variables to merge into prompt_vars
        """
        self.prompt_vars.update(new_vars)

    def set_environment(self, environment: str) -> None:
        """
        Change the environment.

        Args:
            environment: New environment (dev/staging/prod)
        """
        self.environment = environment

    def get_template(self) -> Optional[PromptTemplate]:
        """Get the loaded template"""
        return self._template

    def validate(self) -> tuple[bool, Optional[str]]:
        """
        Validate that the template can be rendered with current variables.

        Returns:
            Tuple of (is_valid, error_message)
        """
        if not self._template:
            return True, None  # No template = valid (no-op)

        return self._template.validate(
            environment=self.environment,
            **self.prompt_vars
        )

    def __repr__(self) -> str:
        return (
            f"PromptManager(prompt_id='{self.prompt_id}', "
            f"environment='{self.environment}', "
            f"has_template={self.has_template()})"
        )
