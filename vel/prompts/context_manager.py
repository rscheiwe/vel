"""
Prompt-aware context manager extending the base ContextManager.

Integrates prompt templates with conversation context management.
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional, Union
from ..core.context import ContextManager
from .manager import PromptManager
from .template import PromptTemplate


class PromptContextManager(ContextManager):
    """
    Context manager with integrated prompt template support.

    Extends ContextManager to automatically inject system prompts from templates
    while maintaining all existing context management features.

    Supports two approaches for providing prompts:
    1. Dynamic prompts: Pass a PromptTemplate instance directly (recommended)
    2. Registry prompts: Pass a prompt_id to look up in the global registry

    Example (dynamic prompt - recommended):
        >>> template = PromptTemplate(
        ...     id="chat-agent:v1",
        ...     system="You are {{role_name}}, a helpful assistant."
        ... )
        >>> ctx_mgr = PromptContextManager(
        ...     prompt=template,
        ...     prompt_vars={"role_name": "Alex"},
        ...     max_history=20
        ... )

    Example (registry prompt):
        >>> ctx_mgr = PromptContextManager(
        ...     prompt_id="chat-agent:v1",
        ...     prompt_vars={"role_name": "Alex"},
        ...     prompt_env="prod",
        ...     max_history=20
        ... )
    """

    def __init__(
        self,
        prompt_id: Optional[str] = None,
        prompt_vars: Optional[Dict[str, Any]] = None,
        prompt_env: str = 'prod',
        max_history: Optional[int] = None,
        summarize: bool = False,
        prompt: Optional[PromptTemplate] = None
    ):
        """
        Initialize prompt-aware context manager.

        Args:
            prompt_id: ID of prompt template to look up in registry (legacy approach)
            prompt_vars: Variables for prompt rendering
            prompt_env: Environment for prompt (dev/staging/prod)
            max_history: Maximum number of messages to retain (None = unlimited)
            summarize: Whether to summarize old messages (not yet implemented)
            prompt: PromptTemplate instance to use directly (preferred approach)

        Note:
            If both `prompt` and `prompt_id` are provided, `prompt` takes precedence.
        """
        super().__init__(max_history=max_history, summarize=summarize)

        # Initialize prompt manager if prompt or prompt_id provided
        self.prompt_manager: Optional[PromptManager] = None
        if prompt is not None:
            # Dynamic prompt - use directly (no registration needed!)
            self.prompt_manager = PromptManager(
                template=prompt,
                prompt_vars=prompt_vars,
                environment=prompt_env
            )
        elif prompt_id:
            # Registry lookup
            self.prompt_manager = PromptManager(
                prompt_id=prompt_id,
                prompt_vars=prompt_vars,
                environment=prompt_env
            )

    def messages_for_llm(
        self,
        run_id: str,
        session_id: Optional[str] = None,
        additional_prompt_vars: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Get messages for LLM with system prompt injected.

        Args:
            run_id: Run identifier
            session_id: Optional session identifier
            additional_prompt_vars: Additional variables for prompt rendering

        Returns:
            Message list with system prompt injected (if template exists)
        """
        # Get base messages from parent class
        messages = super().messages_for_llm(run_id, session_id)

        # If no prompt manager, return messages as-is
        if not self.prompt_manager:
            return messages

        # Inject system prompt
        additional_vars = additional_prompt_vars or {}
        messages_with_prompt = self.prompt_manager.inject_system_prompt(
            messages,
            position=0,
            **additional_vars
        )

        return messages_with_prompt

    def update_prompt_vars(self, **new_vars) -> None:
        """
        Update prompt variables.

        Args:
            **new_vars: New variables to merge into prompt_vars

        Raises:
            RuntimeError: If no prompt manager is configured
        """
        if not self.prompt_manager:
            raise RuntimeError("No prompt manager configured. Initialize with prompt_id.")

        self.prompt_manager.update_vars(**new_vars)

    def set_prompt_environment(self, environment: str) -> None:
        """
        Change the prompt environment.

        Args:
            environment: New environment (dev/staging/prod)

        Raises:
            RuntimeError: If no prompt manager is configured
        """
        if not self.prompt_manager:
            raise RuntimeError("No prompt manager configured. Initialize with prompt_id.")

        self.prompt_manager.set_environment(environment)

    def get_rendered_system_prompt(
        self,
        **additional_vars
    ) -> Optional[str]:
        """
        Get the rendered system prompt without message injection.

        Args:
            **additional_vars: Additional variables for rendering

        Returns:
            Rendered system prompt, or None if no template
        """
        if not self.prompt_manager:
            return None

        return self.prompt_manager.render_system_prompt(**additional_vars)

    def has_prompt_template(self) -> bool:
        """Check if a prompt template is configured"""
        return self.prompt_manager is not None and self.prompt_manager.has_template()

    def __repr__(self) -> str:
        base_repr = super().__repr__()
        if self.prompt_manager:
            return f"Prompt{base_repr[:-1]}, prompt_manager={self.prompt_manager})"
        return base_repr
