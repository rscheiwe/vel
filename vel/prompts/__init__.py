"""
Flexible prompt module for Vel Agent runtime.

Provides:
- Jinja2-based prompt templating with XML formatting
- Environment-based prompt management (dev/staging/prod)
- Version control for prompts
- Context formatting utilities
- Integration with Agent system

Follows:
- 12-Factor Agent principles (Factor 2: Own Your Prompts)
- Anthropic's context engineering best practices
- XML-first structuring for clear context boundaries

Example:
    >>> from vel.prompts import PromptTemplate, register_prompt
    >>>
    >>> template = PromptTemplate(
    ...     id="chat-agent:v1",
    ...     system=\"\"\"
    ...     <system_instructions>
    ...       <role>You are {{role_name}}, a helpful assistant.</role>
    ...     </system_instructions>
    ...     \"\"\",
    ...     variables={"role_name": "Alex"}
    ... )
    >>> register_prompt(template)
    >>>
    >>> from vel import Agent
    >>> agent = Agent(
    ...     id='chat-agent:v1',
    ...     model={'provider': 'anthropic', 'model': 'claude-sonnet-4'},
    ...     prompt_id='chat-agent:v1',
    ...     prompt_vars={'role_name': 'Sarah'}
    ... )
"""

# Core template system
from .template import PromptTemplate, SystemPromptBuilder

# Registry for managing prompts
from .registry import (
    PromptRegistry,
    register_prompt,
    get_prompt,
    has_prompt,
    list_prompts
)

# Prompt manager for Agent integration
from .manager import PromptManager

# Context manager with prompt support
from .context_manager import PromptContextManager

# Formatters and utilities
from .formatters import (
    XMLFormatter,
    MarkdownFormatter,
    ContextCompactor,
    MessageFormatter
)

__all__ = [
    # Template system
    'PromptTemplate',
    'SystemPromptBuilder',

    # Registry
    'PromptRegistry',
    'register_prompt',
    'get_prompt',
    'has_prompt',
    'list_prompts',

    # Manager
    'PromptManager',

    # Context manager
    'PromptContextManager',

    # Formatters
    'XMLFormatter',
    'MarkdownFormatter',
    'ContextCompactor',
    'MessageFormatter',
]
