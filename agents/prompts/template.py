"""
Prompt template system with Jinja2 templating and XML-first formatting.

Follows Anthropic's context engineering best practices:
- XML-structured prompts for clear organization
- Minimal, clear formatting
- Variable interpolation for dynamic context
- Environment-based prompt variations
"""
from __future__ import annotations
from typing import Any, Dict, Optional, Union
from jinja2 import Environment, Template as Jinja2Template, StrictUndefined
import textwrap


class PromptTemplate:
    """
    A prompt template with Jinja2 support and XML-first formatting.

    Supports:
    - Environment-based prompts (dev/staging/prod)
    - Variable interpolation
    - XML-structured content
    - Version control

    Example:
        >>> template = PromptTemplate(
        ...     id="chat-agent:v1",
        ...     system=\"\"\"
        ...     <system_instructions>
        ...       <role>You are {{role_name}}, a helpful assistant.</role>
        ...       <guidelines>
        ...         - Be concise and clear
        ...         - Admit when uncertain
        ...       </guidelines>
        ...     </system_instructions>
        ...     \"\"\",
        ...     variables={"role_name": "Alex"}
        ... )
        >>> template.render(role_name="Sarah")
    """

    def __init__(
        self,
        id: str,
        system: Optional[Union[str, Dict[str, str]]] = None,
        environments: Optional[Dict[str, str]] = None,
        variables: Optional[Dict[str, Any]] = None,
        description: Optional[str] = None,
        version: Optional[str] = None
    ):
        """
        Initialize a prompt template.

        Args:
            id: Unique identifier for the prompt (e.g., "chat-agent:v1")
            system: System prompt template (string or dict mapping env -> template)
            environments: Dict mapping environment name -> system prompt template
                         Takes precedence over `system` if provided
            variables: Default variable values for template interpolation
            description: Human-readable description of the prompt's purpose
            version: Optional version string (can also be embedded in id)

        Note: Either `system` or `environments` must be provided.
        """
        self.id = id
        self.description = description
        self.version = version or self._extract_version(id)
        self.variables = variables or {}

        # Setup Jinja2 environment with strict undefined behavior
        self._jinja_env = Environment(
            undefined=StrictUndefined,
            trim_blocks=True,
            lstrip_blocks=True,
            keep_trailing_newline=False
        )

        # Store templates by environment
        self._templates: Dict[str, Jinja2Template] = {}

        # Parse system prompt(s)
        if environments:
            # Multiple environment-specific templates
            for env_name, template_str in environments.items():
                self._templates[env_name] = self._jinja_env.from_string(
                    textwrap.dedent(template_str).strip()
                )
        elif system:
            # Single template or dict of templates
            if isinstance(system, dict):
                for env_name, template_str in system.items():
                    self._templates[env_name] = self._jinja_env.from_string(
                        textwrap.dedent(template_str).strip()
                    )
            else:
                # Single template applies to all environments
                template_obj = self._jinja_env.from_string(
                    textwrap.dedent(system).strip()
                )
                self._templates['default'] = template_obj
        else:
            raise ValueError("Either 'system' or 'environments' must be provided")

    def _extract_version(self, id: str) -> Optional[str]:
        """Extract version from ID like 'agent-name:v1' -> 'v1'"""
        if ':' in id:
            return id.split(':')[-1]
        return None

    def render(
        self,
        environment: str = 'default',
        **variables
    ) -> str:
        """
        Render the template with provided variables.

        Args:
            environment: Environment to render for (dev/staging/prod/default)
            **variables: Variables to interpolate into the template

        Returns:
            Rendered prompt string

        Raises:
            KeyError: If environment doesn't exist
            jinja2.UndefinedError: If required variable is missing
        """
        # Merge default variables with provided ones (provided takes precedence)
        render_vars = {**self.variables, **variables}

        # Get template for environment (fallback to default)
        template = self._templates.get(environment) or self._templates.get('default')

        if not template:
            available = ', '.join(self._templates.keys())
            raise KeyError(
                f"Environment '{environment}' not found in template '{self.id}'. "
                f"Available: {available}"
            )

        return template.render(**render_vars)

    def get_environments(self) -> list[str]:
        """Get list of available environments for this template"""
        return list(self._templates.keys())

    def validate(self, environment: str = 'default', **variables) -> tuple[bool, Optional[str]]:
        """
        Validate that the template can be rendered with given variables.

        Args:
            environment: Environment to validate
            **variables: Variables to test rendering with

        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            self.render(environment=environment, **variables)
            return True, None
        except Exception as e:
            return False, str(e)

    def __repr__(self) -> str:
        envs = ', '.join(self._templates.keys())
        return f"PromptTemplate(id='{self.id}', environments=[{envs}])"


class SystemPromptBuilder:
    """
    Helper for building XML-structured system prompts following Anthropic best practices.

    Example:
        >>> builder = SystemPromptBuilder()
        >>> builder.add_role("You are a helpful customer support assistant")
        >>> builder.add_capabilities([
        ...     "Answer product questions",
        ...     "Handle refund requests"
        ... ])
        >>> builder.add_guidelines([
        ...     "Be empathetic and professional",
        ...     "Always verify customer identity"
        ... ])
        >>> builder.add_context("company_info", "We sell widgets")
        >>> print(builder.build())
    """

    def __init__(self):
        self._sections: Dict[str, str] = {}
        self._role: Optional[str] = None
        self._capabilities: list[str] = []
        self._guidelines: list[str] = []
        self._context: Dict[str, str] = {}

    def add_role(self, role: str) -> 'SystemPromptBuilder':
        """Add role description"""
        self._role = role
        return self

    def add_capabilities(self, capabilities: list[str]) -> 'SystemPromptBuilder':
        """Add list of capabilities"""
        self._capabilities.extend(capabilities)
        return self

    def add_guidelines(self, guidelines: list[str]) -> 'SystemPromptBuilder':
        """Add list of guidelines"""
        self._guidelines.extend(guidelines)
        return self

    def add_context(self, key: str, value: str) -> 'SystemPromptBuilder':
        """Add context section"""
        self._context[key] = value
        return self

    def add_section(self, name: str, content: str) -> 'SystemPromptBuilder':
        """Add custom section"""
        self._sections[name] = content
        return self

    def build(self) -> str:
        """Build the XML-structured system prompt"""
        parts = []

        # System instructions section
        if self._role or self._capabilities or self._guidelines:
            parts.append("<system_instructions>")

            if self._role:
                parts.append(f"  <role>{self._role}</role>")

            if self._capabilities:
                parts.append("  <capabilities>")
                for cap in self._capabilities:
                    parts.append(f"    - {cap}")
                parts.append("  </capabilities>")

            if self._guidelines:
                parts.append("  <guidelines>")
                for guideline in self._guidelines:
                    parts.append(f"    - {guideline}")
                parts.append("  </guidelines>")

            parts.append("</system_instructions>")

        # Context sections
        if self._context:
            parts.append("\n<context>")
            for key, value in self._context.items():
                parts.append(f"  <{key}>")
                parts.append(f"    {value}")
                parts.append(f"  </{key}>")
            parts.append("</context>")

        # Custom sections
        for name, content in self._sections.items():
            parts.append(f"\n<{name}>")
            parts.append(f"  {content}")
            parts.append(f"</{name}>")

        return "\n".join(parts)
