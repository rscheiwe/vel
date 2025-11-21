"""
Guardrails System - Input/Output/Tool validation.

Provides validation layers for:
- Input guardrails: Validate user messages before LLM call
- Output guardrails: Validate LLM responses before returning to user
- Tool guardrails: Validate tool arguments before execution
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Callable, Union
import asyncio
import inspect


@dataclass
class GuardrailResult:
    """
    Result of a guardrail check.

    Attributes:
        passed: Whether the guardrail check passed
        message: Error message if failed, or info message if passed
        modified_content: Optional transformed content (for content modification guardrails)
    """
    passed: bool
    message: str = ""
    modified_content: Any = None


class GuardrailError(Exception):
    """Raised when a guardrail check fails."""
    def __init__(self, guardrail_name: str, message: str, content: Any = None):
        self.guardrail_name = guardrail_name
        self.message = message
        self.content = content
        super().__init__(f"Guardrail '{guardrail_name}' failed: {message}")


# Type alias for guardrail functions
# Signature: async def guardrail(content: Any, ctx: dict) -> GuardrailResult
GuardrailFn = Callable[[Any, Dict[str, Any]], Union[GuardrailResult, bool]]


async def run_guardrail(
    guardrail: GuardrailFn,
    content: Any,
    ctx: Dict[str, Any]
) -> GuardrailResult:
    """
    Run a single guardrail function.

    Handles both sync and async guardrails, and normalizes return types.

    Args:
        guardrail: Guardrail function to run
        content: Content to validate
        ctx: Context dict with run_id, session_id, etc.

    Returns:
        GuardrailResult with pass/fail status
    """
    # Call guardrail (async or sync)
    if asyncio.iscoroutinefunction(guardrail):
        result = await guardrail(content, ctx)
    else:
        result = guardrail(content, ctx)

    # Normalize result to GuardrailResult
    if isinstance(result, GuardrailResult):
        return result
    elif isinstance(result, bool):
        return GuardrailResult(passed=result)
    else:
        # Assume it's a GuardrailResult-like dict
        return GuardrailResult(
            passed=result.get('passed', False),
            message=result.get('message', ''),
            modified_content=result.get('modified_content')
        )


async def run_guardrails(
    guardrails: List[GuardrailFn],
    content: Any,
    ctx: Dict[str, Any],
    guardrail_type: str = "guardrail"
) -> tuple[bool, Any, str]:
    """
    Run a list of guardrails sequentially.

    Args:
        guardrails: List of guardrail functions
        content: Content to validate
        ctx: Context dict
        guardrail_type: Type name for error messages (e.g., "input", "output", "tool")

    Returns:
        Tuple of (all_passed, modified_content, error_message)
        - all_passed: True if all guardrails passed
        - modified_content: Content after any modifications
        - error_message: Combined error messages if any failed
    """
    current_content = content
    errors = []

    for i, guardrail in enumerate(guardrails):
        try:
            result = await run_guardrail(guardrail, current_content, ctx)

            if not result.passed:
                # Get guardrail name for error message
                name = getattr(guardrail, '__name__', f'{guardrail_type}_{i}')
                errors.append(f"{name}: {result.message}")
            elif result.modified_content is not None:
                # Apply content modification
                current_content = result.modified_content

        except Exception as e:
            name = getattr(guardrail, '__name__', f'{guardrail_type}_{i}')
            errors.append(f"{name}: {str(e)}")

    if errors:
        return False, current_content, "; ".join(errors)

    return True, current_content, ""


class GuardrailEngine:
    """
    Engine for running guardrail checks on agent inputs/outputs/tools.
    """

    def __init__(
        self,
        input_guardrails: Optional[List[GuardrailFn]] = None,
        output_guardrails: Optional[List[GuardrailFn]] = None,
        tool_guardrails: Optional[Dict[str, List[GuardrailFn]]] = None
    ):
        """
        Initialize guardrail engine.

        Args:
            input_guardrails: List of guardrails to run on user input
            output_guardrails: List of guardrails to run on LLM output
            tool_guardrails: Dict mapping tool names to their guardrails
        """
        self.input_guardrails = input_guardrails or []
        self.output_guardrails = output_guardrails or []
        self.tool_guardrails = tool_guardrails or {}

    async def check_input(
        self,
        content: Any,
        ctx: Dict[str, Any]
    ) -> tuple[bool, Any, str]:
        """
        Run input guardrails on user message.

        Args:
            content: User input (message string or full input dict)
            ctx: Context dict

        Returns:
            Tuple of (passed, modified_content, error_message)
        """
        if not self.input_guardrails:
            return True, content, ""

        return await run_guardrails(
            self.input_guardrails, content, ctx, "input"
        )

    async def check_output(
        self,
        content: Any,
        ctx: Dict[str, Any]
    ) -> tuple[bool, Any, str]:
        """
        Run output guardrails on LLM response.

        Args:
            content: LLM response (text string)
            ctx: Context dict

        Returns:
            Tuple of (passed, modified_content, error_message)
        """
        if not self.output_guardrails:
            return True, content, ""

        return await run_guardrails(
            self.output_guardrails, content, ctx, "output"
        )

    async def check_tool(
        self,
        tool_name: str,
        args: Dict[str, Any],
        ctx: Dict[str, Any]
    ) -> tuple[bool, Dict[str, Any], str]:
        """
        Run tool-specific guardrails on tool arguments.

        Args:
            tool_name: Name of the tool being called
            args: Tool arguments
            ctx: Context dict

        Returns:
            Tuple of (passed, modified_args, error_message)
        """
        guardrails = self.tool_guardrails.get(tool_name, [])
        if not guardrails:
            return True, args, ""

        return await run_guardrails(
            guardrails, args, ctx, f"tool:{tool_name}"
        )

    @property
    def has_input_guardrails(self) -> bool:
        return len(self.input_guardrails) > 0

    @property
    def has_output_guardrails(self) -> bool:
        return len(self.output_guardrails) > 0

    def has_tool_guardrails(self, tool_name: str) -> bool:
        return tool_name in self.tool_guardrails and len(self.tool_guardrails[tool_name]) > 0
