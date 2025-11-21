"""
Tool Use Behavior - Control flow after tool execution.

This module provides types for controlling agent behavior after tool calls,
including enums for common patterns and custom handler support.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from ..agent import Agent


class ToolUseBehavior(Enum):
    """
    Enum controlling what happens after a tool executes.

    Use via policies:
        policies={'tool_use_behavior': ToolUseBehavior.STOP_AFTER_TOOL}
    """
    RUN_LLM_AGAIN = "run_llm_again"      # Default: continue to next LLM call
    STOP_AFTER_TOOL = "stop_after_tool"  # Stop after any tool executes
    STOP_AT_TOOLS = "stop_at_tools"      # Stop when specific tools execute
    CUSTOM_HANDLER = "custom_handler"    # Use custom_tool_handler callback


class ToolUseDecision(Enum):
    """
    Simple decision returned by custom tool handler.
    """
    CONTINUE = "continue"  # Run LLM again
    STOP = "stop"          # Return result immediately
    ERROR = "error"        # Abort with error


@dataclass
class ToolEvent:
    """
    Event passed to custom tool handler after tool execution.

    Contains all context needed to make a decision about what happens next.
    """
    tool_name: str
    args: Dict[str, Any]
    output: Any
    step: int
    messages: List[Dict[str, Any]]
    run_id: str
    session_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolUseDirective:
    """
    Rich directive returned by custom tool handler for advanced control.

    Allows modifying messages, state, and triggering handoffs without
    exposing internal Effect system.
    """
    decision: ToolUseDecision
    add_messages: Optional[List[Dict[str, Any]]] = None
    replace_messages: Optional[List[Dict[str, Any]]] = None
    handoff_agent: Optional['Agent'] = None
    final_output: Optional[Any] = None  # Override return value when STOP


@dataclass
class HandoffConfig:
    """
    Configuration for agent-to-agent handoffs.

    Controls how context is shared when handing off to another agent.
    """
    target_agent: 'Agent'
    share_context: bool = True
    input_filter: Optional[Any] = None  # Callable[[list], list]


# Type alias for handler return
ToolHandlerResult = Union[ToolUseDecision, ToolUseDirective]
