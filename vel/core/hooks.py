"""
Lifecycle Hooks - Agent execution event callbacks.

Provides hooks for observability, tracing, and custom logic injection
at key points in the agent execution lifecycle.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable
import asyncio
import uuid


@dataclass
class HookEvent:
    """Base class for hook events."""
    run_id: str
    session_id: Optional[str] = None
    trace_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StepStartHookEvent(HookEvent):
    """Emitted at the start of each execution step."""
    step: int = 0


@dataclass
class StepEndHookEvent(HookEvent):
    """Emitted at the end of each execution step."""
    step: int = 0
    duration_ms: float = 0


@dataclass
class ToolCallHookEvent(HookEvent):
    """Emitted before a tool is called."""
    tool_name: str = ""
    args: Dict[str, Any] = field(default_factory=dict)
    step: int = 0


@dataclass
class ToolResultHookEvent(HookEvent):
    """Emitted after a tool returns."""
    tool_name: str = ""
    args: Dict[str, Any] = field(default_factory=dict)
    result: Any = None
    duration_ms: float = 0
    step: int = 0


@dataclass
class LLMRequestHookEvent(HookEvent):
    """Emitted before calling the LLM."""
    messages: List[Dict[str, Any]] = field(default_factory=list)
    model: str = ""
    step: int = 0


@dataclass
class LLMResponseHookEvent(HookEvent):
    """Emitted after LLM responds."""
    response: Any = None
    duration_ms: float = 0
    step: int = 0
    usage: Optional[Dict[str, int]] = None


@dataclass
class FinishHookEvent(HookEvent):
    """Emitted when agent execution completes."""
    result: Any = None
    total_steps: int = 0
    total_duration_ms: float = 0


@dataclass
class ErrorHookEvent(HookEvent):
    """Emitted when an error occurs."""
    error: Exception = None
    error_message: str = ""
    step: int = 0


@dataclass
class RunStartHookEvent(HookEvent):
    """Emitted at the start of agent.run() / run_stream()."""
    input: Dict[str, Any] = field(default_factory=dict)
    agent_id: str = ""


@dataclass
class RunFinallyHookEvent(HookEvent):
    """
    Always emitted at end of run (in finally block).

    This hook is guaranteed to fire even if an error occurred,
    making it useful for cleanup and observability trace completion.
    """
    output: Any = None
    error: Optional[Exception] = None
    total_steps: int = 0
    total_duration_ms: float = 0


# Type alias for hook functions
HookFn = Callable[[HookEvent], None]


class HookRegistry:
    """
    Registry for agent lifecycle hooks.
    """

    def __init__(self, hooks: Optional[Dict[str, HookFn]] = None):
        """
        Initialize hook registry.

        Args:
            hooks: Dict mapping hook names to handler functions
                   Supported hooks:
                   - on_run_start: Emitted at start of run()
                   - on_step_start: Emitted at start of each step
                   - on_step_end: Emitted at end of each step
                   - on_tool_call: Emitted before tool execution
                   - on_tool_result: Emitted after tool returns
                   - on_llm_request: Emitted before LLM call
                   - on_llm_response: Emitted after LLM responds
                   - on_finish: Emitted on successful completion
                   - on_error: Emitted on error
                   - on_run_finally: Always emitted at end (for cleanup)
        """
        self._hooks = hooks or {}
        self._trace_id = str(uuid.uuid4())

    def register(self, hook_name: str, handler: HookFn) -> None:
        """
        Register a hook handler.

        Args:
            hook_name: Name of the hook (e.g., 'on_step_start')
            handler: Handler function to call when hook fires
        """
        self._hooks[hook_name] = handler

    async def emit(self, hook_name: str, event: HookEvent) -> None:
        """
        Emit a hook event.

        Args:
            hook_name: Name of the hook (e.g., 'on_step_start')
            event: Event object to pass to handler
        """
        handler = self._hooks.get(hook_name)
        if handler:
            # Set trace_id on event
            event.trace_id = self._trace_id

            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(event)
                else:
                    handler(event)
            except Exception as e:
                # Log but don't fail agent execution
                import logging
                logger = logging.getLogger('vel.hooks')
                logger.warning(f"Hook '{hook_name}' raised exception: {e}")

    @property
    def trace_id(self) -> str:
        return self._trace_id

    def has_hook(self, hook_name: str) -> bool:
        return hook_name in self._hooks
