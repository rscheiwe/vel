"""
Base classes for observability integrations.

This module defines the abstract interface that all observability providers
must implement. This enables support for multiple backends (Langfuse,
OpenTelemetry, Datadog, etc.) with a unified API.

Trace Hierarchy:
    Trace (agent run)
    ├── Span: step-1
    │   ├── Generation: openai/gpt-4o
    │   ├── Span: tool:get_weather
    │   └── Span: memory:fact_put
    ├── Span: step-2
    │   └── Generation: openai/gpt-4o
    └── Span: step-3
        └── Generation: openai/gpt-4o
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class SpanKind(Enum):
    """Semantic span types for observability."""
    AGENT_RUN = "agent_run"          # Top-level agent execution
    LLM_GENERATION = "generation"    # LLM call (maps to Langfuse generation)
    TOOL_EXECUTION = "tool"          # Tool call
    STEP = "step"                    # Agent step (LLM + tools)
    MEMORY = "memory"                # FactStore, ReasoningBank operations
    THINKING = "thinking"            # Extended thinking phases
    GUARDRAIL = "guardrail"          # Input/output validation


@dataclass
class SpanContext:
    """
    Context for a trace span.

    Carries the identifiers needed to maintain parent-child relationships
    across async operations.
    """
    trace_id: str
    span_id: str
    parent_span_id: Optional[str] = None
    name: str = ""
    kind: SpanKind = SpanKind.STEP
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)


@dataclass
class GenerationData:
    """
    Data for LLM generation spans.

    Contains all information about an LLM call including input messages,
    output, token usage, and timing.
    """
    model: str
    provider: str
    messages: List[Dict[str, Any]]
    response: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    usage: Optional[Dict[str, int]] = None  # Provider usage in OpenAI, AI SDK, or Langfuse-style keys.
    generation_config: Optional[Dict[str, Any]] = None
    latency_ms: Optional[float] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    error: Optional[str] = None


@dataclass
class ToolData:
    """
    Data for tool execution spans.

    Contains information about a tool call including inputs, outputs,
    and timing.
    """
    tool_name: str
    tool_call_id: Optional[str] = None
    input: Dict[str, Any] = field(default_factory=dict)
    output: Optional[Any] = None
    error: Optional[str] = None
    latency_ms: Optional[float] = None


@dataclass
class MemoryData:
    """
    Data for memory operation spans.

    Contains information about FactStore or ReasoningBank operations.
    """
    operation: str  # 'fact_put', 'fact_get', 'strategy_retrieval', etc.
    namespace: Optional[str] = None
    key: Optional[str] = None
    result: Optional[Any] = None
    latency_ms: Optional[float] = None
    error: Optional[str] = None


class ObservabilityHandler(ABC):
    """
    Abstract base class for observability providers.

    Implement this interface to add support for new observability backends
    (Langfuse, OpenTelemetry, Datadog, etc.)

    Lifecycle:
        1. start_trace() - Called at agent.run() start
        2. start_span() / end_span() - Called for nested operations
        3. log_generation() - Called for each LLM call
        4. log_tool() - Called for each tool execution
        5. end_trace() - Called at agent.run() completion
        6. flush() - Called to ensure all data is sent

    Example Implementation:
        class MyHandler(ObservabilityHandler):
            def start_trace(self, trace_id, name, input, ...):
                # Create trace in backend
                my_backend.create_trace(id=trace_id, name=name, ...)
                return SpanContext(trace_id=trace_id, span_id=trace_id, name=name)

            def log_generation(self, context, data):
                # Log LLM call to backend
                my_backend.log_generation(
                    trace_id=context.trace_id,
                    model=data.model,
                    usage=data.usage,
                    ...
                )
    """

    @abstractmethod
    def start_trace(
        self,
        trace_id: str,
        name: str,
        input: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> SpanContext:
        """
        Start a new trace (top-level agent execution).

        Args:
            trace_id: Unique identifier for this trace (typically run_id)
            name: Human-readable name for the trace
            input: Input data for the agent run
            metadata: Additional key-value metadata
            tags: List of string tags for filtering
            user_id: User identifier for attribution
            session_id: Session identifier for grouping

        Returns:
            SpanContext for the trace root
        """
        ...

    @abstractmethod
    def end_trace(
        self,
        context: SpanContext,
        output: Optional[Any] = None,
        error: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        End a trace.

        Args:
            context: SpanContext from start_trace
            output: Final output from the agent
            error: Error message if execution failed
            metadata: Additional metadata to add
        """
        ...

    @abstractmethod
    def start_span(
        self,
        context: SpanContext,
        name: str,
        kind: SpanKind,
        input: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SpanContext:
        """
        Start a child span within a trace.

        Args:
            context: Parent SpanContext
            name: Span name (e.g., "step-1", "tool:get_weather")
            kind: Semantic type of span
            input: Input data for this span
            metadata: Additional metadata

        Returns:
            SpanContext for the new span
        """
        ...

    @abstractmethod
    def end_span(
        self,
        context: SpanContext,
        output: Optional[Any] = None,
        error: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        End a span.

        Args:
            context: SpanContext from start_span
            output: Output data from this span
            error: Error message if span failed
            metadata: Additional metadata to add
        """
        ...

    @abstractmethod
    def log_generation(
        self,
        context: SpanContext,
        data: GenerationData,
    ) -> None:
        """
        Log an LLM generation.

        This is specialized for LLM calls and includes model info,
        token usage, and cost tracking.

        Args:
            context: Parent SpanContext (typically a step span)
            data: GenerationData with all LLM call details
        """
        ...

    @abstractmethod
    def log_tool(
        self,
        context: SpanContext,
        data: ToolData,
    ) -> None:
        """
        Log a tool execution.

        Args:
            context: Parent SpanContext (typically a step span)
            data: ToolData with tool call details
        """
        ...

    def log_memory(
        self,
        context: SpanContext,
        data: MemoryData,
    ) -> None:
        """
        Log a memory operation (optional override).

        Default implementation is a no-op. Override to track
        FactStore and ReasoningBank operations.

        Args:
            context: Parent SpanContext
            data: MemoryData with operation details
        """
        pass

    def log_event(
        self,
        context: SpanContext,
        name: str,
        data: Optional[Dict[str, Any]] = None,
        level: str = "info",
    ) -> None:
        """
        Log a discrete event within a span (optional override).

        Default implementation is a no-op. Override to log
        discrete events like guardrail triggers.

        Args:
            context: SpanContext where event occurred
            name: Event name
            data: Event data
            level: Severity level (debug, info, warning, error)
        """
        pass

    def set_score(
        self,
        context: SpanContext,
        name: str,
        value: float,
        comment: Optional[str] = None,
    ) -> None:
        """
        Record a score/evaluation (optional override).

        Default implementation is a no-op. Override to record
        quality scores, LLMJudge results, etc.

        Args:
            context: SpanContext to attach score to
            name: Score name (e.g., "quality", "relevance")
            value: Numeric score value
            comment: Optional explanation
        """
        pass

    def flush(self) -> None:
        """
        Ensure all data is sent to the backend.

        Called at the end of agent execution to flush any
        buffered data. Default implementation is a no-op.
        """
        pass

    async def aflush(self) -> None:
        """
        Async flush for providers that support it.

        Default implementation calls sync flush().
        """
        self.flush()


class NoOpHandler(ObservabilityHandler):
    """
    No-op handler that does nothing.

    Used when observability is disabled or when sampling
    excludes a particular run.
    """

    def start_trace(self, trace_id, name, input, **kwargs) -> SpanContext:
        return SpanContext(trace_id=trace_id, span_id=trace_id, name=name)

    def end_trace(self, context, **kwargs) -> None:
        pass

    def start_span(self, context, name, kind, **kwargs) -> SpanContext:
        import uuid
        return SpanContext(
            trace_id=context.trace_id,
            span_id=str(uuid.uuid4()),
            parent_span_id=context.span_id,
            name=name,
            kind=kind,
        )

    def end_span(self, context, **kwargs) -> None:
        pass

    def log_generation(self, context, data) -> None:
        pass

    def log_tool(self, context, data) -> None:
        pass
