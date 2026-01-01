"""
Langfuse integration for Vel observability.

Provides full tracing for agent executions including:
- LLM generations with token usage and cost tracking
- Tool execution spans with inputs/outputs
- Memory operation spans (FactStore, ReasoningBank)
- Nested step hierarchy

Langfuse Concepts Mapped to Vel:
- Trace -> Agent run (run_id)
- Span -> Step, tool execution, memory operation
- Generation -> LLM call with model, messages, response, usage
- Event -> Discrete occurrences (errors, guardrail triggers)
- Score -> Evaluation metrics (LLMJudge results)

Usage:
    from vel import Agent
    from vel.integrations import ObservabilityConfig

    agent = Agent(
        id='my-agent',
        model={'provider': 'openai', 'model': 'gpt-4o'},
        observability=ObservabilityConfig(
            provider='langfuse',
            user_id='user-123',
            tags=['production']
        )
    )

Environment Variables:
    LANGFUSE_PUBLIC_KEY: Langfuse public key
    LANGFUSE_SECRET_KEY: Langfuse secret key
    LANGFUSE_HOST: Langfuse host (optional, for self-hosted)
"""

from __future__ import annotations
import copy
import logging
import random
import time
import uuid
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Literal, Optional

from .base import (
    ObservabilityHandler,
    SpanContext,
    SpanKind,
    GenerationData,
    ToolData,
    MemoryData,
    NoOpHandler,
)

logger = logging.getLogger('vel.integrations.langfuse')


@dataclass
class ObservabilityConfig:
    """
    Configuration for observability integration.

    Vel's observability system captures agent execution traces including
    LLM calls, tool executions, and agent steps with full nested hierarchy.

    Example:
        from vel import Agent
        from vel.integrations import ObservabilityConfig

        # Simple: Enable with defaults (uses LANGFUSE_* env vars)
        agent = Agent(
            id='my-agent',
            model={'provider': 'openai', 'model': 'gpt-4o'},
            observability=ObservabilityConfig(provider='langfuse')
        )

        # Full configuration
        agent = Agent(
            id='my-agent',
            model={'provider': 'openai', 'model': 'gpt-4o'},
            observability=ObservabilityConfig(
                provider='langfuse',
                user_id='user-123',
                session_id='session-abc',
                tags=['production', 'v2'],
                metadata={'team': 'ml-platform'},
                release='1.0.0',
                sample_rate=0.1  # Sample 10% of requests
            )
        )

    Attributes:
        provider: Observability backend ('langfuse' | 'none')
        enabled: Master switch for observability (default: True)

        user_id: User identifier for trace attribution
        session_id: Session identifier for grouping related traces
        trace_name: Custom name for the trace (defaults to agent.id)

        tags: List of string tags for filtering traces
        metadata: Custom key-value metadata attached to traces
        release: Application version/release string

        sample_rate: Probability of tracing (0.0-1.0, default: 1.0)

        capture_input: Whether to capture inputs (default: True)
        capture_output: Whether to capture outputs (default: True)
        capture_tool_io: Whether to capture tool inputs/outputs (default: True)

        public_key: Override LANGFUSE_PUBLIC_KEY env var
        secret_key: Override LANGFUSE_SECRET_KEY env var
        host: Override LANGFUSE_HOST env var
    """

    # Provider selection
    provider: Literal['langfuse', 'none'] = 'langfuse'
    enabled: bool = True

    # User/session tracking
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    trace_name: Optional[str] = None  # Defaults to agent.id

    # Metadata
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    release: Optional[str] = None

    # Sampling
    sample_rate: float = 1.0  # 0.0-1.0

    # Capture controls
    capture_input: bool = True
    capture_output: bool = True
    capture_tool_io: bool = True

    # Langfuse client config (override env vars)
    public_key: Optional[str] = None
    secret_key: Optional[str] = None
    host: Optional[str] = None

    def __post_init__(self):
        """Validate configuration."""
        if self.sample_rate < 0.0 or self.sample_rate > 1.0:
            raise ValueError(f"sample_rate must be 0.0-1.0, got {self.sample_rate}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ObservabilityConfig':
        """Create from dictionary."""
        return cls(**data)

    def with_context(
        self,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        trace_name: Optional[str] = None,
    ) -> 'ObservabilityConfig':
        """
        Create a copy with updated context.

        Useful for per-run context overrides without mutating the original.

        Example:
            # Base config on agent
            agent = Agent(
                id='my-agent',
                observability=ObservabilityConfig(provider='langfuse')
            )

            # Per-request context
            result = await agent.run(
                {'message': 'Hello'},
                observability_context={
                    'user_id': request.user.id,
                    'session_id': request.session.id,
                    'tags': ['premium-user']
                }
            )
        """
        new_config = copy.deepcopy(self)

        if user_id is not None:
            new_config.user_id = user_id
        if session_id is not None:
            new_config.session_id = session_id
        if trace_name is not None:
            new_config.trace_name = trace_name
        if tags is not None:
            new_config.tags = list(set(new_config.tags + tags))
        if metadata is not None:
            new_config.metadata = {**new_config.metadata, **metadata}

        return new_config


@dataclass
class SpanState:
    """Tracks an open span's state for async operations."""
    span_id: str
    span_type: str  # 'span' | 'generation'
    name: str
    parent_id: Optional[str] = None
    start_time: float = field(default_factory=time.time)
    input: Optional[Any] = None
    output: Optional[Any] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    usage: Optional[Dict[str, int]] = None
    model: Optional[str] = None
    error: Optional[str] = None
    ended: bool = False


@dataclass
class TraceState:
    """Tracks all state for a single trace (agent run)."""
    trace_id: str
    agent_id: str
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    start_time: float = field(default_factory=time.time)
    spans: Dict[str, SpanState] = field(default_factory=dict)
    span_stack: List[str] = field(default_factory=list)
    current_step: int = 0
    total_tokens: Dict[str, int] = field(default_factory=lambda: {'input': 0, 'output': 0, 'total': 0})
    metadata: Dict[str, Any] = field(default_factory=dict)
    ended: bool = False

    @property
    def current_span_id(self) -> Optional[str]:
        """Get the current (most recent) span ID for parent assignment."""
        return self.span_stack[-1] if self.span_stack else None

    def push_span(self, span: SpanState) -> None:
        """Push a new span onto the stack."""
        self.spans[span.span_id] = span
        self.span_stack.append(span.span_id)

    def pop_span(self) -> Optional[SpanState]:
        """Pop and return the current span."""
        if not self.span_stack:
            return None
        span_id = self.span_stack.pop()
        return self.spans.get(span_id)


class LangfuseHandler(ObservabilityHandler):
    """
    Langfuse observability handler for Vel agents.

    Manages trace lifecycle and span hierarchy for both streaming
    and non-streaming agent execution.

    Requires langfuse package: pip install langfuse

    Configuration via environment or explicit:
        LANGFUSE_PUBLIC_KEY
        LANGFUSE_SECRET_KEY
        LANGFUSE_HOST (optional, for self-hosted)
    """

    def __init__(self, config: ObservabilityConfig, agent_id: str):
        self.config = config
        self.agent_id = agent_id
        self._client = None
        self._traces: Dict[str, TraceState] = {}
        self._langfuse_traces: Dict[str, Any] = {}  # trace_id -> Langfuse trace object
        self._init_client()

    def _init_client(self):
        """Initialize Langfuse client with lazy loading."""
        if not self.config.enabled or self.config.provider != 'langfuse':
            return

        try:
            from langfuse import Langfuse

            self._client = Langfuse(
                public_key=self.config.public_key,
                secret_key=self.config.secret_key,
                host=self.config.host,
                release=self.config.release,
                enabled=self.config.enabled,
            )
            logger.debug("Langfuse client initialized")
        except ImportError:
            logger.warning(
                "langfuse package not installed. Install with: pip install langfuse"
            )
            self._client = None
        except Exception as e:
            logger.warning(f"Failed to initialize Langfuse client: {e}")
            self._client = None

    def should_sample(self) -> bool:
        """Determine if this run should be traced based on sample_rate."""
        return random.random() < self.config.sample_rate

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
        """Start a new trace for this agent run."""
        # Create internal state
        state = TraceState(
            trace_id=trace_id,
            agent_id=self.agent_id,
            session_id=session_id or self.config.session_id,
            user_id=user_id or self.config.user_id,
            metadata=metadata or {},
        )
        self._traces[trace_id] = state

        # Create Langfuse trace
        if self._client:
            try:
                trace_input = input if self.config.capture_input else {'message': '[redacted]'}
                merged_metadata = {**self.config.metadata, **(metadata or {})}
                merged_tags = list(set(self.config.tags + (tags or [])))

                trace = self._client.trace(
                    id=trace_id,
                    name=name,
                    input=trace_input,
                    metadata=merged_metadata,
                    tags=merged_tags,
                    user_id=state.user_id,
                    session_id=state.session_id,
                )
                self._langfuse_traces[trace_id] = trace
                logger.debug(f"Started trace: {trace_id}")
            except Exception as e:
                logger.warning(f"Failed to start Langfuse trace: {e}")

        return SpanContext(
            trace_id=trace_id,
            span_id=trace_id,
            name=name,
            kind=SpanKind.AGENT_RUN,
            metadata=metadata or {},
            tags=tags or [],
        )

    def end_trace(
        self,
        context: SpanContext,
        output: Optional[Any] = None,
        error: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """End a trace."""
        state = self._traces.get(context.trace_id)
        if not state:
            return

        # Force-close any unclosed spans
        while state.span_stack:
            self._force_end_span(state)

        state.ended = True

        # Update Langfuse trace
        trace = self._langfuse_traces.get(context.trace_id)
        if trace:
            try:
                update_data = {}
                if output is not None and self.config.capture_output:
                    update_data['output'] = output
                if metadata:
                    update_data['metadata'] = {**state.metadata, **metadata}
                if error:
                    update_data['level'] = 'ERROR'
                    update_data['status_message'] = error

                if update_data:
                    trace.update(**update_data)
                logger.debug(f"Ended trace: {context.trace_id}")
            except Exception as e:
                logger.warning(f"Failed to end Langfuse trace: {e}")

        # Cleanup
        del self._traces[context.trace_id]
        if context.trace_id in self._langfuse_traces:
            del self._langfuse_traces[context.trace_id]

    def _force_end_span(self, state: TraceState) -> None:
        """Force-end the current span for error recovery."""
        if not state.span_stack:
            return
        span_id = state.span_stack.pop()
        span = state.spans.get(span_id)
        if span and not span.ended:
            span.ended = True
            logger.debug(f"Force-ended span: {span_id}")

    def start_span(
        self,
        context: SpanContext,
        name: str,
        kind: SpanKind,
        input: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SpanContext:
        """Start a child span within a trace."""
        state = self._traces.get(context.trace_id)
        span_id = str(uuid.uuid4())

        # Create internal state
        span_state = SpanState(
            span_id=span_id,
            span_type='span',
            name=name,
            parent_id=context.span_id,
            input=input,
            metadata=metadata or {},
        )

        if state:
            state.push_span(span_state)

        # Create Langfuse span
        trace = self._langfuse_traces.get(context.trace_id)
        if trace:
            try:
                span_input = input if self.config.capture_input else None
                trace.span(
                    id=span_id,
                    name=name,
                    input=span_input,
                    metadata=metadata or {},
                    parent_observation_id=context.span_id if context.span_id != context.trace_id else None,
                )
                logger.debug(f"Started span: {name} ({span_id})")
            except Exception as e:
                logger.warning(f"Failed to start Langfuse span: {e}")

        return SpanContext(
            trace_id=context.trace_id,
            span_id=span_id,
            parent_span_id=context.span_id,
            name=name,
            kind=kind,
            metadata=metadata or {},
        )

    def end_span(
        self,
        context: SpanContext,
        output: Optional[Any] = None,
        error: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """End a span."""
        state = self._traces.get(context.trace_id)
        if not state:
            return

        span_state = state.spans.get(context.span_id)
        if span_state:
            span_state.ended = True
            span_state.output = output
            span_state.error = error

            # Pop from stack if it's the current span
            if state.span_stack and state.span_stack[-1] == context.span_id:
                state.span_stack.pop()

        # Update Langfuse span
        trace = self._langfuse_traces.get(context.trace_id)
        if trace:
            try:
                span_output = output if self.config.capture_output else None
                end_data = {'end_time': time.time()}
                if span_output is not None:
                    end_data['output'] = span_output
                if metadata:
                    end_data['metadata'] = {**(span_state.metadata if span_state else {}), **metadata}
                if error:
                    end_data['level'] = 'ERROR'
                    end_data['status_message'] = error

                # Use span() with same ID to update
                trace.span(id=context.span_id, **end_data)
                logger.debug(f"Ended span: {context.span_id}")
            except Exception as e:
                logger.warning(f"Failed to end Langfuse span: {e}")

    def log_generation(
        self,
        context: SpanContext,
        data: GenerationData,
    ) -> None:
        """Log an LLM generation."""
        trace = self._langfuse_traces.get(context.trace_id)
        if not trace:
            return

        try:
            # Normalize usage to Langfuse format
            usage = None
            if data.usage:
                usage = {
                    'input': data.usage.get('prompt_tokens', data.usage.get('promptTokens', 0)),
                    'output': data.usage.get('completion_tokens', data.usage.get('completionTokens', 0)),
                    'total': data.usage.get('total_tokens', data.usage.get('totalTokens', 0)),
                }

                # Update trace totals
                state = self._traces.get(context.trace_id)
                if state:
                    state.total_tokens['input'] += usage['input']
                    state.total_tokens['output'] += usage['output']
                    state.total_tokens['total'] += usage['total']

            # Prepare output
            output = data.response
            if data.tool_calls:
                output = {
                    'response': data.response,
                    'tool_calls': data.tool_calls,
                }

            # Prepare input (messages)
            gen_input = data.messages if self.config.capture_input else [{'role': 'user', 'content': '[redacted]'}]
            gen_output = output if self.config.capture_output else '[redacted]'

            # Create generation
            generation = trace.generation(
                name=f"{data.provider}/{data.model}",
                model=data.model,
                input=gen_input,
                output=gen_output,
                usage=usage,
                metadata={
                    'provider': data.provider,
                    'generation_config': data.generation_config,
                    'latency_ms': data.latency_ms,
                },
                level='ERROR' if data.error else 'DEFAULT',
                status_message=data.error,
                parent_observation_id=context.span_id if context.span_id != context.trace_id else None,
            )
            generation.end()
            logger.debug(f"Logged generation: {data.provider}/{data.model}")
        except Exception as e:
            logger.warning(f"Failed to log Langfuse generation: {e}")

    def log_tool(
        self,
        context: SpanContext,
        data: ToolData,
    ) -> None:
        """Log a tool execution."""
        trace = self._langfuse_traces.get(context.trace_id)
        if not trace:
            return

        try:
            tool_input = data.input if self.config.capture_tool_io else {'args': '[redacted]'}
            tool_output = data.output if self.config.capture_tool_io else '[redacted]'

            span = trace.span(
                name=f"tool:{data.tool_name}",
                input=tool_input,
                output=tool_output,
                metadata={
                    'tool_call_id': data.tool_call_id,
                    'latency_ms': data.latency_ms,
                },
                level='ERROR' if data.error else 'DEFAULT',
                status_message=data.error,
                parent_observation_id=context.span_id if context.span_id != context.trace_id else None,
            )
            span.end()
            logger.debug(f"Logged tool: {data.tool_name}")
        except Exception as e:
            logger.warning(f"Failed to log Langfuse tool span: {e}")

    def log_memory(
        self,
        context: SpanContext,
        data: MemoryData,
    ) -> None:
        """Log a memory operation (FactStore, ReasoningBank)."""
        trace = self._langfuse_traces.get(context.trace_id)
        if not trace:
            return

        try:
            span = trace.span(
                name=f"memory:{data.operation}",
                input={'namespace': data.namespace, 'key': data.key} if data.namespace else None,
                output=data.result,
                metadata={
                    'operation': data.operation,
                    'latency_ms': data.latency_ms,
                },
                level='ERROR' if data.error else 'DEFAULT',
                status_message=data.error,
                parent_observation_id=context.span_id if context.span_id != context.trace_id else None,
            )
            span.end()
            logger.debug(f"Logged memory operation: {data.operation}")
        except Exception as e:
            logger.warning(f"Failed to log Langfuse memory span: {e}")

    def log_event(
        self,
        context: SpanContext,
        name: str,
        data: Optional[Dict[str, Any]] = None,
        level: str = "info",
    ) -> None:
        """Log a discrete event within a span."""
        trace = self._langfuse_traces.get(context.trace_id)
        if trace:
            try:
                trace.event(
                    name=name,
                    input=data,
                    level=level.upper(),
                    parent_observation_id=context.span_id if context.span_id != context.trace_id else None,
                )
                logger.debug(f"Logged event: {name}")
            except Exception as e:
                logger.warning(f"Failed to log Langfuse event: {e}")

    def set_score(
        self,
        context: SpanContext,
        name: str,
        value: float,
        comment: Optional[str] = None,
    ) -> None:
        """Record a score/evaluation."""
        if self._client:
            try:
                self._client.score(
                    trace_id=context.trace_id,
                    name=name,
                    value=value,
                    comment=comment,
                )
                logger.debug(f"Set score: {name}={value}")
            except Exception as e:
                logger.warning(f"Failed to set Langfuse score: {e}")

    def flush(self) -> None:
        """
        Ensure all data is sent to Langfuse.

        Note: The Langfuse SDK is non-blocking (async background threads), so this
        flush typically adds negligible latency. However, if network issues cause
        retries, this could block. If latency from observability is observed or
        suspected, consider adding a timeout parameter:

            self._client.flush(timeout=1.0)  # Limit wait to 1 second

        See: https://langfuse.com/docs/sdk/python/low-level-sdk
        """
        if self._client:
            try:
                self._client.flush()
                logger.debug("Flushed Langfuse client")
            except Exception as e:
                logger.warning(f"Failed to flush Langfuse client: {e}")

    def shutdown(self) -> None:
        """Shutdown the Langfuse client."""
        if self._client:
            try:
                self._client.shutdown()
                logger.debug("Shutdown Langfuse client")
            except Exception as e:
                logger.warning(f"Failed to shutdown Langfuse client: {e}")


def build_handler(
    config: ObservabilityConfig,
    agent_id: str
) -> ObservabilityHandler:
    """
    Build the appropriate handler based on configuration.

    Args:
        config: ObservabilityConfig
        agent_id: Agent identifier

    Returns:
        ObservabilityHandler instance
    """
    if not config.enabled or config.provider == 'none':
        return NoOpHandler()

    if config.provider == 'langfuse':
        return LangfuseHandler(config, agent_id)

    # Unknown provider - return no-op
    logger.warning(f"Unknown observability provider: {config.provider}")
    return NoOpHandler()
