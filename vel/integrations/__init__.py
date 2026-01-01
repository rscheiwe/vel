"""
Vel Integrations - Observability and Tracing

This module provides observability integrations for monitoring and tracing
agent executions.

Supported Providers:
- Langfuse: Full tracing with LLM generations, tool spans, and cost tracking

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
"""

from .base import (
    ObservabilityHandler,
    SpanContext,
    SpanKind,
    GenerationData,
    ToolData,
)
from .langfuse import ObservabilityConfig, LangfuseHandler

__all__ = [
    # Base classes
    'ObservabilityHandler',
    'SpanContext',
    'SpanKind',
    'GenerationData',
    'ToolData',
    # Langfuse
    'ObservabilityConfig',
    'LangfuseHandler',
]
