"""
Resume/recover legs must open their observability trace correctly.

`_begin_leg_trace` used to call `observer.start_trace(trace_id=..., name=...)`
without `input`, which is a REQUIRED parameter of `ObservabilityHandler` — so
any run that resumed or recovered died with:

    TypeError: start_trace() missing 1 required positional argument: 'input'

It went unnoticed because it only fires on a leg, not on a first-pass run, so it
presented as an intermittent failure; and because the recording handlers in the
other trace tests declare `input=None`, making the argument optional in exactly
the tests that would have caught it.

The handler here therefore mirrors the ABSTRACT signature exactly. That strictness
is the point of the file — a lenient stub cannot fail this way.
"""

import pytest

from vel.agent import Agent
from vel.integrations.base import ObservabilityHandler, SpanContext, SpanKind
from vel.integrations.langfuse import ObservabilityConfig


class StrictHandler(ObservabilityHandler):
    """A handler whose `start_trace` matches `ObservabilityHandler` exactly."""

    def __init__(self):
        self.started = []

    def should_sample(self):
        return True

    def start_trace(
        self,
        trace_id: str,
        name: str,
        input: dict,
        metadata=None,
        tags=None,
        user_id=None,
        session_id=None,
    ) -> SpanContext:
        self.started.append(
            {
                "trace_id": trace_id,
                "name": name,
                "input": input,
                "metadata": metadata,
                "tags": tags,
                "user_id": user_id,
                "session_id": session_id,
            }
        )
        return SpanContext(
            trace_id=trace_id, span_id=trace_id, name=name, kind=SpanKind.AGENT_RUN
        )

    def end_trace(self, context, **kwargs):
        return None

    def start_span(self, context, name, kind, **kwargs):
        return SpanContext(
            trace_id=context.trace_id, span_id="s", parent_span_id=context.span_id,
            name=name, kind=kind,
        )

    def end_span(self, context, **kwargs):
        return None

    def log_generation(self, context, data):
        return None

    def log_tool(self, context, data):
        return None

    def log_memory(self, context, data):
        return None

    def log_event(self, *args, **kwargs):
        return None

    def set_score(self, *args, **kwargs):
        return None

    def flush(self):
        return None

    async def aflush(self):
        return None


def _agent_with(handler, **obs_kwargs):
    agent = Agent(id="leg-agent", model={"provider": "openai", "model": "x"})
    agent._observer = handler
    agent._observability_config = ObservabilityConfig(**obs_kwargs) if obs_kwargs else None
    return agent


def test_leg_trace_supplies_required_input():
    handler = StrictHandler()
    agent = _agent_with(handler)

    observer, trace_ctx = agent._begin_leg_trace("run-1")

    assert trace_ctx is not None
    assert observer is handler
    assert len(handler.started) == 1
    # The regression: `input` must be PRESENT. A leg continues a run whose input
    # was already recorded, so an empty mapping is correct — but omitting the
    # argument is a TypeError.
    assert handler.started[0]["input"] == {}
    assert handler.started[0]["trace_id"] == "run-1"


def test_leg_trace_falls_back_to_agent_id_for_name():
    handler = StrictHandler()
    agent = _agent_with(handler)

    agent._begin_leg_trace("run-2")

    assert handler.started[0]["name"] == "leg-agent"


def test_leg_trace_carries_observability_config_through():
    # A leg belongs to the same trace as the run it continues, so it must be
    # attributed to the same user and session — otherwise resumed runs detach
    # from their session in the backend.
    handler = StrictHandler()
    agent = _agent_with(
        handler,
        trace_name="custom-trace",
        user_id="u-1",
        session_id="s-1",
        tags=["a"],
        metadata={"k": "v"},
    )

    agent._begin_leg_trace("run-3")

    started = handler.started[0]
    assert started["name"] == "custom-trace"
    assert started["user_id"] == "u-1"
    assert started["session_id"] == "s-1"
    assert started["tags"] == ["a"]
    assert started["metadata"] == {"k": "v"}


def test_no_trace_opened_when_not_sampling():
    class NotSampling(StrictHandler):
        def should_sample(self):
            return False

    handler = NotSampling()
    agent = _agent_with(handler)

    observer, trace_ctx = agent._begin_leg_trace("run-4")

    assert trace_ctx is None
    assert handler.started == []


def test_no_observer_is_not_an_error():
    agent = Agent(id="leg-agent", model={"provider": "openai", "model": "x"})
    agent._observer = None

    observer, trace_ctx = agent._begin_leg_trace("run-5")

    assert observer is None
    assert trace_ctx is None
