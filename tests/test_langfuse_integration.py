from datetime import datetime

from vel.integrations.base import GenerationData, SpanContext
from vel.integrations.langfuse import (
    LangfuseHandler,
    TraceState,
    _normalize_usage_details,
)


def test_normalize_usage_details_accepts_supported_usage_shapes():
    assert _normalize_usage_details({
        "prompt_tokens": 1,
        "completion_tokens": 2,
        "total_tokens": 3,
    }) == {"input": 1, "output": 2, "total": 3}
    assert _normalize_usage_details({
        "promptTokens": 4,
        "completionTokens": 5,
        "totalTokens": 9,
    }) == {"input": 4, "output": 5, "total": 9}
    assert _normalize_usage_details({
        "inputTokens": 6,
        "outputTokens": 7,
        "totalTokens": 13,
    }) == {"input": 6, "output": 7, "total": 13}
    assert _normalize_usage_details({"input": 8, "output": 9}) == {
        "input": 8,
        "output": 9,
        "total": 17,
    }


def test_log_generation_sends_usage_details_and_observation_times():
    class FakeGeneration:
        def __init__(self):
            self.end_kwargs = None

        def end(self, **kwargs):
            self.end_kwargs = kwargs

    class FakeTrace:
        def __init__(self):
            self.generation_kwargs = None
            self.generation_client = FakeGeneration()

        def generation(self, **kwargs):
            self.generation_kwargs = kwargs
            return self.generation_client

    handler = LangfuseHandler.__new__(LangfuseHandler)
    handler.config = type(
        "Config",
        (),
        {"capture_input": True, "capture_output": True},
    )()
    handler._langfuse_traces = {"trace-1": FakeTrace()}
    handler._traces = {"trace-1": TraceState(trace_id="trace-1", agent_id="agent-1")}

    handler.log_generation(
        SpanContext(trace_id="trace-1", span_id="step-1"),
        GenerationData(
            model="gpt-4o",
            provider="openai",
            messages=[{"role": "user", "content": "hello"}],
            response="hi",
            usage={"inputTokens": 10, "outputTokens": 5, "totalTokens": 15},
            latency_ms=1200,
            start_time=1_700_000_000.0,
            end_time=1_700_000_001.2,
        ),
    )

    trace = handler._langfuse_traces["trace-1"]
    assert trace.generation_kwargs["usage_details"] == {
        "input": 10,
        "output": 5,
        "total": 15,
    }
    assert isinstance(trace.generation_kwargs["start_time"], datetime)
    assert isinstance(trace.generation_client.end_kwargs["end_time"], datetime)
    assert trace.generation_kwargs["metadata"]["latency_ms"] == 1200
    assert handler._traces["trace-1"].total_tokens == {
        "input": 10,
        "output": 5,
        "total": 15,
    }


class _FakeSpan:
    def end(self, **kwargs):
        self.end_kwargs = kwargs


class _RecordingTrace:
    def __init__(self):
        self.generation_kwargs = None
        self.span_kwargs = None
        self._gen = _FakeSpan()
        self._span = _FakeSpan()

    def generation(self, **kwargs):
        self.generation_kwargs = kwargs
        return self._gen

    def span(self, **kwargs):
        self.span_kwargs = kwargs
        return self._span


def _handler_with_trace(trace):
    handler = LangfuseHandler.__new__(LangfuseHandler)
    handler.config = type("Config", (), {"capture_input": True, "capture_output": True, "capture_tool_io": True})()
    handler._langfuse_traces = {"trace-1": trace}
    handler._traces = {"trace-1": TraceState(trace_id="trace-1", agent_id="agent-1")}
    return handler


def test_log_generation_returns_observation_id_and_supply_is_honored():
    from vel.integrations.base import GenerationData as _GD, SpanContext as _SC
    trace = _RecordingTrace()
    handler = _handler_with_trace(trace)
    # Auto-generated id is passed to Langfuse AND returned.
    auto_id = handler.log_generation(_SC(trace_id="trace-1", span_id="step-1"), _GD(model="m", provider="openai", messages=[]))
    assert auto_id is not None
    assert trace.generation_kwargs["id"] == auto_id
    # Supplied id is honored verbatim.
    supplied = handler.log_generation(_SC(trace_id="trace-1", span_id="step-1"), _GD(model="m", provider="openai", messages=[], observation_id="gen-fixed"))
    assert supplied == "gen-fixed"
    assert trace.generation_kwargs["id"] == "gen-fixed"


def test_log_tool_returns_observation_id_and_supply_is_honored():
    from vel.integrations.base import ToolData as _TD, SpanContext as _SC
    trace = _RecordingTrace()
    handler = _handler_with_trace(trace)
    auto_id = handler.log_tool(_SC(trace_id="trace-1", span_id="step-1"), _TD(tool_name="t"))
    assert auto_id is not None
    assert trace.span_kwargs["id"] == auto_id
    supplied = handler.log_tool(_SC(trace_id="trace-1", span_id="step-1"), _TD(tool_name="t", observation_id="tool-fixed"))
    assert supplied == "tool-fixed"
    assert trace.span_kwargs["id"] == "tool-fixed"


def test_end_span_sends_datetime_end_time():
    from vel.integrations.base import SpanContext as _SC
    trace = _RecordingTrace()
    handler = _handler_with_trace(trace)
    handler.end_span(_SC(trace_id="trace-1", span_id="span-1"), output="done")
    assert isinstance(trace.span_kwargs["end_time"], datetime)
