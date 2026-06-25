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
