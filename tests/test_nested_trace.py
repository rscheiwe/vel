"""Nested-trace: a nested agent run attaches to the PARENT trace instead of
forking a new one. This is the mechanism the memory-extraction pipeline uses
(a coordinator opens the trace, then runs sub-agents with trace_context)."""

import pytest

from vel.agent import Agent
from vel.providers import BaseProvider
from vel.events import TextStartEvent, TextDeltaEvent, TextEndEvent, FinishMessageEvent
from vel.integrations.base import ObservabilityHandler, SpanContext, SpanKind, TraceContext


class FinalProvider(BaseProvider):
    name = "final"

    def __init__(self, answer="ok"):
        self._answer = answer

    async def generate(self, messages, model, tools, generation_config=None):
        return {"done": True, "answer": self._answer, "usage": {"total_tokens": 7}}

    async def stream(self, messages, model, tools, generation_config=None):
        if False:
            yield None  # pragma: no cover


class StreamProvider(BaseProvider):
    name = "streamer"

    def __init__(self, text="streamed"):
        self._text = text

    async def stream(self, messages, model, tools, generation_config=None):
        yield TextStartEvent(block_id="b")
        yield TextDeltaEvent(block_id="b", delta=self._text)
        yield TextEndEvent(block_id="b")
        yield FinishMessageEvent(finish_reason="stop")

    async def generate(self, messages, model, tools, generation_config=None):
        return {"done": True}


class RecordingHandler(ObservabilityHandler):
    def __init__(self):
        self.traces_started = []
        self.traces_ended = []
        self.spans = []
        self.spans_ended = []
        self.generations = []

    def should_sample(self):
        return True

    def start_trace(self, trace_id, name, input=None, **kwargs):
        self.traces_started.append(trace_id)
        return SpanContext(trace_id=trace_id, span_id=trace_id, name=name, kind=SpanKind.AGENT_RUN)

    def end_trace(self, context, **kwargs):
        self.traces_ended.append(context.trace_id)

    def start_span(self, context, name, kind, input=None, metadata=None, observation_id=None):
        sid = observation_id or f"span-{len(self.spans) + 1}"
        self.spans.append({"trace_id": context.trace_id, "span_id": sid, "parent": context.span_id, "name": name})
        return SpanContext(trace_id=context.trace_id, span_id=sid, parent_span_id=context.span_id, name=name, kind=kind)

    def end_span(self, context, **kwargs):
        self.spans_ended.append(context.span_id)

    def log_generation(self, context, data):
        self.generations.append({"trace_id": context.trace_id, "parent": context.span_id, "obs": data.observation_id})
        return data.observation_id or "gen-1"

    def log_tool(self, context, data):
        return data.observation_id or "tool-1"

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


@pytest.mark.asyncio
async def test_direct_nested_run_attaches_to_parent_trace():
    handler = RecordingHandler()
    root = handler.start_trace(trace_id="T", name="root")
    extract_span = handler.start_span(root, "extract", SpanKind.STEP)

    sub = Agent(id="sub", model={"provider": "final", "model": "m"}, tools=[])
    sub._custom_provider = FinalProvider("hello")

    result = await sub.run({"message": "hi"}, trace_context=TraceContext(handler=handler, span=extract_span))

    assert result == "hello"
    # Exactly ONE trace was started (the parent root) — the sub-agent did NOT fork.
    assert handler.traces_started == ["T"]
    # The sub-agent's generation is logged on the parent handler, under trace "T".
    assert handler.generations, "sub-agent generation not logged on the parent handler"
    assert all(g["trace_id"] == "T" for g in handler.generations)
    # The sub-agent closed only its own span(s); it never ended the parent trace.
    assert handler.traces_ended == []


@pytest.mark.asyncio
async def test_streaming_nested_run_attaches_to_parent_trace():
    """M0: run_stream (not just run) honors trace_context and nests under the
    parent trace instead of forking a new one."""
    handler = RecordingHandler()
    root = handler.start_trace(trace_id="T", name="root")
    parent_span = handler.start_span(root, "extract", SpanKind.STEP)

    sub = Agent(id="sub", model={"provider": "streamer", "model": "m"}, tools=[])
    sub._custom_provider = StreamProvider("hello")

    events = [
        ev async for ev in sub.run_stream(
            {"message": "hi"}, trace_context=TraceContext(handler=handler, span=parent_span)
        )
    ]

    # The stream still produced the normal event envelope.
    assert any(e.get("type") == "text-delta" for e in events)
    # Exactly ONE trace (the parent) — the streaming sub-run did NOT fork.
    assert handler.traces_started == ["T"]
    # Its step generation is logged on the parent handler, under trace "T".
    assert handler.generations, "streaming sub-run generation not logged on parent handler"
    assert all(g["trace_id"] == "T" for g in handler.generations)
    # The child closed only its own span(s); it never ended/flushed the parent.
    assert handler.traces_ended == []


@pytest.mark.asyncio
async def test_reflection_phases_nest_under_one_trace():
    """M2: extended thinking is traced — phases are spans under ONE trace
    (previously reflection emitted no trace at all)."""
    from vel.thinking import ThinkingConfig

    class PhaseProvider(BaseProvider):
        name = "fake"

        def __init__(self, responses):
            self._r = list(responses)

        async def stream(self, messages, model, tools, generation_config=None):
            c = self._r.pop(0) if self._r else "x"
            yield TextStartEvent(block_id="b")
            yield TextDeltaEvent(block_id="b", delta=c)
            yield TextEndEvent(block_id="b")
            yield FinishMessageEvent(finish_reason="stop")

        async def generate(self, messages, model, tools, generation_config=None):
            return {}

    handler = RecordingHandler()
    agent = Agent(id="think", model={"provider": "fake", "model": "m"})
    agent.providers.register(PhaseProvider(["analysis", "critique", "refined\nConfidence: 95%", "answer"]))
    agent._observer = handler

    events = [ev async for ev in agent.run_stream(
        {"message": "q"},
        thinking=ThinkingConfig(mode="reflection", confidence_threshold=0.9, thinking_tools=False),
    )]

    # One owned trace; phases are spans under it; generations all share the trace.
    assert len(handler.traces_started) == 1
    phase_spans = [s for s in handler.spans if s["name"] in ("analyzing", "critiquing", "refining", "concluding")]
    assert len(phase_spans) >= 4, [s["name"] for s in handler.spans]
    assert handler.generations and all(g["trace_id"] == handler.traces_started[0] for g in handler.generations)
    # Per-phase reasoning blocks stream (analyze/critique/refine as reasoning).
    assert sum(1 for e in events if e.get("type") == "reasoning-start") >= 3


@pytest.mark.asyncio
async def test_harness_resume_leg_is_traced(tmp_path):
    """M0 gap (c): a durable resume leg is traced (before, it was invisible)."""
    from vel.tools import ToolSpec
    from vel.harness import ApprovalDecision

    class Scripted(BaseProvider):
        name = "scripted"

        def __init__(self, script):
            self._s = list(script)

        async def stream(self, messages, model, tools, generation_config=None):
            batch = self._s.pop(0) if self._s else [
                TextStartEvent(block_id="e"), TextDeltaEvent(block_id="e", delta="done"),
                TextEndEvent(block_id="e"), FinishMessageEvent(finish_reason="stop"),
            ]
            for ev in batch:
                yield ev

        async def generate(self, messages, model, tools, generation_config=None):
            return {}

    from vel.events import ToolInputAvailableEvent

    async def danger(target: str = "", ctx: dict = None) -> dict:
        return {"deleted": target}

    harness = {
        "enabled": True,
        "db_path": str(tmp_path / "vel.db"),
        "approval": {"enabled": True, "mode": "durable", "require_for_tools": ["danger"]},
    }
    agent = Agent(
        id="m0", model={"provider": "scripted", "model": "m"},
        tools=[ToolSpec.from_function(danger, name="danger")], harness=harness,
    )
    agent._custom_provider = Scripted([
        [ToolInputAvailableEvent(tool_call_id="tc1", tool_name="danger", input={"target": "x"}),
         FinishMessageEvent(finish_reason="tool_calls")],
        [TextStartEvent(block_id="b"), TextDeltaEvent(block_id="b", delta="all done"),
         TextEndEvent(block_id="b"), FinishMessageEvent(finish_reason="stop")],
    ])
    agent._observer = RecordingHandler()  # leg 1 (suspend)

    suspend_events = [ev async for ev in agent.run_stream({"message": "delete x"})]
    appr = next(e for e in suspend_events if e["type"] == "data-harness-approval-required")

    # Fresh handler for the resume leg — proves the resume leg alone is traced.
    resume_handler = RecordingHandler()
    agent._observer = resume_handler
    _ = [ev async for ev in agent.resume(
        appr["data"]["run_id"], [ApprovalDecision(appr["data"]["approval_id"], "approve")]
    )]

    assert resume_handler.traces_started, "resume leg opened no trace (untraced)"
    assert resume_handler.generations, "resume leg logged no LLM generation"


@pytest.mark.asyncio
async def test_non_nested_run_opens_its_own_trace_via_custom_handler():
    # Sanity: without trace_context, behavior is unchanged (agent owns its trace).
    handler = RecordingHandler()
    agent = Agent(id="solo", model={"provider": "final", "model": "m"}, tools=[])
    agent._custom_provider = FinalProvider("done")
    agent._observer = handler  # inject our recording handler as the agent's observer

    result = await agent.run({"message": "hi"})
    assert result == "done"
    assert handler.traces_started == [agent_run_id_of(handler)]
    assert handler.traces_ended == handler.traces_started  # owns end/flush


def agent_run_id_of(handler):
    # the solo run minted its own trace id
    return handler.traces_started[0]
