"""Langfuse v4/v5 SDK (OpenTelemetry) tracing for vel 0.6.0. Test author: claude (onto-decisions OG-8).

Langfuse Cloud removes the legacy ingestion endpoint and SDK v2 on 2026-11-16; the v2 handler's `client.trace()`
does not exist in SDK >= 3. These tests drive the real SDK (`langfuse>=4.7,<6`) into an in-memory OTel exporter, so
they need no keys and no network, and assert on what would be exported:

- one run is one Langfuse trace: a root `agent` observation named after the trace name (else the agent id) carrying
  the run's input and output, with the trace attributes (session, user, tags, name) on it; a failed run's root is
  ERROR with the error as its status message;
- every observation of the run shares the trace id; a step is a `span` under the root, a model call a `generation`
  under its step (model, usage, input messages, output), a tool call a `tool` observation `tool:<name>` under its step;
- `log_generation` / `log_tool` return the id of the observation they created (Langfuse assigns ids under OTel; a
  caller-supplied id cannot be honoured);
- a caller-supplied trace id that is already a 32-hex Langfuse trace id is used as is; any other id is turned into
  one deterministically (`Langfuse.create_trace_id(seed=...)`);
- events and scores land on the trace; a Langfuse failure never breaks the run;
- an app can share one Langfuse client across agents (`Agent(observability=..., observability_client=client)`):
  the handler uses it, never makes its own and never shuts it down; `ObservabilityConfig(flush_after_run=False)`
  means a run never flushes the client;
- an installed v2 SDK (no `start_observation`) still gets the legacy handler, so a project pinned to v2 keeps
  tracing until its own migration.
"""

from __future__ import annotations

import json
import uuid
from typing import Any, Dict, List

import pytest
from langfuse import Langfuse
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from vel import Agent, ToolSpec
from vel.events import FinishMessageEvent, TextDeltaEvent, TextEndEvent, TextStartEvent, ToolInputAvailableEvent
from vel.integrations.base import GenerationData, SpanContext, SpanKind, ToolData
from vel.integrations.langfuse import LangfuseHandler, ObservabilityConfig, build_handler
from vel.providers import BaseProvider

TYPE = "langfuse.observation.type"
INPUT = "langfuse.observation.input"
OUTPUT = "langfuse.observation.output"
LEVEL = "langfuse.observation.level"
STATUS = "langfuse.observation.status_message"


class ScriptedProvider(BaseProvider):
    name = "scripted"

    def __init__(self, script: List[List[Any]], fail: bool = False):
        self._script = list(script)
        self._fail = fail

    async def stream(self, messages, model, tools, generation_config=None):
        if self._fail:
            raise RuntimeError("provider exploded")
        batch = self._script.pop(0) if self._script else _text("done")
        for ev in batch:
            yield ev

    async def generate(self, messages, model, tools, generation_config=None):
        if self._fail:
            raise RuntimeError("provider exploded")
        batch = self._script.pop(0) if self._script else _text("done")
        text = "".join(ev.delta for ev in batch if isinstance(ev, TextDeltaEvent))
        return {"done": True, "answer": text, "usage": {"total_tokens": 3}}


def _text(t: str) -> List[Any]:
    return [TextStartEvent(block_id="b"), TextDeltaEvent(block_id="b", delta=t), TextEndEvent(block_id="b"),
            FinishMessageEvent(finish_reason="stop")]


def _tool(cid: str, name: str, args: Dict[str, Any]) -> List[Any]:
    return [ToolInputAvailableEvent(tool_call_id=cid, tool_name=name, input=args),
            FinishMessageEvent(finish_reason="tool_calls")]


async def echo(x: str = "", ctx: dict = None) -> dict:
    return {"echo": x}


ECHO = ToolSpec.from_function(echo, name="echo")


@pytest.fixture
def lf(monkeypatch):
    """A real Langfuse client exporting into memory (no network). The SDK keeps one tracer per public key for the
    process, so each test gets its own key (else spans go to the first test's exporter)."""
    for k in ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "LANGFUSE_HOST", "LANGFUSE_BASE_URL"):
        monkeypatch.delenv(k, raising=False)
    exporter = InMemorySpanExporter()
    client = Langfuse(public_key=f"pk-lf-test-{uuid.uuid4().hex}", secret_key="sk-lf-test", host="http://127.0.0.1:9",
                      span_exporter=exporter, tracer_provider=TracerProvider())
    client.exporter = exporter
    yield client
    try:
        client.shutdown()
    except Exception:
        pass


def _spans(client) -> List[Any]:
    client.flush()
    return list(client.exporter.get_finished_spans())


def _hex(n: int, width: int) -> str:
    return format(n, f"0{width}x")


def _by_type(spans, kind):
    return [s for s in spans if s.attributes.get(TYPE) == kind]


def _config(**kw) -> ObservabilityConfig:
    return ObservabilityConfig(provider="langfuse", **kw)


async def _stream(agent, **kw):
    return [e async for e in agent.run_stream({"message": "hello there"}, **kw)]


def _agent(lf, script, *, tools=(), fail=False, **config) -> Agent:
    agent = Agent(id="helper", model={"provider": "scripted", "model": "m-1"}, tools=list(tools),
                  observability=_config(**config), observability_client=lf)
    agent._custom_provider = ScriptedProvider(script, fail=fail)
    return agent


@pytest.mark.asyncio
async def test_a_run_is_one_trace_with_a_root_agent_observation(lf):
    agent = _agent(lf, [_text("hi back")], session_id="chat-1", user_id="u@x.test", tags=["app"],
                   trace_name="chat-turn")

    await _stream(agent)
    spans = _spans(lf)

    assert spans, "nothing was exported"
    assert len({s.context.trace_id for s in spans}) == 1, "every observation of one run shares one trace"
    roots = _by_type(spans, "agent")
    assert len(roots) == 1
    root = roots[0]
    assert root.name == "chat-turn"
    assert root.attributes["langfuse.trace.name"] == "chat-turn"
    assert root.attributes["session.id"] == "chat-1"
    assert root.attributes["user.id"] == "u@x.test"
    assert "app" in root.attributes["langfuse.trace.tags"]
    assert "hello there" in root.attributes[INPUT]
    assert "hi back" in root.attributes[OUTPUT]


@pytest.mark.asyncio
async def test_the_trace_name_defaults_to_the_agent_id(lf):
    await _stream(_agent(lf, [_text("ok")]))

    (root,) = _by_type(_spans(lf), "agent")
    assert root.name == "helper"
    assert root.attributes["langfuse.trace.name"] == "helper"


@pytest.mark.asyncio
async def test_a_model_call_is_a_generation_under_its_step(lf):
    await _stream(_agent(lf, [_text("answer")]))
    spans = _spans(lf)

    (root,) = _by_type(spans, "agent")
    (gen,) = _by_type(spans, "generation")
    steps = _by_type(spans, "span")
    assert steps, "a step is a span"
    step_ids = {s.context.span_id for s in steps}
    assert gen.parent.span_id in step_ids, "the generation sits under its step"
    assert all(s.parent.span_id == root.context.span_id for s in steps), "steps sit under the root"
    assert gen.attributes["langfuse.observation.model.name"] == "m-1"
    assert "hello there" in gen.attributes[INPUT]
    assert "answer" in gen.attributes[OUTPUT]


@pytest.mark.asyncio
async def test_a_tool_call_is_a_tool_observation_under_its_step(lf):
    await _stream(_agent(lf, [_tool("c1", "echo", {"x": "ping"}), _text("pong")], tools=[ECHO]))
    spans = _spans(lf)

    (tool,) = _by_type(spans, "tool")
    assert tool.name == "tool:echo"
    assert "ping" in tool.attributes[INPUT]
    assert "ping" in tool.attributes[OUTPUT]
    assert tool.parent.span_id in {s.context.span_id for s in _by_type(spans, "span")}
    assert len(_by_type(spans, "generation")) == 2


@pytest.mark.asyncio
async def test_a_failed_run_ends_its_root_as_an_error(lf):
    agent = _agent(lf, [], fail=True)
    try:
        events = await _stream(agent)
    except RuntimeError as exc:  # vel re-raises a provider failure from run_stream
        assert "provider exploded" in str(exc)
    else:
        assert any(e["type"] == "error" for e in events)

    (root,) = _by_type(_spans(lf), "agent")
    assert root.attributes[LEVEL] == "ERROR"
    assert "provider exploded" in root.attributes[STATUS]


@pytest.mark.asyncio
async def test_run_traces_like_run_stream(lf):
    agent = _agent(lf, [_text("plain")], session_id="chat-2")

    assert await agent.run({"message": "hello there"}) == "plain"
    spans = _spans(lf)
    (root,) = _by_type(spans, "agent")
    assert root.attributes["session.id"] == "chat-2"
    assert _by_type(spans, "generation")


@pytest.mark.asyncio
async def test_per_run_context_overrides_the_session_and_user(lf):
    agent = _agent(lf, [_text("ok")], session_id="base", user_id="base-user")

    await _stream(agent, observability_context={"session_id": "chat-9", "user_id": "u9", "tags": ["t9"]})

    (root,) = _by_type(_spans(lf), "agent")
    assert root.attributes["session.id"] == "chat-9"
    assert root.attributes["user.id"] == "u9"
    assert "t9" in root.attributes["langfuse.trace.tags"]


def test_returned_observation_ids_are_the_created_observations(lf):
    handler = LangfuseHandler(_config(), "helper", client=lf)
    root = handler.start_trace(trace_id="run-1", name="helper", input={"message": "x"})
    step = handler.start_span(root, "step-1", SpanKind.STEP)

    gen_id = handler.log_generation(step, GenerationData(model="m", provider="p", messages=[], response="r",
                                                         observation_id="gen-fixed"))
    tool_id = handler.log_tool(step, ToolData(tool_name="t", input={}, output={}, observation_id="tool-fixed"))
    handler.end_span(step)
    handler.end_trace(root, output="done")
    spans = _spans(lf)

    gen = next(s for s in spans if s.attributes.get(TYPE) == "generation")
    tool = next(s for s in spans if s.attributes.get(TYPE) == "tool")
    assert gen_id == _hex(gen.context.span_id, 16)
    assert tool_id == _hex(tool.context.span_id, 16)
    assert step.span_id == _hex(next(s for s in spans if s.name == "step-1").context.span_id, 16)


def test_trace_ids_are_langfuse_trace_ids(lf):
    handler = LangfuseHandler(_config(), "helper", client=lf)
    given = "0123456789abcdef0123456789abcdef"
    a = handler.start_trace(trace_id=given, name="a", input={})
    handler.end_trace(a)
    b = handler.start_trace(trace_id="not-hex-run-id", name="b", input={})
    handler.end_trace(b)
    spans = _spans(lf)

    by_name = {s.name: _hex(s.context.trace_id, 32) for s in spans}
    assert by_name["a"] == given
    assert by_name["b"] == Langfuse.create_trace_id(seed="not-hex-run-id")


def test_usage_and_errors_on_a_generation(lf):
    handler = LangfuseHandler(_config(), "helper", client=lf)
    root = handler.start_trace(trace_id="run-u", name="helper", input={})
    handler.log_generation(root, GenerationData(model="m", provider="p", messages=[{"role": "user", "content": "q"}],
                                                usage={"inputTokens": 10, "outputTokens": 5, "totalTokens": 15},
                                                latency_ms=1200, error="rate limited"))
    handler.end_trace(root)

    (gen,) = _by_type(_spans(lf), "generation")
    assert json.loads(gen.attributes["langfuse.observation.usage_details"]) == {"input": 10, "output": 5,
                                                                               "total": 15}
    assert gen.attributes[LEVEL] == "ERROR"
    assert gen.attributes[STATUS] == "rate limited"
    assert gen.name == "p/m"
    assert gen.parent.span_id == next(s for s in _by_type(_spans(lf), "agent")).context.span_id


def test_events_and_scores_land_on_the_trace(lf, monkeypatch):
    scores = []
    monkeypatch.setattr(lf, "create_score", lambda **kw: scores.append(kw))
    handler = LangfuseHandler(_config(), "helper", client=lf)
    root = handler.start_trace(trace_id="run-e", name="helper", input={})
    handler.log_event(root, "guardrail", {"why": "blocked"}, level="warning")
    handler.set_score(root, "quality", 0.75, comment="fine")
    handler.end_trace(root)
    spans = _spans(lf)

    (event,) = _by_type(spans, "event")
    assert event.name == "guardrail"
    assert event.attributes[LEVEL] == "WARNING"
    assert _hex(event.context.trace_id, 32) == Langfuse.create_trace_id(seed="run-e")
    assert scores == [{"trace_id": Langfuse.create_trace_id(seed="run-e"), "name": "quality", "value": 0.75,
                       "comment": "fine"}]


@pytest.mark.asyncio
async def test_a_langfuse_failure_never_breaks_the_run(lf, monkeypatch):
    def boom(*a, **kw):
        raise RuntimeError("langfuse down")

    monkeypatch.setattr(lf, "start_observation", boom)
    agent = _agent(lf, [_text("still here")])

    events = await _stream(agent)

    assert not any(e["type"] == "error" for e in events)
    assert "still here" in "".join(e.get("delta", "") for e in events if e["type"] == "text-delta")


@pytest.mark.asyncio
async def test_a_shared_client_is_used_and_never_shut_down(lf, monkeypatch):
    made = []
    import langfuse

    monkeypatch.setattr(langfuse, "Langfuse", lambda *a, **kw: made.append(kw))
    calls = []
    monkeypatch.setattr(lf, "shutdown", lambda: calls.append("shutdown"))
    agent = _agent(lf, [_text("ok")])

    await _stream(agent)
    agent._observer.shutdown()

    assert made == [], "the handler made its own client"
    assert calls == []
    assert _by_type(_spans(lf), "agent")


@pytest.mark.asyncio
async def test_flush_after_run_false_never_flushes(lf, monkeypatch):
    flushes = []
    real = lf.flush
    monkeypatch.setattr(lf, "flush", lambda: flushes.append(1))
    await _stream(_agent(lf, [_text("ok")], flush_after_run=False))
    assert flushes == []

    await _stream(_agent(lf, [_text("ok")]))
    assert flushes, "the default still flushes after a run"
    monkeypatch.setattr(lf, "flush", real)


def test_without_a_client_the_handler_makes_one_from_the_config(monkeypatch):
    made = []

    class FakeLangfuse:
        def __init__(self, **kw):
            made.append(kw)

        def start_observation(self, **kw):  # a v4+ SDK
            raise AssertionError("not used")

    import langfuse

    monkeypatch.setattr(langfuse, "Langfuse", FakeLangfuse)
    handler = build_handler(_config(public_key="pk", secret_key="sk", host="https://lf.test", release="1.2"), "a")

    assert isinstance(handler, LangfuseHandler)
    assert made and made[0]["public_key"] == "pk" and made[0]["secret_key"] == "sk"
    assert made[0].get("host") == "https://lf.test" or made[0].get("base_url") == "https://lf.test"


def test_a_v2_sdk_gets_the_legacy_handler(monkeypatch):
    class V2Langfuse:
        def __init__(self, **kw):
            pass

        def trace(self, **kw):  # v2 only: no start_observation
            return None

    import langfuse

    monkeypatch.setattr(langfuse, "Langfuse", V2Langfuse)
    from vel.integrations.langfuse import LegacyLangfuseHandler

    handler = build_handler(_config(public_key="pk", secret_key="sk"), "a")

    assert isinstance(handler, LegacyLangfuseHandler)
    assert not isinstance(handler, LangfuseHandler)


def test_the_langfuse_extra_requires_sdk_4_7_and_below_6():
    import pathlib
    import tomllib

    data = tomllib.loads((pathlib.Path(__file__).parents[1] / "pyproject.toml").read_text())
    extras = data["project"]["optional-dependencies"]
    for name in ("langfuse", "observability"):
        assert extras[name] == ["langfuse>=4.7,<6"], name
    assert "langfuse>=4.7,<6" in extras["dev"], "CI installs .[dev] and must run these tests"


def test_a_passed_v2_client_gets_the_legacy_handler():
    class V2Client:
        def trace(self, **kw):  # v2 only: no start_observation
            return None

    from vel.integrations.langfuse import LegacyLangfuseHandler

    handler = build_handler(_config(), "a", client=V2Client())

    assert isinstance(handler, LegacyLangfuseHandler)
    assert not isinstance(handler, LangfuseHandler)
