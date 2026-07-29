"""
Opt-In Parallel Tools Example

Runs two tool calls in one model step. With the default policy they run
serially; with both ToolSpec(parallel_safe=True) and
policies={'tool_execution': 'parallel'}, their handlers overlap.

This example uses a scripted provider, so it does not need an API key.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

from vel import Agent, ToolSpec
from vel.events import (
    FinishMessageEvent,
    TextDeltaEvent,
    TextEndEvent,
    TextStartEvent,
    ToolInputAvailableEvent,
)
from vel.providers import BaseProvider


DELAY_SECONDS = 1.0


class ScriptedProvider(BaseProvider):
    name = "scripted"

    def __init__(self) -> None:
        self._step = 0

    async def stream(self, messages, model, tools, generation_config=None):
        self._step += 1
        if self._step == 1:
            yield ToolInputAvailableEvent(
                tool_call_id="call_a",
                tool_name="slow_a",
                input={"value": "A"},
            )
            yield ToolInputAvailableEvent(
                tool_call_id="call_b",
                tool_name="slow_b",
                input={"value": "B"},
            )
            yield FinishMessageEvent(finish_reason="tool_calls")
            return

        yield TextStartEvent(block_id="answer")
        yield TextDeltaEvent(block_id="answer", delta="Both tools completed.")
        yield TextEndEvent(block_id="answer")
        yield FinishMessageEvent(finish_reason="stop")

    async def generate(self, messages, model, tools, generation_config=None) -> dict[str, Any]:
        return {}


async def slow_a(value: str = "", ctx: dict | None = None) -> dict:
    await asyncio.sleep(DELAY_SECONDS)
    return {"tool": "slow_a", "value": value}


async def slow_b(value: str = "", ctx: dict | None = None) -> dict:
    await asyncio.sleep(DELAY_SECONDS)
    return {"tool": "slow_b", "value": value}


async def run_case(*, parallel: bool) -> float:
    tools = [
        ToolSpec.from_function(slow_a, name="slow_a", parallel_safe=True),
        ToolSpec.from_function(slow_b, name="slow_b", parallel_safe=True),
    ]
    policies = {"max_steps": 5}
    if parallel:
        policies["tool_execution"] = "parallel"

    agent = Agent(
        id="parallel-demo",
        model={"provider": "scripted", "model": "demo"},
        tools=tools,
        policies=policies,
    )
    agent._custom_provider = ScriptedProvider()

    started = time.monotonic()
    async for event in agent.run_stream({"message": "run both tools"}):
        if event["type"] in {"tool-output-available", "text-delta"}:
            print(event)
    return time.monotonic() - started


async def main() -> None:
    serial = await run_case(parallel=False)
    parallel = await run_case(parallel=True)

    print("-" * 70)
    print(f"default serial: {serial:.2f}s")
    print(f"opt-in parallel: {parallel:.2f}s")


if __name__ == "__main__":
    asyncio.run(main())
