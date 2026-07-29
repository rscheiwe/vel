"""
Cancellation Example

Shows cooperative cancellation with cancel_token. The stream closes the open text
block, emits finish-step, then abort and finish.

This example uses a scripted provider, so it does not need an API key.
"""

from __future__ import annotations

import asyncio
from typing import Any

from vel import Agent
from vel.events import FinishMessageEvent, TextDeltaEvent, TextEndEvent, TextStartEvent
from vel.providers import BaseProvider


class SlowProvider(BaseProvider):
    name = "scripted"

    async def stream(self, messages, model, tools, generation_config=None):
        yield TextStartEvent(block_id="t0")
        for chunk in ["e", "c", "h", "o", " ", "t", "e", "x", "t"]:
            await asyncio.sleep(0.1)
            yield TextDeltaEvent(block_id="t0", delta=chunk)
        yield TextEndEvent(block_id="t0")
        yield FinishMessageEvent(finish_reason="stop")

    async def generate(self, messages, model, tools, generation_config=None) -> dict[str, Any]:
        return {}


async def main() -> None:
    cancel_token = asyncio.Event()
    agent = Agent(
        id="cancel-demo",
        model={"provider": "scripted", "model": "demo"},
        policies={"max_steps": 3},
    )
    agent._custom_provider = SlowProvider()

    seen_deltas = 0
    async for event in agent.run_stream({"message": "stream then cancel"}, cancel_token=cancel_token):
        print(event)
        if event["type"] == "text-delta":
            seen_deltas += 1
            if seen_deltas == 3:
                cancel_token.set()


if __name__ == "__main__":
    asyncio.run(main())
