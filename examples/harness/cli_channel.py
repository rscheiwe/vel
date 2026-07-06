"""Run a harness agent as an interactive terminal chat via CLIChannel.

A Channel is a transport adapter over ``RunManager``: it turns an inbound message
into a durable run, streams the run's events back out, and resolves durable
approvals inline. ``CLIChannel`` is the zero-dependency reference adapter — it
prints streamed output and, when a tool needs approval, prompts ``y/N`` on stdin.

Run it::

    export OPENAI_API_KEY=...
    python examples/harness/cli_channel.py
    # then chat; ask it to "delete /tmp/old.log" to trigger an approval prompt.
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

# Load examples/.env (git-ignored) so keys live in one place. Zero-dependency.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _env import load_env  # noqa: E402
load_env()

from vel import Agent, ToolSpec  # noqa: E402
from vel.harness import CLIChannel, HarnessConfig  # noqa: E402


async def delete_file(path: str = "", ctx: dict = None) -> dict:
    return {"deleted": path}


def build_agent() -> Agent:
    return Agent(
        id="cli-assistant",
        model={"provider": "openai", "model": "gpt-4o"},
        tools=[ToolSpec.from_function(delete_file, requires_confirmation=True)],
        harness={"enabled": True, "approval": {"enabled": True, "mode": "durable"}},
    )


async def main() -> None:
    agent = build_agent()
    # One session id ⇒ a continuous conversation (memory + approve-once).
    channel = CLIChannel(agent, harness=agent.harness_config, session_id="cli")
    await channel.repl()


if __name__ == "__main__":
    asyncio.run(main())
