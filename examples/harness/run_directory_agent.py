"""Load an agent from a directory with vel.load_agent and run it.

Filesystem-first authoring (inspired by eve): an agent is a directory of files —
``agent.toml`` (model + harness config), ``instructions.md`` (system prompt),
``tools/*.py`` (one ``tool`` export per file), and ``skills/*.py`` (one ``skill``
export per file). ``load_agent`` compiles them into an ``Agent`` + ``HarnessConfig``.

Run it::

    export OPENAI_API_KEY=...
    python examples/harness/run_directory_agent.py "remember that the launch is May 4"
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

# Load examples/.env (git-ignored) so keys live in one place. Zero-dependency.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _env import load_env  # noqa: E402
load_env()

from vel import load_agent  # noqa: E402

AGENT_DIR = Path(__file__).parent / "agent_dir"


async def main(task: str) -> None:
    agent = load_agent(AGENT_DIR)
    print(f"Loaded agent '{agent.id}' with tools {agent._tool_names}")
    print(f"Skills: {[s.name for s in agent.harness_config.skills]}\n")

    async for event in agent.run_stream({"message": task}):
        etype = event.get("type")
        if etype == "text-delta":
            print(event.get("delta", ""), end="", flush=True)
        elif etype == "tool-output-available":
            print(f"\n[tool result] {event['output']}")
        elif etype == "data-harness-approval-required":
            print(f"\n[approval required] {event['data']['tool_name']} "
                  "(publish is gated — see agent.toml)")
    print()


if __name__ == "__main__":
    question = sys.argv[1] if len(sys.argv) > 1 else "remember that the launch is May 4"
    asyncio.run(main(question))
