"""A research / work assistant built on Vel Harness Mode — end-to-end config.

This is the "agent harness" shape applied to knowledge work rather than coding:
a long-horizon agent that plans, gathers, and synthesizes, using a **sandbox as
its durable workspace** for plan tracking + artifacts.

How the harness pieces map to a research assistant
---------------------------------------------------
* **Sandbox = the agent's desk.** It maintains ``plan.md`` (a living checklist it
  edits as it works), drops notes into ``findings/``, and writes the deliverable
  to ``report.md``. Because the sandbox ref rides on the checkpoint, the plan and
  artifacts survive suspend/resume.
* **Compaction** keeps the long research loop inside the context window by
  summarizing old turns — the plan/findings live on disk, not in the prompt.
* **Budget** caps steps/cost so an open-ended task can't run away.
* **Durable HITL approval** gates only *consequential* actions (sending email,
  publishing) — NOT the agent's routine workspace file edits, so plan tracking
  stays friction-free.
* **Skill** injects the research operating instructions + tools for the run.

Run it
------
Local dev (no isolation — uses a tmp dir; fine for trying it out)::

    export OPENAI_API_KEY=...        # or your provider's key
    python examples/harness/research_assistant.py "Compare RAG vs long-context for support bots"

Production isolation: set ``provider='e2b'`` (``pip install 'vel-ai[sandbox-e2b]'``
and an E2B key) instead of the local-subprocess provider below.
"""
from __future__ import annotations

import asyncio
import os
import sys
from functools import lru_cache
from pathlib import Path

# Load examples/.env (git-ignored) so keys live in one place. Zero-dependency.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _env import load_env  # noqa: E402
load_env()

from vel import Agent, ToolSpec  # noqa: E402
from vel.harness import (
    ApprovalConfig,
    CompactionConfig,
    HarnessBudgetConfig,
    HarnessConfig,
    SandboxConfig,
    Skill,
    SkillRef,
    default_registry,
)


# --------------------------------------------------------------------------- #
# Tools the assistant uses beyond the sandbox file tools.
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=1)
def _exa():
    """Lazily build the Exa client (pip install exa-py; needs EXA_API_KEY)."""
    from exa_py import Exa  # optional dep for this example
    return Exa(api_key=os.environ["EXA_API_KEY"])


async def web_search(query: str = "", num_results: int = 8, ctx: dict = None) -> dict:
    """Search the web via Exa and return sources for the agent to read and cite.

    Uses type='auto' (balanced relevance/speed) with highlights — query-relevant
    excerpts that keep token usage predictable (Exa's recommended mode for agent
    workflows). Exa handles search + content extraction end-to-end, so no
    separate page-fetch tool is required for most research tasks.
    """
    exa = _exa()
    # exa-py's client is synchronous; run it off the event loop.
    res = await asyncio.to_thread(
        exa.search,
        query,
        type="auto",
        num_results=num_results,
        contents={"highlights": True},
    )
    return {
        "query": query,
        "results": [
            {
                "title": r.title,
                "url": r.url,
                "highlights": getattr(r, "highlights", None),
            }
            for r in res.results
        ],
    }


async def send_email(to: str = "", subject: str = "", body: str = "", ctx: dict = None) -> dict:
    """Send an email. Consequential -> gated behind durable HITL approval."""
    return {"sent_to": to, "subject": subject}


# --------------------------------------------------------------------------- #
# The "research" skill: operating instructions + tools, activated for the run.
# The instructions are what make the agent *use the sandbox for plan tracking*.
# --------------------------------------------------------------------------- #
RESEARCH_SKILL = Skill(
    name="research",
    description="Plan-driven research/work assistant operating procedure.",
    instructions=(
        "You are a meticulous research assistant. Work in your sandbox workspace "
        "(/workspace) and track your plan on disk so it survives interruptions:\n"
        "1. FIRST, write /workspace/plan.md with a numbered checklist of steps.\n"
        "2. As you complete each step, sandbox_edit plan.md to mark it [x] and add "
        "   brief notes. Re-read plan.md whenever you need to reorient.\n"
        "3. Save raw evidence/notes under /workspace/findings/<topic>.md as you go.\n"
        "4. When the plan is complete, synthesize /workspace/report.md (with sources) "
        "   and give the user a concise summary plus the report path.\n"
        "\n"
        "Searching the web (web_search):\n"
        "- Decompose a broad question into several SPECIFIC queries rather than one "
        "  vague one; issue them across steps and triangulate.\n"
        "- Each result has query-relevant `highlights` and a `url`. Read the "
        "  highlights; record the `url` as the citation for any claim you keep.\n"
        "- Prefer reading your own /workspace/findings notes over re-searching the "
        "  same thing. Cross-check important claims against more than one source.\n"
        "\n"
        "Cite sources (URLs) for every non-obvious claim. Only use send_email after "
        "the user has approved it."
    ),
    tools=[
        ToolSpec.from_function(web_search),
        # send_email is gated by name in ApprovalConfig.require_for_tools below.
        ToolSpec.from_function(send_email),
    ],
)
default_registry.register(RESEARCH_SKILL)


def _sandbox_config() -> SandboxConfig:
    """E2B by default (real cloud isolation; needs E2B_API_KEY). Override with
    VEL_SANDBOX_PROVIDER=local_subprocess for keyless local dev."""
    provider = os.environ.get("VEL_SANDBOX_PROVIDER", "e2b")
    opts: dict = {}
    if provider == "local_subprocess":
        opts["unsafe_local"] = True  # required by the dev-only local provider
    return SandboxConfig(
        enabled=True,
        provider=provider,
        lifecycle="per_session",        # one workspace reused across the session
        workdir="/workspace",
        tools=["read", "write", "edit", "list", "bash", "python"],
        provider_options=opts,
    )


def build_research_assistant() -> Agent:
    harness = HarnessConfig(
        enabled=True,
        durable=True,                       # checkpointable suspend/resume
        budget=HarnessBudgetConfig(
            max_steps=60,                   # long-horizon, but bounded
            max_cost_usd=2.0,
        ),
        compaction=CompactionConfig(
            enabled=True,
            strategy="summarize",           # LLM-summarize old turns
            trigger_token_ratio=0.7,        # compact at 70% of the context window
            keep_last_messages=8,
        ),
        approval=ApprovalConfig(
            enabled=True,
            mode="durable",                 # suspend -> persist -> resume on decision
            require_for_tools=["send_email"],   # gate ONLY consequential actions
            require_for_confirmation_flag=False,  # do NOT gate routine sandbox writes
        ),
        sandbox=_sandbox_config(),
        skills=[SkillRef(name="research")],
    )

    return Agent(
        id="research-assistant",
        model={"provider": "openai", "model": "gpt-4o"},
        harness=harness,
    )


async def main(task: str) -> None:
    agent = build_research_assistant()
    print(f"\n=== Research task: {task} ===\n")
    async for event in agent.run_stream({"message": task}):
        etype = event.get("type", "")
        if etype == "text-delta":
            print(event.get("delta", ""), end="", flush=True)
        elif etype == "tool-input-available":
            print(f"\n  [tool] {event.get('toolName')} {event.get('input')}")
        elif etype == "data-harness-compaction":
            d = event["data"]
            print(f"\n  [compacted {d['before_tokens']}->{d['after_tokens']} tokens]")
        elif etype == "data-harness-sandbox":
            print(f"\n  [sandbox {event['data']['event']}]")
        elif etype == "data-harness-approval-required":
            d = event["data"]
            print(f"\n  [APPROVAL NEEDED] {d['tool_name']} (approval_id={d['approval_id']})")
            print("  -> in a real app, POST the decision and call agent.resume(run_id, [...])")
        elif etype == "data-harness-budget-exhausted":
            print(f"\n  [budget exhausted: {event['data']['reason']}]")
    print("\n")


if __name__ == "__main__":
    question = sys.argv[1] if len(sys.argv) > 1 else "Summarize the state of agent harnesses in 2026."
    asyncio.run(main(question))
