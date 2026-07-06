"""Approval policies + approve-once-per-session memory (Harness Mode).

Vel's durable HITL approval can be driven by a **policy predicate** instead of a
static tool-name list: ``(ApprovalContext) -> ApprovalStatus``. The policy sees
the tool name and its actual input, plus the set of tools already approved this
session, so it can:

* auto-approve safe calls (``"approved"``) — run, no human prompt,
* auto-deny obviously-bad calls (``"denied"``) — blocked, no human prompt,
* escalate to a human (``"user-approval"``) — durable suspend/resume,
* abstain (``None``) — fall through to the static ``require_for_tools`` rules.

With ``remember_approvals=True`` (the default), once a tool is approved for a
session it is not re-prompted for the rest of that session (eve's ``approvedTools``).

Run it::

    export OPENAI_API_KEY=...
    python examples/harness/approval_policy.py "delete /etc/hosts then email the team"
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
from vel.harness import ApprovalContext, ApprovalDecision  # noqa: E402


# --------------------------------------------------------------------------- #
# Tools
# --------------------------------------------------------------------------- #
async def read_file(path: str = "", ctx: dict = None) -> dict:
    return {"path": path, "content": f"<contents of {path}>"}


async def delete_path(path: str = "", ctx: dict = None) -> dict:
    return {"deleted": path}


async def send_email(to: str = "", body: str = "", ctx: dict = None) -> dict:
    return {"sent_to": to, "chars": len(body)}


# --------------------------------------------------------------------------- #
# Approval policy — input-aware, with a safe-by-default posture.
# --------------------------------------------------------------------------- #
def approval_policy(ctx: ApprovalContext):
    if ctx.tool_name == "read_file":
        return "approved"                       # reads are always fine
    if ctx.tool_name == "delete_path":
        target = ctx.tool_input.get("path", "")
        if not target.startswith("/workspace/"):
            return "denied"                      # never delete outside the workspace
        return "user-approval"                   # workspace deletes need a human
    if ctx.tool_name == "send_email":
        return "user-approval"                   # outbound email always needs a human
    return None                                   # anything else: fall through


def build_agent() -> Agent:
    return Agent(
        id="approval-policy-demo",
        model={"provider": "openai", "model": "gpt-4o"},
        tools=[
            ToolSpec.from_function(read_file),
            ToolSpec.from_function(delete_path),
            ToolSpec.from_function(send_email),
        ],
        harness={
            "enabled": True,
            "approval": {
                "enabled": True,
                "mode": "durable",
                "policy": approval_policy,
                "remember_approvals": True,   # approve-once-per-session
            },
        },
    )


async def _drive(agent: Agent, task: str, session_id: str) -> None:
    """Run one turn, auto-approving any suspension (a real UI would ask a human)."""
    run_id = None
    pending = []
    async for event in agent.run_stream({"message": task}, session_id=session_id):
        etype = event.get("type")
        if etype == "data-harness-run-started":
            run_id = event["data"]["run_id"]
        elif etype == "text-delta":
            print(event.get("delta", ""), end="", flush=True)
        elif etype == "tool-output-available":
            print(f"\n[tool result] {event['output']}")
        elif etype == "data-harness-approval-required":
            print(f"\n[approval required] {event['data']['tool_name']}")
            pending.append(event["data"])
        elif etype == "data-harness-suspended":
            print("[suspended — awaiting human decision]")

    # Resolve any suspension by approving (demo). Resume continues the run.
    if pending and run_id:
        decisions = [ApprovalDecision(p["approval_id"], "approve") for p in pending]
        print("[human approves] resuming…")
        async for event in agent.resume(run_id, decisions):
            if event.get("type") == "text-delta":
                print(event.get("delta", ""), end="", flush=True)
            elif event.get("type") == "tool-output-available":
                print(f"\n[tool result] {event['output']}")
    print()


async def main(task: str) -> None:
    agent = build_agent()
    session = "demo-session"
    print(f"=== Turn 1: {task}\n")
    await _drive(agent, task, session)
    # Second turn in the SAME session: any tool approved above won't re-prompt.
    print("\n=== Turn 2 (same session): do it again — approved tools are remembered\n")
    await _drive(agent, task, session)


if __name__ == "__main__":
    question = sys.argv[1] if len(sys.argv) > 1 else "delete /workspace/tmp then email ops@example.com a summary"
    asyncio.run(main(question))
