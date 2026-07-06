"""Crash recovery — resume a run that died mid-step without re-running tools.

Harness Mode normally checkpoints once per step. With
``harness.checkpoint_each_tool=True`` it also persists a running checkpoint after
*each* tool result, recording which of the step's tool calls have completed.
After a crash, ``agent.recover(run_id)`` rehydrates the run and re-executes only
the tools that had NOT completed — the finished ones are skipped (their results
are already in the persisted context). This mirrors eve's ``replayed`` guard:
completed side effects are not repeated; only the one tool in flight at crash
time is at-least-once.

This script *simulates* a crash by abandoning the event stream after the first
tool result, then recovers. ``charge_card`` appends to a module-level ledger so
you can see the completed charge is NOT repeated on recovery.

Run it::

    export OPENAI_API_KEY=...
    python examples/harness/crash_recovery.py
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

LEDGER: list[str] = []  # side effects we must NOT duplicate on recovery


async def charge_card(customer: str = "", amount: float = 0.0, ctx: dict = None) -> dict:
    LEDGER.append(f"{customer}:{amount}")
    print(f"  💳 charged {customer} ${amount}  (ledger size={len(LEDGER)})")
    return {"charged": customer, "amount": amount}


def build_agent(db_path: str) -> Agent:
    return Agent(
        id="crash-recovery-demo",
        model={"provider": "openai", "model": "gpt-4o"},
        tools=[ToolSpec.from_function(charge_card)],
        harness={
            "enabled": True,
            "db_path": db_path,
            "checkpoint_each_tool": True,   # per-tool durability (opt-in)
            "approval": {"enabled": False},
        },
    )


async def main() -> None:
    db_path = ".vel/crash_recovery_demo.db"
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    Path(db_path).unlink(missing_ok=True)

    agent = build_agent(db_path)
    task = (
        "Charge these three customers by calling charge_card once each: "
        "alice $10, bob $20, carol $30."
    )

    print("=== Run — then 'crash' after the first step is durably checkpointed ===")
    run_id = None
    charged = False
    async for event in agent.run_stream({"message": task}):
        etype = event.get("type")
        if etype == "data-harness-run-started":
            run_id = event["data"]["run_id"]
        elif etype == "tool-output-available":
            charged = True
        elif etype == "finish-step" and charged:
            # Crash at the step boundary: the first step's tool result is now
            # durably checkpointed (the per-tool checkpoint persists *after* the
            # tool-output event, so we wait for finish-step). Abandon the run.
            print("  ⚡ simulating crash (abandoning the run)")
            break

    print(f"\nLedger after crash: {LEDGER}")

    print("\n=== Recover — completed charges are skipped, not repeated ===")
    async for event in agent.recover(run_id):
        etype = event.get("type")
        if etype == "data-harness-recovered":
            print(f"  ↻ recovered (skipped {event['data']['skipped_tools']} completed tool call(s))")
        elif etype == "text-delta":
            print(event.get("delta", ""), end="", flush=True)

    print(f"\n\nFinal ledger: {LEDGER}")
    print(f"Distinct charges: {sorted(set(LEDGER))}  (no duplicates ⇒ recovery worked)")


if __name__ == "__main__":
    asyncio.run(main())
