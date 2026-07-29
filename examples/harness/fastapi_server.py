"""Reference HTTP layer for Vel Harness Mode (M4).

Framework-agnostic by design — Vel itself stays transport-free; this FastAPI app
is a *reference* host that wires HTTP onto :class:`vel.harness.RunManager`. It is
NOT a dependency of Vel. Run with::

    pip install fastapi uvicorn
    uvicorn examples.harness.fastapi_server:app --reload

Endpoints (mirror spec §6.11):
    POST   /runs                      -> {run_id}            start a durable run
    GET    /runs/{run_id}/stream?cursor=N  (SSE)            reconnectable event stream
    POST   /runs/{run_id}/approvals   {approval_id,decision} resume after a HITL gate
    GET    /runs/{run_id}             -> {status}            run status

The reconnect contract: every event is persisted to ``vel_events`` with a
monotonic id. Clients pass the last id they saw as ``?cursor=N``; the server
replays ``id > N`` then live-tails. Killing the SSE connection mid-run and
reconnecting with the last cursor is therefore lossless.
"""
from __future__ import annotations

import json
from typing import Any, Dict

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from vel import Agent, ToolSpec
from vel.harness import ApprovalDecision, HarnessConfig, RunManager

# ---------------------------------------------------------------------------
# Demo agent. Replace with your own. The `delete_file` tool requires approval,
# so a run that calls it will suspend until POST /runs/{id}/approvals arrives.
# ---------------------------------------------------------------------------
async def delete_file(path: str = '', ctx: dict = None) -> dict:
    return {'deleted': path}


def build_agent() -> Agent:
    return Agent(
        id='harness-demo',
        model={'provider': 'openai', 'model': 'gpt-4o'},
        tools=[ToolSpec.from_function(delete_file, requires_confirmation=True)],
        harness=HarnessConfig(
            enabled=True,
            durable=True,
            approval={'enabled': True, 'mode': 'durable',
                      'require_for_confirmation_flag': True},
        ),
    )


app = FastAPI(title="Vel Harness Mode reference server")

# Single-process default: SQLite event log + in-process pub/sub.
_runs = RunManager()           # owns detached runs + durable event log + pub/sub

# --- Scaling across workers (see docs/harness.md → Scaling across workers) ---
# Multi-worker with durable state + cross-worker LIVE tail, choose ONE:
#   (a) deployment-only: Postgres event log + load-balancer sticky sessions by
#       run_id (no code change beyond the DSN):
#         _runs = RunManager(store_backend="postgres", dsn=os.environ["VEL_PG_DSN"])
#   (b) shared pub/sub so any worker can live-tail any run (pip install vel-ai[harness-redis]):
#         from vel.harness import RedisPubSub
#         _runs = RunManager(
#             store_backend="postgres", dsn=os.environ["VEL_PG_DSN"],
#             pubsub=RedisPubSub(url=os.environ["VEL_REDIS_URL"]),
#         )
# Vel provides the seam; your app picks the backend and your deployment owns
# worker count / load-balancer affinity / Postgres+Redis provisioning.

_agent = build_agent()


@app.post("/runs")
async def start_run(request: Request) -> JSONResponse:
    body: Dict[str, Any] = await request.json()
    run_id = await _runs.start(
        _agent,
        input=body.get('input', {'message': body.get('message', '')}),
        session_id=body.get('session_id'),
        harness=_agent.harness_config,
    )
    return JSONResponse({"run_id": run_id})


@app.get("/runs/{run_id}/stream")
async def stream_run(run_id: str, cursor: int = 0) -> StreamingResponse:
    async def event_source():
        # Replay persisted events after `cursor`, then live-tail (lossless on
        # reconnect). Each SSE frame carries the event id so the client can
        # advance its cursor and resume from exactly where it left off.
        async for event in _runs.stream(run_id, cursor=cursor):
            # RunManager tags each event with its monotonic log id under
            # `_cursor`; echo it as the SSE `id:` so the browser's EventSource
            # sends Last-Event-ID on reconnect (mapped back to ?cursor=).
            eid = event.get("_cursor", "")
            payload = {key: value for key, value in event.items() if key != "_cursor"}
            yield f"id: {eid}\ndata: {json.dumps(payload)}\n\n"
        yield "data: [DONE]\n\n"
    return StreamingResponse(
        event_source(),
        media_type="text/event-stream",
        headers={"x-vercel-ai-ui-message-stream": "v1"},
    )


@app.post("/runs/{run_id}/approvals")
async def decide(run_id: str, request: Request) -> JSONResponse:
    body = await request.json()
    decision = ApprovalDecision(
        approval_id=body["approval_id"],
        decision=body["decision"],          # 'approve' | 'reject'
        note=body.get("note"),
        decided_by=body.get("decided_by"),
    )
    # Resume the suspended run; new events flow to any /stream subscribers and
    # are appended to the durable log for late reconnects.
    await _runs.resume(run_id, [decision])
    return JSONResponse({"ok": True, "run_id": run_id})


@app.get("/runs/{run_id}")
async def status(run_id: str) -> JSONResponse:
    return JSONResponse({"run_id": run_id, "status": await _runs.get_status(run_id)})


# ---------------------------------------------------------------------------
# Browser flow (reconnectable EventSource):
#   1. POST /runs -> { run_id }
#   2. open EventSource("/runs/{run_id}/stream?cursor=0")
#   3. on `data-harness-approval-required`, render the approval card
#      (the matching `tool-input-available` already renders in existing UIs)
#   4. POST /runs/{run_id}/approvals { approval_id, decision }
#   5. keep the same EventSource (or reconnect with the last seen id as cursor)
# ---------------------------------------------------------------------------
