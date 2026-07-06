"""Wire a harness agent into Slack via SlackChannel + a tiny FastAPI host.

``SlackChannel`` is a transport adapter over ``RunManager``: it posts the agent's
replies with ``chat.postMessage`` and renders durable approvals as Block Kit
buttons. As with the FastAPI reference host, Vel stays transport-free — YOUR app
owns the HTTP endpoints. Route:

* Slack **Events API** (message events)  ->  ``channel.handle_message(...)``
* Slack **Interactivity** (button clicks) ->  ``channel.submit_interaction(payload)``

Setup:
1. Create a Slack app; add a bot token scope ``chat:write`` and event ``message.channels``.
2. Enable Interactivity and point both request URLs at this server.
3. ``pip install 'vel-ai[channels-slack]' fastapi uvicorn``
4. Put ``SLACK_BOT_TOKEN`` / ``SLACK_SIGNING_SECRET`` in ``examples/.env``.

Run it::

    uvicorn examples.harness.slack_channel:app --reload

This file is illustrative (signature verification, retries, and threading are
left out for brevity) — not a dependency of Vel.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Load examples/.env (git-ignored) so keys live in one place. Zero-dependency.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _env import load_env  # noqa: E402
load_env()

from vel import Agent, ToolSpec  # noqa: E402
from vel.harness import SlackChannel  # noqa: E402

try:
    from fastapi import FastAPI, Request  # optional, example-only
except ImportError:  # pragma: no cover
    raise SystemExit("This example needs FastAPI: pip install fastapi uvicorn")


async def delete_file(path: str = "", ctx: dict = None) -> dict:
    return {"deleted": path}


def build_agent() -> Agent:
    return Agent(
        id="slack-assistant",
        model={"provider": "openai", "model": "gpt-4o"},
        tools=[ToolSpec.from_function(delete_file, requires_confirmation=True)],
        harness={"enabled": True, "approval": {"enabled": True, "mode": "durable"}},
    )


app = FastAPI(title="Vel Slack channel (reference)")
_agent = build_agent()
# SlackChannel reads SLACK_BOT_TOKEN from the environment by default.
_channel = SlackChannel(_agent, harness=_agent.harness_config)


@app.post("/slack/events")
async def slack_events(request: Request):
    payload = await request.json()
    # URL verification handshake.
    if payload.get("type") == "url_verification":
        return {"challenge": payload["challenge"]}
    event = payload.get("event", {})
    if event.get("type") == "message" and not event.get("bot_id"):
        await _channel.handle_message(channel=event["channel"], text=event.get("text", ""))
    return {"ok": True}


@app.post("/slack/interactivity")
async def slack_interactivity(request: Request):
    form = await request.form()
    import json
    payload = json.loads(form["payload"])
    await _channel.submit_interaction(payload)
    return {"ok": True}
