---
layout: default
title: Channels
nav_order: 15
---

# Channels

A **Channel** is a transport adapter over [`RunManager`](harness#background-runs--reconnect):
it turns an inbound platform message into a durable harness run, streams the
run's events back out, and resolves durable approvals inline. Channels generalize
the FastAPI reference host into a reusable base with concrete adapters (CLI,
Slack). Vel itself stays transport-free — this is optional convenience.

## The `Channel` contract

`Channel` (in `vel.harness.channels`) drives a run over `RunManager`:

- `handle(text, session_id=...)` — start a run for an inbound message and pump it
  to completion (or to a suspension), returning the `run_id`.
- `send(text)` — **abstract**; deliver an outbound message to the platform.
- `request_approval(pending)` — resolve approval-required events. Return a list of
  `ApprovalDecision`s to resume immediately, or `None` to leave the run suspended
  (for platforms that resolve asynchronously via a callback). Default: leave
  suspended.

The shared driver streams the run, renders each event with `format_event`
(text is buffered into one message; tool and `data-harness-*` events become short
notes), collects any approval-required events, and — if the run suspends and the
channel resolves synchronously — calls `request_approval` then `RunManager.resume`
and continues.

## CLI channel

`CLIChannel` is the zero-dependency reference adapter: an interactive terminal
chat. It prints streamed output and prompts `y/N` on stdin for durable approvals.

```python
from vel import Agent, ToolSpec
from vel.harness import CLIChannel

agent = Agent(
    id="cli-assistant",
    model={"provider": "openai", "model": "gpt-4o"},
    tools=[ToolSpec.from_function(delete_file, requires_confirmation=True)],
    harness={"enabled": True, "approval": {"enabled": True, "mode": "durable"}},
)

channel = CLIChannel(agent, harness=agent.harness_config, session_id="cli")
await channel.repl()   # chat loop; /quit to exit
```

Runnable: [`examples/harness/cli_channel.py`](../examples/harness/cli_channel.py).

## Slack channel

`SlackChannel` posts replies with `chat.postMessage` and renders durable
approvals as Block Kit buttons. As with the FastAPI host, **your app owns the
HTTP endpoints**; route Slack's Events API to `handle_message(...)` and its
Interactivity endpoint to `submit_interaction(payload)`:

```python
from vel.harness import SlackChannel

channel = SlackChannel(agent, harness=agent.harness_config)  # reads SLACK_BOT_TOKEN

# Slack message event -> one agent turn
await channel.handle_message(channel="C123", text="delete /tmp/old.log")

# Block Kit button click -> resume the suspended run
await channel.submit_interaction(slack_payload)
```

`slack_sdk` is soft-imported — an absent SDK only errors when a `SlackChannel` is
constructed. Install with `pip install 'vel-ai[channels-slack]'`. A FastAPI host
wiring both endpoints is in
[`examples/harness/slack_channel.py`](../examples/harness/slack_channel.py).

## Writing your own

Subclass `Channel`, implement `send` (and optionally `request_approval`), and
call `handle(text)` from your platform's inbound handler. Everything else —
starting the durable run, streaming, reconnect, approvals — is inherited.
