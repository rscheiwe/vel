# Harness Mode examples

Reference host wiring HTTP onto Vel's transport-free Harness Mode. Vel does not
depend on FastAPI — this is illustrative.

## Keys / `.env`

Put your API keys in `examples/.env` (git-ignored). Start from the template:

```bash
cp examples/.env.example examples/.env   # then fill in keys
```

The example scripts auto-load `examples/.env` at startup via a tiny zero-dep
loader (`examples/_env.py`); real environment variables always take precedence.

## `fastapi_server.py`

A minimal FastAPI app over `vel.harness.RunManager` demonstrating:

- `POST /runs` — start a durable, detached run
- `GET /runs/{run_id}/stream?cursor=N` — reconnectable SSE (replay-then-tail;
  lossless on disconnect via the monotonic `_cursor`)
- `POST /runs/{run_id}/approvals` — submit a human decision and resume a
  suspended run
- `GET /runs/{run_id}` — run status

Run it:

```bash
pip install fastapi uvicorn
uvicorn examples.harness.fastapi_server:app --reload
```

The example uses SQLite by default. For a Postgres event log, install
`vel-ai[harness-postgres]` and initialize `RunManager(store_backend="postgres",
dsn="postgresql://...")`.

Then:

```bash
curl -X POST localhost:8000/runs -d '{"message":"delete /tmp/old.log"}'
# -> {"run_id":"..."}
curl -N "localhost:8000/runs/<run_id>/stream?cursor=0"
# on data-harness-approval-required:
curl -X POST localhost:8000/runs/<run_id>/approvals \
  -d '{"approval_id":"<id>","decision":"approve"}'
```

## `research_assistant.py`

A complete **research / work assistant** config — the agent-harness pattern
applied to knowledge work. Uses the **sandbox as a durable workspace** for plan
tracking (`plan.md` it edits as it works), `findings/`, and a final `report.md`,
plus compaction (long-horizon), a run budget, a research skill, and HITL approval
gating only consequential actions (`send_email`) — routine workspace edits stay
ungated.

```bash
export OPENAI_API_KEY=...
python examples/harness/research_assistant.py "Compare RAG vs long-context for support bots"
```

Defaults to the **E2B** sandbox (`E2B_API_KEY`) and **Exa** web search
(`EXA_API_KEY`). Install the example deps:

```bash
pip install 'vel-ai[sandbox-e2b,examples]'   # e2b + exa-py
```

For keyless local dev, set `VEL_SANDBOX_PROVIDER=local_subprocess` (no E2B
needed; not real isolation).

## `approval_policy.py`

Driving durable approval with a **policy predicate** instead of a static list.
The policy is input-aware — it auto-approves reads, auto-denies deletes outside
`/workspace`, and escalates workspace-deletes / email to a human. Also shows
`remember_approvals` (approve-once-per-session): a second turn doesn't re-prompt
for an already-approved tool.

```bash
export OPENAI_API_KEY=...
python examples/harness/approval_policy.py "delete /workspace/tmp then email ops@example.com"
```

## `crash_recovery.py`

Per-tool durability + recovery. With `checkpoint_each_tool=True`, a run that dies
mid-step is resumed with `agent.recover(run_id)` — completed tools are skipped,
not re-run. The script simulates a crash after the first `charge_card` and shows
the ledger has no duplicate charges after recovery.

```bash
export OPENAI_API_KEY=...
python examples/harness/crash_recovery.py
```

## `agent_dir/` + `run_directory_agent.py`

Filesystem-first authoring: an agent is a directory — `agent.toml` (model +
harness config), `instructions.md`, `tools/*.py` (one `tool` export each), and
`skills/*.py` (one `skill` export each). `load_agent("agent_dir")` compiles it
into an `Agent` + `HarnessConfig`.

```bash
export OPENAI_API_KEY=...
python examples/harness/run_directory_agent.py "remember that the launch is May 4"
```

## `cli_channel.py`

A **Channel** is a transport adapter over `RunManager`. `CLIChannel` is the
zero-dependency reference: an interactive terminal chat that streams output and
prompts `y/N` for durable approvals.

```bash
export OPENAI_API_KEY=...
python examples/harness/cli_channel.py     # then ask it to "delete /tmp/old.log"
```

## `slack_channel.py`

`SlackChannel` wires the same agent into Slack — replies via `chat.postMessage`,
approvals as Block Kit buttons. A tiny FastAPI host routes Slack's Events API and
Interactivity to the channel. Needs a Slack app + tokens.

```bash
pip install 'vel-ai[channels-slack]' fastapi uvicorn
# set SLACK_BOT_TOKEN / SLACK_SIGNING_SECRET in examples/.env
uvicorn examples.harness.slack_channel:app --reload
```

See [`docs/harness.md`](../../docs/harness.md), [`docs/agent-directory.md`](../../docs/agent-directory.md),
and [`docs/channels.md`](../../docs/channels.md) for the full guides.
