# Harness Mode

Harness Mode gives Vel agents **durable, long-horizon execution**: checkpointable
runs that can suspend (for human approval), survive disconnects and process
restarts, and resume — plus automatic context compaction, a generalized run
budget, an optional rented sandbox, and skills.

It is an **opt-in, default-off bolt-on**. With no `harness=` config, every
existing Vel code path runs exactly as before (same as the RLM and Extended
Thinking subsystems). See the backwards-compatibility contract enforced by
`tests/test_harness/test_backwards_compat_agent.py`.

## Quick start

```python
from vel import Agent, ToolSpec

async def delete_file(path: str = "") -> dict:
    return {"deleted": path}

agent = Agent(
    id="assistant",
    model={"provider": "openai", "model": "gpt-4o"},
    tools=[ToolSpec.from_function(delete_file, requires_confirmation=True)],
    harness={
        "enabled": True,
        "budget": {"max_steps": 50},
        "approval": {"enabled": True, "mode": "durable"},
    },
)

async for event in agent.run_stream({"message": "Clean up temp files"}):
    ...  # same stream events as a normal run, plus additive data-harness-* events
```

`harness` accepts a `HarnessConfig` or a plain dict (coerced, including nested
sub-configs). You can also override per run: `agent.run_stream(..., harness=cfg)`.

## What you get

| Capability | Config | Notes |
|---|---|---|
| Run budget | `budget` (`max_steps`/`max_tokens`/`max_cost_usd`/`max_wallclock_seconds`) | On exhaustion the run synthesizes a partial answer instead of erroring |
| Auto-compaction | `compaction` | `summarize` / `reduce` / `memory_offload`; never severs tool-call/result pairs |
| Durable HITL approval | `approval` | Suspend → persist → resume on a human decision (survives restart) |
| Sandbox-as-tool | `sandbox` | Rented isolation (E2B) or local-subprocess for dev; mutating tools auto-gated |
| Skills | `skills` | Named bundles of instructions + tools activated for the run |
| Background runs + reconnect | `RunManager` | Detached runs, durable event log, lossless SSE reconnect |

## Additive stream events

All new events use the `data-harness-*` convention, so unknown-type-tolerant
consumers (e.g. `useChat`) ignore them and existing UIs keep working:

`data-harness-run-started`, `-step`, `-compaction`, `-approval-required`,
`-suspended`, `-resumed`, `-budget-exhausted`, `-sandbox`, `-run-finished`.

## Durable HITL approval

A tool requires approval when its name is in `approval.require_for_tools` **or**
its `ToolSpec.requires_confirmation` is `True` (and
`require_for_confirmation_flag` is on, the default).

In `mode="durable"` (default), when the model calls such a tool the run:

1. appends the assistant tool-call to context,
2. **snapshots a checkpoint** (messages + pending approvals + budget),
3. emits `data-harness-approval-required` (+ the usual `tool-input-available`
   so existing approval-card UIs render) and `data-harness-suspended`,
4. returns — the run is now suspended, persisted, and safe to resume later.

Resume with the human's decision (in the same process or a brand-new one):

```python
from vel.harness import ApprovalDecision

async for event in agent.resume(run_id, [ApprovalDecision(approval_id, "approve")]):
    ...   # approved tools execute; rejected tools get a deny result; the loop continues
```

State lives in `vel_checkpoints` / `vel_approvals` (SQLite by default at
`.vel/vel.db`; Postgres via alembic migration `0002_harness_mode`). Resume is
refused if the agent's model/tools/config changed (`config_hash` guard); pass
`force=True` to override.

`mode="inline"` preserves the legacy in-process `tool_approval_callback` path
(no suspension).

### Approval policies (predicate) + session memory

Instead of (or alongside) the static `require_for_tools` list, an approval
**policy** decides per call from the tool name, its actual input, and the set of
tools already approved this session:

```python
from vel.harness import ApprovalContext

def policy(ctx: ApprovalContext):
    if ctx.tool_name == "read_file":
        return "approved"                    # auto-approve, no prompt
    if ctx.tool_name == "delete_path":
        if not ctx.tool_input.get("path", "").startswith("/workspace/"):
            return "denied"                  # auto-deny, no prompt
        return "user-approval"               # escalate to a human (suspend)
    return None                              # abstain -> fall through to static rules

agent = Agent(..., harness={
    "enabled": True,
    "approval": {"enabled": True, "mode": "durable", "policy": policy},
})
```

`ApprovalStatus` (the policy's return type) mirrors eve:

| Return | Meaning |
|---|---|
| `"approved"` | run the tool, no human prompt |
| `"denied"` | block the tool (denied result), no human prompt |
| `"user-approval"` / `True` | require a human decision (durable suspend/resume) |
| `"not-applicable"` / `False` | no approval needed |
| `None` | policy abstains — fall through to `require_for_tools` / the confirmation flag |

`ApprovalContext` fields: `tool_name`, `tool_input`, `tool_call_id`, `run_id`,
`session_id`, `step`, `approved_tools` (a `frozenset`), `requires_confirmation`.

**Approve-once-per-session.** With `approval.remember_approvals` (default `True`),
once a tool is approved for a session — by a human decision or an `"approved"`
policy result — it is not re-prompted for the rest of that session (eve's
`approvedTools`). Set it to `False` to prompt on every call. Session memory is
durable (persists across runs and restarts, keyed by `session_id`).

Precedence in the gate: **session memory → policy → static rules**. A runnable
example is in [`examples/harness/approval_policy.py`](../examples/harness/approval_policy.py).

## Crash recovery

Harness Mode checkpoints once per step by default. Set
`harness.checkpoint_each_tool=True` to also persist a running checkpoint after
**each** tool result — recording which of the step's tool calls have completed.
If the process then dies mid-step, recover the run:

```python
async for event in agent.recover(run_id):   # or RunManager.recover(run_id)
    ...   # emits data-harness-recovered, then continues the run
```

`recover` rehydrates the last `running` checkpoint and re-executes only the tools
that had **not** completed — the finished ones are skipped because their results
are already in the persisted context (eve's `replayed` guard). Guarantee boundary:
completed tools are exactly-once; the single tool in flight at crash time is
**at-least-once** (it may re-run), so keep consequential tools idempotent or gate
them behind approval. Without `checkpoint_each_tool`, recovery restarts from the
last step boundary. Example:
[`examples/harness/crash_recovery.py`](../examples/harness/crash_recovery.py).

Parallel tool execution makes the same idempotency rule load-bearing even
without a crash. Only mark a tool `parallel_safe=True` when it can run beside
other tools in the same step without observing or corrupting shared state.

## Auto-compaction

When estimated prompt tokens exceed `compaction.trigger_token_ratio` of the
model's context window, the loop compacts the older, fully-resolved turns before
the next step and emits `data-harness-compaction`. Invariants:

- never compact across an unresolved tool-call/tool-result boundary,
- always keep the last `keep_last_messages` and the most recent user turn.

Strategies: `summarize` (LLM summary), `reduce` (deterministic digest),
`memory_offload` (offload salient facts to memory, falls back to `reduce` if no
memory is configured).

## Background runs + reconnectable stream

For long browser tasks, drive runs detached via `RunManager` and attach the
browser to a durable, reconnectable event log rather than the loop's lifetime:

```python
from vel.harness import RunManager, ApprovalDecision

rm = RunManager()
run_id = await rm.start(agent, input={"message": "..."}, session_id="s1", harness=cfg)

async for event in rm.stream(run_id, cursor=0):   # replay then live-tail
    ...

await rm.resume(run_id, [ApprovalDecision(approval_id, "approve")])
status = await rm.get_status(run_id)
```

Reconnection is lossless: every event is persisted with a monotonic cursor;
clients reconnect with the last cursor they saw and receive exactly the missed
events, then live tail. A framework-agnostic FastAPI reference host is in
[`examples/harness/fastapi_server.py`](../examples/harness/fastapi_server.py)
(FastAPI is not a Vel dependency).

`RunManager` uses a SQLite event log by default. For a Postgres-backed event
log, install `pip install 'vel-ai[harness-postgres]'` and pass a DSN:

```python
rm = RunManager(store_backend="postgres", dsn="postgresql://...")
```

## Scaling across workers

A harness agent behind an API endpoint runs in one of two ways, and the
multi-worker story differs for each. **Vel itself is transport- and
deployment-agnostic** — it provides the durable event log and a pluggable live
pub/sub *seam*; your application code and its deployment own the topology.

**Model A — loop tied to the request.** The endpoint does
`async for event in agent.run_stream(..., harness=cfg): yield sse(event)`. The
loop runs inside the worker handling that request and streams over that one
connection. Multi-worker is purely a deployment concern: run N workers, the load
balancer distributes requests, each run is self-contained — **no shared pub/sub
needed.** Trade-off: the live stream ends if the connection drops (durable
suspend/resume still works via the shared checkpoint store; live tail of an
in-flight run does not).

**Model B — detached runs via `RunManager`.** `await rm.start(...)` runs the loop
as a background task in *that* worker's process; the browser attaches via a
separate `GET /runs/{id}/stream` request that may land on a different worker.
Two layers must be shared:

| Layer | How it's shared | Owner |
|---|---|---|
| Durable state (checkpoints, approvals, **event log**) | Postgres (`store_backend="postgres"`) | your app config |
| **Live** pub/sub (real-time notify to SSE subscribers) | in-process by default; pluggable | see below |

With the default in-process pub/sub, a stream request on **worker B** can still
**replay** all past events for a run from the shared event log (lossless
catch-up by cursor), but cannot **live-tail** a run still executing on **worker
A** — the notify queue lives in A's memory. Close that gap one of two ways:

1. **Sticky sessions (deployment, zero code).** Configure the load balancer to
   route all requests for a `run_id` to the same worker (hash-on-path or cookie
   affinity). Postgres + affinity gives durable reconnect *and* live tail with no
   Vel changes. This is the simplest production setup.

2. **Shared pub/sub (optional Vel backend).** Pass a `RedisPubSub` so any worker
   can live-tail any run. Install `pip install 'vel-ai[harness-redis]'`:

   ```python
   from vel.harness import RunManager, RedisPubSub

   rm = RunManager(
       store_backend="postgres", dsn="postgresql://...",
       pubsub=RedisPubSub(url="redis://localhost:6379"),
   )
   ```

   `RunManager` publishes each event (with its cursor) to Redis; a `stream()` on
   any worker replays history from the event log, then live-tails via Redis.
   This is also what lets you run agents in a **separate worker pool** from the
   SSE servers. Default (no `pubsub=`) is single-process `InProcessPubSub` —
   nothing changes unless you opt in.

**Rule of thumb:** single process → nothing to do. Multi-worker → Postgres +
sticky-session affinity (deployment only). Agents in a tier separate from the
SSE servers, or no affinity available → add `RedisPubSub`.

## Sandbox

Isolation is **outsourced** — Vel never builds it. Select a provider and expose
built-in sandbox tools:

```python
harness = {
    "enabled": True,
    "sandbox": {
        "enabled": True,
        "provider": "e2b",            # or "local_subprocess" for explicit dev/test use
        "tools": ["read", "write", "edit", "list", "bash", "python"],
    },
}
```

When `sandbox.enabled`, the controller creates a session at run start, injects
the configured tools (so the model sees them on step 1), persists the sandbox
ref on the checkpoint (so it reconnects on resume), and tears it down per
`lifecycle` (`per_run` closes; `per_session`/`persistent` stay warm). It emits
`data-harness-sandbox` (`created`/`connected`/`closed`).

Mutating/exec tools (`write`/`edit`/`bash`/`python`) default to
`requires_confirmation=True`, so by default they flow through the durable
approval gate. The E2B adapter is soft-imported — install with
`pip install 'vel-ai[sandbox-e2b]'`; absent SDKs only error if that provider is
selected. A gated local-subprocess provider exists for dev/tests (it is **not**
real isolation; requires `provider_options={"unsafe_local": True}`).

### The sandbox as a durable workspace (plan tracking)

The intended pattern for a work/research assistant is to use the sandbox as the
agent's **desk**: a skill instructs it to maintain `/workspace/plan.md` (a living
checklist it edits as it works), drop evidence into `/workspace/findings/`, and
write the deliverable to `/workspace/report.md`. Because the sandbox ref is on
the checkpoint, the plan and artifacts survive suspend/resume and disconnects.
When a sandbox file should be delivered to the browser, call `sandbox_read` with
`expose_file=True`; it emits a standard `file` stream event before the normal
tool output.

For friction-free plan tracking, gate only *consequential* actions and leave the
agent's own workspace edits ungated:

```python
"approval": {
    "enabled": True, "mode": "durable",
    "require_for_tools": ["send_email", "publish_report"],  # gate these
    "require_for_confirmation_flag": False,  # don't gate routine sandbox writes
},
```

A complete, runnable research/work assistant is in
[`examples/harness/research_assistant.py`](../examples/harness/research_assistant.py).

## Skills

A skill is a named bundle of instructions + tools (+ optional model/budget),
activated for a run. Register globally or pass inline:

```python
from vel.harness import Skill, SkillRef, default_registry

default_registry.register(Skill(
    name="researcher",
    instructions="You are an expert researcher. Always cite sources.",
    tools=[ToolSpec.from_function(web_search)],
))

agent = Agent(..., harness={"enabled": True, "skills": [SkillRef(name="researcher")]})
```

v1 activates all configured skills for the whole run (their instructions are
merged into the system context and their tools unioned into the run's tool set).

## Composition with RLM / Extended Thinking

For v1, if `rlm`/`thinking` are configured alongside `harness`, the RLM/Thinking
controllers take precedence (they're routed first). Full composition is future
work.

## Limitations (v1)

- Live-tail pub/sub is **in-process by default**. For multi-worker live tail,
  opt into `RedisPubSub` or use sticky-session affinity — see
  [Scaling across workers](#scaling-across-workers). Reconnect/replay is durable
  on SQLite or Postgres regardless.
- Dynamic mid-run skill activation is not yet supported (all configured skills
  are active for the whole run).

Previously deferred, now supported:
- `RunManager(pubsub=RedisPubSub(url=...))` for cross-worker live tail
  (`pip install 'vel-ai[harness-redis]'`); default stays single-process.
- Sub-agents via `Agent.as_tool(durable=True)` run through the harness
  (budget/compaction/sandbox/checkpointing); approval stays inline because a
  tool call must return (a sub-agent cannot suspend the parent mid-step).
  `durable=False` (default) keeps the original non-durable behavior.
- `RunManager(store_backend="postgres", dsn=...)` backs the event log with
  Postgres (`pip install 'vel-ai[harness-postgres]'`).
- `sandbox.lifecycle="per_session"`/`"persistent"` reuses one workspace across a
  session's runs (reconnect by ref; E2B reconnects across process restarts).
- `compaction.strategy="memory_offload"` writes raw turns to the FactStore and a
  distilled note to ReasoningBank (when memory is configured; no-op otherwise).

## See also

- [Agent Directories](agent-directory) — author a harness agent as a directory of
  files and load it with `load_agent`.
- [Channels](channels) — drive a harness run over CLI, Slack, or your own
  transport with the `Channel` adapter over `RunManager`.
