---
layout: default
title: Agent Directories
nav_order: 14
---

# Agent Directories

Vel agents are usually built programmatically with `Agent(...)`. For
inspectable, git-diffable projects you can also author an agent as a **directory
of files** (filesystem-first, inspired by Vercel eve) and compile it with
`load_agent`. This is pure sugar over the existing API — no new runtime behavior.

## Layout

```
my_agent/
  agent.toml         # [model], [agent], [harness] (+ nested [harness.*])
  instructions.md    # system prompt (optional)
  tools/*.py         # one tool per file; exports `tool` (ToolSpec or callable)
  skills/*.py        # one skill per file; exports `skill` (a Skill)
```

Only `agent.toml` with a `[model]` table is required; every other file/directory
is optional and auto-discovered by name.

## `agent.toml`

```toml
[agent]
id = "desk-assistant"          # defaults to the directory name if omitted
# any other simple Agent(...) kwarg may go here (e.g. prompt_env)

[model]
provider = "openai"
model = "gpt-4o"

[harness]                       # optional — omit for a plain agent
enabled = true

[harness.budget]
max_steps = 20

[harness.approval]
mode = "durable"
require_for_tools = ["publish_note"]
```

The `[harness]` table (and its nested `[harness.approval|budget|sandbox|compaction]`
tables) maps straight onto [`HarnessConfig`](harness). If the directory ships
skills, the harness is enabled automatically.

## Tools and skills

Each file under `tools/` defines **one** tool and exports it as `tool` — either a
`ToolSpec` or a plain callable (wrapped via `ToolSpec.from_function`):

```python
# tools/search_notes.py
from vel import ToolSpec

async def search_notes(query: str = "", ctx: dict = None) -> dict:
    return {"query": query, "results": [...]}

tool = ToolSpec.from_function(search_notes)
```

Each file under `skills/` exports a `skill` (a `vel.harness.Skill`):

```python
# skills/note_taking.py
from vel.harness import Skill

skill = Skill(name="note-taking", instructions="Add 2-3 hashtags to each note.")
```

## Loading

```python
from vel import load_agent

agent = load_agent("my_agent")
async for event in agent.run_stream({"message": "..."}):
    ...
```

`load_agent` maps `instructions.md` onto the agent's `system_prompt`, discovers
tools/skills, and builds the `HarnessConfig`. It raises `FileNotFoundError` for a
missing directory or `agent.toml`, and `ValueError` if `[model]` is absent or a
tool/skill module doesn't export the expected symbol.

A complete example lives in
[`examples/harness/agent_dir/`](../examples/harness/agent_dir) with a runner at
[`examples/harness/run_directory_agent.py`](../examples/harness/run_directory_agent.py).
