# PRD — Vel (Agent Module)

## Goals
- General chat agent, easily extended into a research assistant.
- Streaming to CLI, Python SDK, and Web via SSE.
- Very agentic planner/actor loop with structured tool calls.
- Strict JSON schema validation for tools.
- Event‑sourced runs; Postgres + Redis; replay/rewind/retry.
- Env‑aware prompts; RAG via pgvector (Postgres).
- Human‑in‑the‑loop tool (UI‑driven), opt‑in per agent.
- Packaging: pip‑installable package + FastAPI service.
- Orchestration: local async or queue (flag‑selectable).
- Observability v1: structured JSON logs.

(See README for quickstart. See inline docstrings throughout for contracts.)
