# Vel — Agent Runtime (12‑Factor Agents aligned)

Vel is a Python agent runtime with first‑class streaming, an extensible tool & provider system, and a stateless reducer core. It follows the 12‑Factor Agents guidance: natural‑language → structured tool calls, own prompts/context, unified state, lifecycle APIs, human‑in‑the‑loop as a tool, and compact errors into context.

## Quickstart (SDK)

```bash
pip install -e .[service,dev]
export OPENAI_API_KEY=...

python examples/quickstart.py
```

## Service

```bash
uvicorn agents_service.main:app --reload
# Then POST /runs and GET /runs/{id}/stream (SSE)
```

## Repo layout
- `agents/` — core runtime (reducer, agent loop, tools, providers, context, RAG, storage)
- `agents_service/` — FastAPI service with SSE streaming + lifecycle APIs
- `docs/PRD.md` — product spec mapped to your requirements
- `examples/` — CLI & SDK demos
- `tests/` — minimal v1 tests
