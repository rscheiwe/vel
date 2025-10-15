from __future__ import annotations
import asyncio, json
from typing import Dict, Any
from fastapi import FastAPI
from fastapi.responses import StreamingResponse, HTMLResponse, JSONResponse
from pydantic import BaseModel
from vel import Agent

app = FastAPI(title="Vel Service")

class RunRequest(BaseModel):
    agent_id: str = "chat-general:v1"
    input: Dict[str, Any] = {"message":"hello"}
    provider: str = "openai"
    model: str = "gpt-4o"

@app.get("/", response_class=HTMLResponse)
def ui():
    return """
    <h1>Vel</h1>
    <p>POST /runs - streaming SSE</p>
    <p>POST /runs/sync - non-streaming JSON</p>
    """

@app.post("/runs")
async def start_run(req: RunRequest):
    """Streaming endpoint - returns SSE events compatible with Vercel AI SDK V5 UI Stream Protocol"""
    agent = Agent(
        id=req.agent_id,
        model={"provider": req.provider, "model": req.model},
        tools=["get_weather"],
        policies={"max_steps": 12}
    )
    async def event_stream():
        try:
            async for e in agent.run_stream(req.input):
                yield f"data: {json.dumps(e)}\n\n"
                await asyncio.sleep(0)
            # Stream termination marker
            yield "data: [DONE]\n\n"
        except Exception as err:
            # Emit error event
            error_event = {"type": "error", "error": str(err)}
            yield f"data: {json.dumps(error_event)}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Vercel-AI-UI-Message-Stream": "v1"  # V5 UI Stream Protocol header
        }
    )

@app.post("/runs/sync")
async def start_run_sync(req: RunRequest):
    """Non-streaming endpoint - returns final answer only"""
    agent = Agent(
        id=req.agent_id,
        model={"provider": req.provider, "model": req.model},
        tools=["get_weather"],
        policies={"max_steps": 12}
    )
    try:
        answer = await agent.run(req.input)
        return JSONResponse({
            "status": "completed",
            "answer": answer
        })
    except Exception as e:
        return JSONResponse({
            "status": "failed",
            "error": str(e)
        }, status_code=500)

@app.get("/healthz")
def healthz():
    return {"ok": True}
