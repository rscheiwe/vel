from __future__ import annotations
import asyncio, json
from typing import Dict, Any
from fastapi import FastAPI
from fastapi.responses import StreamingResponse, HTMLResponse
from pydantic import BaseModel
from agents import Agent

app = FastAPI(title="Vel Service")

class RunRequest(BaseModel):
    agent_id: str = "chat-general:v1"
    input: Dict[str, Any] = {"message":"hello"}

@app.get("/", response_class=HTMLResponse)
def ui():
    return "<h1>Vel</h1><p>POST /runs to stream SSE.</p>"

@app.post("/runs")
async def start_run(req: RunRequest):
    agent = Agent(
        id=req.agent_id,
        model={"provider":"openai", "model":"gpt-4o"},
        tools=["get_weather"],
        policies={"max_steps": 12}
    )
    async def event_stream():
        async for e in agent.run_stream(req.input):
            yield f"data: {json.dumps(e)}\n\n"
            await asyncio.sleep(0)
    return StreamingResponse(event_stream(), media_type="text/event-stream")

@app.get("/healthz")
def healthz():
    return {"ok": True}
