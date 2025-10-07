from __future__ import annotations
import asyncio
from typing import AsyncGenerator
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from agents import Agent

app = FastAPI(title="Vel Service")

@app.post("/runs/{message}")
async def start_run(message: str):
    agent = Agent(
        id="chat-general:v1",
        model={"provider":"openai", "model":"gpt-4o"},
        tools=["get_weather"],
        policies={"max_steps": 8}
    )
    async def event_stream():
        async for e in agent.run_stream({"message":message}):
            yield f"data: {e}\n\n"
            await asyncio.sleep(0)
    return StreamingResponse(event_stream(), media_type="text/event-stream")

@app.get("/healthz")
def healthz():
    return {"ok": True}
