import asyncio, pytest
from agents import Agent, run_stream

@pytest.mark.asyncio
async def test_stream_smoke():
    agent = Agent(
        id="chat-general:v1",
        model={"provider":"openai", "model":"gpt-4o"},
        tools=["get_weather"],
        policies={"max_steps": 5}
    )
    seen = False
    async for e in run_stream(agent, {"message":"hi"}):
        if e.get("kind") == "final":
            seen = True
    assert seen
