"""
Recoverable Tool Error Example

Demonstrates that a tool handler can raise an exception without killing the
agent run. Vel emits a tool-output-error event, feeds the failure back to the
model as a tool result, and lets the model recover with a normal answer.
"""

import asyncio
import os

from dotenv import load_dotenv

from vel import Agent, ToolSpec


load_dotenv()


def lookup_order(order_id: str, ctx: dict = None) -> dict:
    """Look up an order by numeric id."""
    if not order_id.isdigit():
        raise ValueError("order_id must contain only digits")
    return {"order_id": order_id, "status": "shipped"}


async def main() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not set")
        return

    agent = Agent(
        id="tool-error-recovery:v1",
        model={"provider": "openai", "model": "gpt-4o-mini"},
        tools=[ToolSpec.from_function(lookup_order)],
        policies={"max_steps": 5},
    )

    prompt = (
        "Use lookup_order with order_id='ABC-123'. "
        "If the tool fails, explain the problem and ask for a numeric order id."
    )

    print(f"User: {prompt}")
    print("-" * 70)

    async for event in agent.run_stream({"message": prompt}):
        event_type = event.get("type")
        if event_type == "tool-input-available":
            print(f"\n[tool] {event.get('toolName')} <- {event.get('input')}")
        elif event_type == "tool-output-error":
            print(f"[tool] {event.get('toolCallId')} -> ERROR: {event.get('errorText')}")
        elif event_type == "text-delta":
            print(event.get("delta", ""), end="", flush=True)

    print()


if __name__ == "__main__":
    asyncio.run(main())
