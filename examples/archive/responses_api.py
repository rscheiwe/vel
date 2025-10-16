"""
OpenAI Responses API Example

Demonstrates using the OpenAI Responses API (/v1/responses) instead of
the Chat Completions API (/v1/chat/completions).

The Responses API provides structured events:
- response.text.delta for text streaming
- response.reasoning.delta for reasoning (o1/o3 models)
- response.output_item.added for synthesis
- response.function_call_arguments.delta for tool calls
- Provider-executed tools (web_search, computer)

Note: Requires OpenAI API key with Responses API access (currently limited availability).
"""
import asyncio
import os
from vel import Agent, ToolSpec, register_tool

# Register a simple weather tool for Example 3
register_tool(ToolSpec(
    name='get_weather',
    input_schema={'type': 'object', 'properties': {'city': {'type': 'string'}}, 'required': ['city']},
    output_schema={'type': 'object', 'properties': {'temp_f': {'type': 'number'}}, 'required': ['temp_f']},
    handler=lambda inp, ctx: {'temp_f': 72.0}
))

async def main():
    # Check API key
    if not os.getenv('OPENAI_API_KEY'):
        print("Error: OPENAI_API_KEY environment variable not set")
        exit(1)

    print("=" * 70)
    print("OpenAI Responses API Example")
    print("=" * 70)
    print()
    print("Using provider: 'openai-responses' (Responses API /v1/responses)")
    print("Comparing to: 'openai' (Chat Completions API /v1/chat/completions)")
    print()

    # Example 1: Basic text streaming with Responses API
    print("=" * 70)
    print("Example 1: Basic Text Streaming")
    print("=" * 70)
    print()

    agent = Agent(
        id='responses-agent:v1',
        model={
            'provider': 'openai-responses',  # Use Responses API
            'model': 'gpt-4o'
        },
        policies={'max_steps': 3}
    )

    print("Question: 'What is the capital of France?'")
    print()
    print("Streaming events:")
    print("-" * 70)

    async for event in agent.run_stream({'message': 'What is the capital of France?'}):
        event_type = event.get('type')

        if event_type == 'start':
            print("[START] Generation begins")

        elif event_type == 'step-start':
            print("[STEP-START] New agentic step")

        elif event_type == 'text-start':
            print(f"[TEXT-START] Block ID: {event.get('id')}")
            print("Text: ", end='', flush=True)

        elif event_type == 'text-delta':
            print(event.get('delta', ''), end='', flush=True)

        elif event_type == 'text-end':
            print()
            print(f"[TEXT-END] Block ID: {event.get('id')}")

        elif event_type == 'finish-message':
            print(f"[FINISH-MESSAGE] Reason: {event.get('finishReason')}")

        elif event_type == 'step-finish':
            print("[STEP-FINISH] Step complete")

        elif event_type == 'finish':
            print("[FINISH] Generation complete")

    print()

    # Example 2: Reasoning model with Responses API
    print("=" * 70)
    print("Example 2: Reasoning Model (o1)")
    print("=" * 70)
    print()

    reasoning_agent = Agent(
        id='responses-reasoning:v1',
        model={
            'provider': 'openai-responses',
            'model': 'o1'  # Reasoning model
        },
        policies={'max_steps': 3}
    )

    print("Question: 'What is 13^2 + 7?'")
    print()
    print("Streaming events:")
    print("-" * 70)

    reasoning_text = []
    answer_text = []

    async for event in reasoning_agent.run_stream({'message': 'What is 13^2 + 7?'}):
        event_type = event.get('type')

        if event_type == 'reasoning-start':
            print("[REASONING-START]")
            reasoning_text = []

        elif event_type == 'reasoning-delta':
            reasoning_text.append(event.get('delta', ''))

        elif event_type == 'reasoning-end':
            print("[REASONING-END]")
            print()
            if reasoning_text:
                print("Full reasoning:")
                print(''.join(reasoning_text))
            else:
                print("(Reasoning content is encrypted/hidden by OpenAI)")
            print()

        elif event_type == 'text-start':
            print("[TEXT-START] Final answer:")
            answer_text = []

        elif event_type == 'text-delta':
            answer_text.append(event.get('delta', ''))

        elif event_type == 'text-end':
            print(''.join(answer_text))
            print()

        elif event_type == 'finish':
            print("[FINISH]")

    print()

    # Example 3: Tool calling with Responses API
    print("=" * 70)
    print("Example 3: Tool Calling")
    print("=" * 70)
    print()

    tool_agent = Agent(
        id='responses-tools:v1',
        model={
            'provider': 'openai-responses',
            'model': 'gpt-4o'
        },
        tools=['get_weather'],
        policies={'max_steps': 8}
    )

    print("Question: 'What is the weather in San Francisco?'")
    print()
    print("Streaming events:")
    print("-" * 70)

    async for event in tool_agent.run_stream({'message': 'What is the weather in San Francisco?'}):
        event_type = event.get('type')

        if event_type == 'tool-input-start':
            print(f"[TOOL-INPUT-START] Tool: {event.get('toolName')}, ID: {event.get('toolCallId')}")

        elif event_type == 'tool-input-delta':
            # Streaming tool arguments
            pass  # Skip for brevity

        elif event_type == 'tool-input-available':
            print(f"[TOOL-INPUT-AVAILABLE] Tool: {event.get('toolName')}")
            print(f"  Input: {event.get('input')}")

        elif event_type == 'tool-output-available':
            print(f"[TOOL-OUTPUT-AVAILABLE]")
            print(f"  Output: {event.get('output')}")

        elif event_type == 'text-delta':
            print(event.get('delta', ''), end='', flush=True)

        elif event_type == 'text-end':
            print()

        elif event_type == 'finish':
            print("[FINISH]")

    print()
    print("=" * 70)
    print("Done!")
    print("=" * 70)
    print()
    print("Note: The Responses API provides more structured events compared to")
    print("Chat Completions, making it easier to handle complex interactions.")


if __name__ == '__main__':
    asyncio.run(main())
