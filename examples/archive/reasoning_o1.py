"""
Reasoning Model Example (o1/o3)

Demonstrates how Vel handles reasoning content from OpenAI's o1/o3 models.

IMPORTANT: Use 'openai-responses' provider (not 'openai') for reasoning event indicators.
The Responses API provides structured events including reasoning-start/end even when
OpenAI encrypts the actual reasoning content.

Vel automatically emits reasoning-start, reasoning-delta, and reasoning-end events
that follow the AI SDK V5 stream protocol.
"""
import asyncio
import os
from vel import Agent, MessageReducer

async def main():
    # Check API key
    if not os.getenv('OPENAI_API_KEY'):
        print("Error: OPENAI_API_KEY environment variable not set")
        exit(1)

    # Create agent with o1 model - MUST use 'openai-responses' provider
    # Note: o1/o3 models don't support tools
    agent = Agent(
        id='reasoning-agent:v1',
        model={
            'provider': 'openai-responses',  # Use Responses API for reasoning events
            'model': 'o3-mini'  # Use 'o1-mini' or 'o3-mini' for faster/cheaper
        },
        policies={'max_steps': 3}
    )

    print("=" * 70)
    print("Reasoning Model Example (o1) - Responses API")
    print("=" * 70)
    print()
    print("Provider: 'openai-responses' (required for reasoning events)")
    print("Asking: 'What is the square root of 169, step by step?'")
    print()
    print("Watch for reasoning-start, reasoning-delta, and reasoning-end events:")
    print("-" * 70)
    print()

    reducer = MessageReducer()

    reducer.add_user_message("What is the square root of 169? Explain step by step.")

    async for event in agent.run_stream(
        input={'message': 'What is the square root of 169? Explain step by step.'}
    ):
        event_type = event.get('type')
        reducer.process_event(event)
        print(event)

        # if event_type == 'reasoning-start':
        #     print(f"[REASONING START] Block ID: {event.get('id')}")
        #     reasoning_content = []

        # elif event_type == 'reasoning-delta':
        #     delta = event.get('delta', '')
        #     reasoning_content.append(delta)
        #     # Print reasoning as it streams (truncate for readability)
        #     if delta:
        #         preview = delta[:80] + '...' if len(delta) > 80 else delta
        #         print(f"[REASONING] {preview}")

        # elif event_type == 'reasoning-end':
        #     print(f"[REASONING END] Block ID: {event.get('id')}")
        #     print()
        #     if reasoning_content:
        #         print("Full reasoning:")
        #         print(''.join(reasoning_content))
        #     else:
        #         print("(Reasoning content is encrypted/hidden by OpenAI)")
        #     print()

        # elif event_type == 'text-start':
        #     print(f"[TEXT START] Block ID: {event.get('id')}")
        #     text_content = []

        # elif event_type == 'text-delta':
        #     text_content.append(event.get('delta', ''))

        # elif event_type == 'text-end':
        #     print(f"[TEXT END] Block ID: {event.get('id')}")
        #     print()
        #     print("Final answer:")
        #     print(''.join(text_content))
        #     print()

        # elif event_type == 'finish':
        #     print("-" * 70)
        #     print("Streaming complete!")
    messages = reducer.get_messages()
    print()
    print("=" * 70)
    print("Done!")
    print("=" * 70)

if __name__ == '__main__':
    asyncio.run(main())
