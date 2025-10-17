"""
Basic Streaming Example

Demonstrates basic streaming with MessageReducer and provider interchange.
Shows how to:
- Stream text responses
- Use MessageReducer to aggregate events
- Switch between providers (OpenAI, Anthropic, Gemini)
"""
import asyncio
import json
import os
from vel import Agent, MessageReducer

# ====== CONFIGURATION ======
# Change this to test different providers
PROVIDER = 'openai'  # Options: 'openai', 'anthropic', 'gemini'

PROVIDER_CONFIG = {
    'openai': {'provider': 'openai', 'model': 'gpt-4o-mini'},
    'anthropic': {'provider': 'anthropic', 'model': 'claude-sonnet-4-20250514'},
    'gemini': {'provider': 'google', 'model': 'gemini-1.5-flash'}
}

async def main():
    # Check for API key
    if PROVIDER == 'openai' and not os.getenv('OPENAI_API_KEY'):
        print("Error: OPENAI_API_KEY not set")
        return
    elif PROVIDER == 'anthropic' and not os.getenv('ANTHROPIC_API_KEY'):
        print("Error: ANTHROPIC_API_KEY not set")
        return
    elif PROVIDER == 'gemini' and not os.getenv('GOOGLE_API_KEY'):
        print("Error: GOOGLE_API_KEY not set")
        return

    print("=" * 70)
    print("Basic Streaming Example")
    print("=" * 70)
    print(f"Provider: {PROVIDER}")
    print(f"Model: {PROVIDER_CONFIG[PROVIDER]['model']}")
    print()

    # Create agent
    agent = Agent(
        id='basic-agent:v1',
        model=PROVIDER_CONFIG[PROVIDER],
        policies={'max_steps': 3}
    )

    # Create reducer
    reducer = MessageReducer()
    user_input = "What is the capital of France? Give a brief answer."
    reducer.add_user_message(user_input)

    print(f"User: {user_input}")
    print()
    print("Streaming response:")
    print("-" * 70)

    # Stream response and process events
    accumulated_text = []
    async for event in agent.run_stream({'message': user_input}):
        reducer.process_event(event)
        print(event)
        # Display text as it streams
        # if event.get('type') == 'text-delta':
        #     delta = event.get('delta', '')
        #     accumulated_text.append(delta)
        #     print(delta, end='', flush=True)

    print()
    print("-" * 70)
    print()

    # Get aggregated messages
    messages = reducer.get_messages()

    print("=" * 70)
    print("Aggregated Messages (AI SDK format)")
    print("=" * 70)
    print(json.dumps(messages, indent=2))
    print()

    # Display summary
    assistant_msg = messages[1]
    parts = assistant_msg['parts']
    print("=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"Total parts: {len(parts)}")
    for i, part in enumerate(parts):
        print(f"  Part {i + 1}: {part['type']}")
    print()

if __name__ == '__main__':
    asyncio.run(main())
