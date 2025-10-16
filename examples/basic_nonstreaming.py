"""
Basic Non-Streaming Example

Demonstrates non-streaming execution with MessageReducer and provider interchange.
Shows how to:
- Get complete responses without streaming
- Use MessageReducer to structure messages
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
    print("Basic Non-Streaming Example")
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
    print("Getting response...")
    print()

    # Get non-streaming response
    response = await agent.run({'message': user_input})

    print("=" * 70)
    print("Response")
    print("=" * 70)
    print(response)
    print()

    # Manually add response to reducer for storage
    # Note: For non-streaming, we create a simple text part
    reducer._accumulated_text.append(response)
    reducer._flush_accumulated_text()

    # Get aggregated messages
    messages = reducer.get_messages()

    print("=" * 70)
    print("Aggregated Messages (AI SDK format)")
    print("=" * 70)
    print(json.dumps(messages, indent=2))
    print()

if __name__ == '__main__':
    asyncio.run(main())
