"""
Reasoning Non-Streaming Example

Demonstrates reasoning/thinking with MessageReducer and provider interchange.
Shows how to:
- Get complete reasoning responses
- Use MessageReducer to structure reasoning parts
- Switch between providers (OpenAI o1, Claude with thinking)

Note:
- OpenAI: Uses Responses API with o1 models
- Claude: Uses extended thinking mode
- Gemini: Standard model (no reasoning mode)
- Non-streaming aggregates the reasoning but doesn't show intermediate steps
"""
import asyncio
import json
import os
from vel import Agent, MessageReducer

# ====== CONFIGURATION ======
# Change this to test different providers
PROVIDER = 'openai'  # Options: 'openai', 'anthropic', 'gemini'

PROVIDER_CONFIG = {
    # OpenAI o1 with Responses API
    'openai': {'provider': 'openai-responses', 'model': 'o3-mini'},

    # Claude with extended thinking
    'anthropic': {
        'provider': 'anthropic',
        'model': 'claude-sonnet-4-20250514'
    },

    # Gemini (standard, no reasoning mode)
    'gemini': {'provider': 'google', 'model': 'gemini-1.5-flash'}
}

GENERATION_CONFIG = {
    # Claude requires thinking parameter for extended thinking
    # Note: max_tokens must be greater than budget_tokens
    'anthropic': {
        'max_tokens': 8192,  # Must be > budget_tokens
        'thinking': {
            'type': 'enabled',
            'budget_tokens': 5000
        }
    }
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
    print("Reasoning Non-Streaming Example")
    print("=" * 70)
    print(f"Provider: {PROVIDER}")
    print(f"Model: {PROVIDER_CONFIG[PROVIDER]['model']}")
    if PROVIDER == 'openai':
        print("Note: OpenAI encrypts reasoning content")
    elif PROVIDER == 'anthropic':
        print("Note: Claude's thinking is visible")
    print()

    # Create agent with generation config for thinking
    gen_config = GENERATION_CONFIG.get(PROVIDER)
    agent = Agent(
        id='reasoning-agent:v1',
        model=PROVIDER_CONFIG[PROVIDER],
        policies={'max_steps': 3},
        generation_config=gen_config
    )

    # Create reducer
    reducer = MessageReducer()
    user_input = "What is the square root of 169? Explain your reasoning step by step."
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
    reducer._accumulated_text.append(response)
    reducer._flush_accumulated_text()

    # Get aggregated messages
    messages = reducer.get_messages()

    print("=" * 70)
    print("Aggregated Messages (AI SDK format)")
    print("=" * 70)
    print(json.dumps(messages, indent=2))
    print()

    print("=" * 70)
    print("Note")
    print("=" * 70)
    print("Non-streaming mode doesn't preserve reasoning events.")
    print("Use streaming mode (reasoning_streaming.py) to see")
    print("reasoning-start/delta/end events and MessageReducer")
    print("aggregation of reasoning parts.")
    print()

if __name__ == '__main__':
    asyncio.run(main())
