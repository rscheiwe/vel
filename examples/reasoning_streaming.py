"""
Reasoning Streaming Example

Demonstrates reasoning/thinking with MessageReducer and provider interchange.
Shows how to:
- Stream reasoning/thinking from models
- See reasoning-start/delta/end events
- Use MessageReducer to aggregate reasoning parts
- Switch between providers (OpenAI o1, Claude with thinking)

Note:
- OpenAI: Uses Responses API with o1 models (reasoning encrypted)
- Claude: Uses extended thinking mode (thinking visible)
- Gemini: Standard model (no reasoning mode)
"""
import asyncio
import json
import os
from dotenv import load_dotenv
from vel import Agent, MessageReducer

# Load environment variables from .env file
load_dotenv()

# ====== CONFIGURATION ======
# Change this to test different providers
PROVIDER = 'anthropic'  # Options: 'openai', 'anthropic', 'gemini'

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
    print("Reasoning Streaming Example")
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
    print("Streaming response:")
    print("-" * 70)

    # Stream response and process events
    reasoning_chunks = []
    text_chunks = []

    async for event in agent.run_stream({'message': user_input}):
        reducer.process_event(event)
        event_type = event.get('type')

        # Display events
        if event_type == 'reasoning-start':
            print("\n[🧠 REASONING START]")
        elif event_type == 'reasoning-delta':
            delta = event.get('delta', '')
            if delta:  # Only print if not empty (OpenAI encrypts)
                reasoning_chunks.append(delta)
                print(delta, end='', flush=True)
        elif event_type == 'reasoning-end':
            if reasoning_chunks:
                print()
            print("[🧠 REASONING END]\n")
        elif event_type == 'text-start':
            print("[📝 TEXT START]")
        elif event_type == 'text-delta':
            delta = event.get('delta', '')
            text_chunks.append(delta)
            print(delta, end='', flush=True)
        elif event_type == 'text-end':
            print("\n[📝 TEXT END]")

    print("-" * 70)
    print()

    # Get aggregated messages
    messages = reducer.get_messages()

    print("=" * 70)
    print("Summary")
    print("=" * 70)
    assistant_msg = messages[1]
    parts = assistant_msg['parts']
    print(f"Total parts: {len(parts)}")

    for i, part in enumerate(parts):
        part_type = part['type']
        if part_type == 'reasoning':
            reasoning_text = part.get('text', '')
            if reasoning_text:
                print(f"  Part {i + 1}: reasoning ({len(reasoning_text)} chars)")
            else:
                print(f"  Part {i + 1}: reasoning (encrypted)")
        elif part_type == 'text':
            text = part.get('text', '')
            print(f"  Part {i + 1}: text ({len(text)} chars)")
        else:
            print(f"  Part {i + 1}: {part_type}")
    print()

    # Show message structure
    print("=" * 70)
    print("Message Structure (AI SDK format)")
    print("=" * 70)
    print(json.dumps(messages, indent=2))
    print()

if __name__ == '__main__':
    asyncio.run(main())
