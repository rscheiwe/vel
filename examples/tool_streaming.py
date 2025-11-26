"""
Tool Usage Streaming Example

Demonstrates streaming tool calls with MessageReducer and provider interchange.
Shows how to:
- Stream responses with tool calls
- See tool input/output events in real-time
- Use MessageReducer to aggregate tool interactions
- Switch between providers (OpenAI, Anthropic, Gemini)
"""
import asyncio
import json
import os
from dotenv import load_dotenv
from vel import Agent, MessageReducer, ToolSpec

# Load environment variables from .env file
load_dotenv()

# Create a simple weather tool (no registration needed!)
def get_weather(city: str, ctx: dict = None) -> dict:
    """Get weather for a city."""
    return {
        'temp_f': 72.0,
        'condition': 'sunny',
        'city': city
    }

weather_tool = ToolSpec.from_function(get_weather)

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
    print("Tool Usage Streaming Example")
    print("=" * 70)
    print(f"Provider: {PROVIDER}")
    print(f"Model: {PROVIDER_CONFIG[PROVIDER]['model']}")
    print()

    # Create agent with tool
    agent = Agent(
        id='tool-agent:v1',
        model=PROVIDER_CONFIG[PROVIDER],
        tools=[weather_tool],  # Pass ToolSpec directly
        policies={'max_steps': 5}
    )

    # Create reducer
    reducer = MessageReducer()
    user_input = "What's the weather in San Francisco?"
    reducer.add_user_message(user_input)

    print(f"User: {user_input}")
    print()
    print("Streaming response:")
    print("-" * 70)

    # Stream response and process events
    async for event in agent.run_stream({'message': user_input}):
        reducer.process_event(event)
        event_type = event.get('type')

        # Display events
        if event_type == 'tool-input-start':
            print(f"\n[TOOL CALL] {event.get('toolName')}")
        elif event_type == 'tool-input-available':
            print(f"  Input: {event.get('input')}")
        elif event_type == 'tool-output-available':
            print(f"  Output: {event.get('output')}")
        elif event_type == 'text-delta':
            print(event.get('delta', ''), end='', flush=True)

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
        part_type = part['type']
        if 'tool-' in part_type:
            print(f"  Part {i + 1}: {part_type} (state: {part.get('state', 'N/A')})")
        else:
            print(f"  Part {i + 1}: {part_type}")
    print()

if __name__ == '__main__':
    asyncio.run(main())
