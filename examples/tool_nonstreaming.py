"""
Tool Usage Non-Streaming Example

Demonstrates non-streaming tool calls with MessageReducer and provider interchange.
Shows how to:
- Get complete responses with tool calls
- Use MessageReducer to structure tool interactions
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
    print("Tool Usage Non-Streaming Example")
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

if __name__ == '__main__':
    asyncio.run(main())
