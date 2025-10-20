"""
Multi-Step Non-Streaming Example

Demonstrates multi-step reasoning with MessageReducer and provider interchange.
Shows how to:
- Get complete multi-step responses
- Use MessageReducer to structure complex interactions
- Switch between providers (OpenAI, Anthropic, Gemini)
"""
import asyncio
import json
import os
from dotenv import load_dotenv
from vel import Agent, MessageReducer, ToolSpec, register_tool

# Load environment variables from .env file
load_dotenv()

# Register tools for multi-step reasoning
register_tool(ToolSpec(
    name='search_web',
    input_schema={
        'type': 'object',
        'properties': {
            'query': {'type': 'string', 'description': 'Search query'}
        },
        'required': ['query']
    },
    output_schema={
        'type': 'object',
        'properties': {
            'results': {'type': 'array', 'items': {'type': 'string'}}
        }
    },
    handler=lambda inp, ctx: {
        'results': [
            f"Result 1 for '{inp['query']}'",
            f"Result 2 for '{inp['query']}'"
        ]
    }
))

register_tool(ToolSpec(
    name='calculate',
    input_schema={
        'type': 'object',
        'properties': {
            'expression': {'type': 'string', 'description': 'Math expression'}
        },
        'required': ['expression']
    },
    output_schema={
        'type': 'object',
        'properties': {
            'result': {'type': 'number'}
        }
    },
    handler=lambda inp, ctx: {
        'result': eval(inp['expression'])  # Note: eval is unsafe, use for demo only
    }
))

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
    print("Multi-Step Non-Streaming Example")
    print("=" * 70)
    print(f"Provider: {PROVIDER}")
    print(f"Model: {PROVIDER_CONFIG[PROVIDER]['model']}")
    print()

    # Create agent with multiple tools
    agent = Agent(
        id='multi-step-agent:v1',
        model=PROVIDER_CONFIG[PROVIDER],
        tools=['search_web', 'calculate'],
        policies={'max_steps': 8}
    )

    # Create reducer
    reducer = MessageReducer()
    user_input = "Search for the population of Tokyo, then calculate what 10% of that number would be."
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
