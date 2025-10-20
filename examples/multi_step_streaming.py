"""
Multi-Step Streaming Example

Demonstrates multi-step reasoning with MessageReducer and provider interchange.
Shows how to:
- Stream multi-step agent execution
- See start-step/finish-step events
- Track multiple tool calls across steps
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
    print("Multi-Step Streaming Example")
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
    print("Streaming response:")
    print("-" * 70)

    # Stream response and process events
    step_num = 0
    async for event in agent.run_stream({'message': user_input}):
        reducer.process_event(event)
        event_type = event.get('type')

        # Display events
        if event_type == 'start-step':
            step_num += 1
            print(f"\n[STEP {step_num} START]")
        elif event_type == 'finish-step':
            print(f"[STEP {step_num} END]\n")
        elif event_type == 'tool-input-start':
            print(f"  → Tool: {event.get('toolName')}")
        elif event_type == 'tool-input-available':
            print(f"    Input: {event.get('input')}")
        elif event_type == 'tool-output-available':
            print(f"    Output: {event.get('output')}")
        elif event_type == 'text-delta':
            print(event.get('delta', ''), end='', flush=True)
        elif event_type == 'text-end':
            print()

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
    print(f"Total steps: {step_num}")

    step_starts = [p for p in parts if p['type'] == 'start-step']
    tool_parts = [p for p in parts if 'tool-' in p['type']]
    text_parts = [p for p in parts if p['type'] == 'text']

    print(f"Start-step events: {len(step_starts)}")
    print(f"Tool calls: {len(tool_parts)}")
    print(f"Text responses: {len(text_parts)}")
    print()

if __name__ == '__main__':
    asyncio.run(main())
