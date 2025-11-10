"""
Example: Tool Use Behavior - stop_on_first_tool

Demonstrates how to configure agents to halt execution after tool calls
and return raw tool output instead of continuing to the LLM for a final answer.

This is useful when:
- You want structured tool output (JSON) not prose
- You're using the LLM for routing/intent detection only
- You want minimal latency (skip the final LLM call)
"""

import asyncio
import os
from vel import Agent, ToolSpec, register_tool

# Define example tools
async def get_weather(input: dict, ctx: dict) -> dict:
    """Get weather for a city - we want this to halt and return raw data"""
    city = input['city']
    # Simulated weather data
    return {
        'city': city,
        'temperature': 72,
        'condition': 'sunny',
        'humidity': 45
    }

async def send_email(input: dict, ctx: dict) -> dict:
    """Send an email - we want this to continue to LLM for confirmation"""
    to = input['to']
    subject = input['subject']
    # Simulated email sending
    return {
        'status': 'sent',
        'message_id': 'msg-12345',
        'to': to
    }

# Register tools
register_tool(ToolSpec(
    name='get_weather',
    input_schema={
        'type': 'object',
        'properties': {'city': {'type': 'string'}},
        'required': ['city']
    },
    output_schema={
        'type': 'object',
        'properties': {
            'city': {'type': 'string'},
            'temperature': {'type': 'number'},
            'condition': {'type': 'string'},
            'humidity': {'type': 'number'}
        },
        'required': ['city', 'temperature', 'condition']
    },
    handler=get_weather
))

register_tool(ToolSpec(
    name='send_email',
    input_schema={
        'type': 'object',
        'properties': {
            'to': {'type': 'string'},
            'subject': {'type': 'string'},
            'body': {'type': 'string'}
        },
        'required': ['to', 'subject']
    },
    output_schema={
        'type': 'object',
        'properties': {
            'status': {'type': 'string'},
            'message_id': {'type': 'string'}
        },
        'required': ['status']
    },
    handler=send_email
))


async def example_1_global_stop():
    """Example 1: Global stop_on_first_tool - ALL tools halt execution"""
    print("\n=== Example 1: Global stop_on_first_tool ===\n")

    agent = Agent(
        id='weather-agent',
        model={'provider': 'openai', 'model': 'gpt-4o'},
        tools=['get_weather', 'send_email'],
        policies={
            'max_steps': 8,
            'stop_on_first_tool': True  # ALL tools will halt
        }
    )

    # This will execute get_weather and return raw JSON
    result = await agent.run({'message': "What's the weather in San Francisco?"})

    print(f"Result type: {type(result)}")
    print(f"Raw tool output: {result}")
    # Expected: {'city': 'San Francisco', 'temperature': 72, 'condition': 'sunny', 'humidity': 45}


async def example_2_per_tool_behavior():
    """Example 2: Per-tool behavior - only specific tools halt"""
    print("\n=== Example 2: Per-tool behavior ===\n")

    agent = Agent(
        id='assistant-agent',
        model={'provider': 'openai', 'model': 'gpt-4o'},
        tools=['get_weather', 'send_email'],
        policies={
            'max_steps': 8,
            'tool_behavior': {
                'get_weather': {'stop_on_first_use': True},   # Halts after execution
                'send_email': {'stop_on_first_use': False}    # Continues to LLM
            }
        }
    )

    # Test 1: get_weather halts and returns raw output
    print("--- Test 1: get_weather (should halt) ---")
    result1 = await agent.run({'message': "What's the weather in NYC?"})
    print(f"Result type: {type(result1)}")
    print(f"Result: {result1}")
    # Expected: Dict with weather data

    # Test 2: send_email continues to LLM for natural language response
    print("\n--- Test 2: send_email (should continue) ---")
    result2 = await agent.run({
        'message': "Send an email to alice@example.com with subject 'Meeting Tomorrow'"
    })
    print(f"Result type: {type(result2)}")
    print(f"Result: {result2}")
    # Expected: String with natural language confirmation


async def example_3_streaming():
    """Example 3: Streaming with stop_on_first_tool"""
    print("\n=== Example 3: Streaming with stop_on_first_tool ===\n")

    agent = Agent(
        id='weather-stream-agent',
        model={'provider': 'openai', 'model': 'gpt-4o'},
        tools=['get_weather'],
        policies={
            'tool_behavior': {
                'get_weather': {'stop_on_first_use': True}
            }
        }
    )

    # Stream events - should see tool-output-available, then finish
    print("Streaming events:")
    async for event in agent.run_stream({'message': "Weather in Boston?"}):
        print(f"  {event['type']}: {event.get('output', event.get('delta', ''))[:50]}")


async def example_4_no_policy():
    """Example 4: No policy - default behavior (continues to LLM)"""
    print("\n=== Example 4: No policy (default behavior) ===\n")

    agent = Agent(
        id='default-agent',
        model={'provider': 'openai', 'model': 'gpt-4o'},
        tools=['get_weather'],
        # No policies specified - defaults to False
    )

    result = await agent.run({'message': "What's the weather in Seattle?"})
    print(f"Result type: {type(result)}")
    print(f"Result: {result}")
    # Expected: String with natural language response from LLM


async def main():
    """Run all examples"""
    if not os.getenv('OPENAI_API_KEY'):
        print("⚠️  OPENAI_API_KEY not set - examples will fail")
        print("Set it with: export OPENAI_API_KEY='sk-...'")
        return

    await example_1_global_stop()
    await example_2_per_tool_behavior()
    await example_3_streaming()
    await example_4_no_policy()

    print("\n✅ All examples complete!")


if __name__ == '__main__':
    asyncio.run(main())
