#!/usr/bin/env python3
"""
Langfuse Observability Integration Example

This example demonstrates how to use Langfuse for tracing and observability
in Vel agents. Langfuse provides full visibility into agent executions including:
- LLM generations with token usage and cost tracking
- Tool execution spans with inputs/outputs
- Nested step hierarchy
- User and session tracking

Requirements:
    pip install vel-ai[langfuse]

    # Set environment variables (or provide in config)
    export LANGFUSE_PUBLIC_KEY=pk-...
    export LANGFUSE_SECRET_KEY=sk-...
    # Optional: export LANGFUSE_HOST=https://your-instance.langfuse.com

Usage:
    python langfuse_example.py
"""

import asyncio
import os
from dotenv import load_dotenv
from vel import Agent, ToolSpec
from vel.integrations import ObservabilityConfig

# Load environment variables from .env file
load_dotenv()


# Define some tools for the agent
def get_weather(city: str) -> dict:
    """Get the current weather for a city."""
    # Simulated weather data
    weather_data = {
        'New York': {'temp': 72, 'condition': 'sunny', 'humidity': 45},
        'London': {'temp': 55, 'condition': 'cloudy', 'humidity': 80},
        'Tokyo': {'temp': 68, 'condition': 'partly cloudy', 'humidity': 60},
    }
    return weather_data.get(city, {'temp': 65, 'condition': 'unknown', 'humidity': 50})


def search_news(query: str, limit: int = 3) -> dict:
    """Search for recent news articles."""
    # Simulated news search
    return {
        'query': query,
        'results': [
            {'title': f'News about {query} - Article 1', 'source': 'Reuters'},
            {'title': f'Latest on {query} - Article 2', 'source': 'AP'},
            {'title': f'{query} update - Article 3', 'source': 'BBC'},
        ][:limit]
    }


async def basic_example():
    """Basic Langfuse integration with default settings."""
    print("\n=== Basic Langfuse Integration ===\n")

    agent = Agent(
        id='weather-agent',
        model={'provider': 'openai', 'model': 'gpt-4o-mini'},
        tools=[ToolSpec.from_function(get_weather)],

        # Enable Langfuse with minimal config (uses env vars for keys)
        observability=ObservabilityConfig(
            provider='langfuse',
        )
    )

    result = await agent.run({'message': 'What is the weather in New York?'})
    print(f"Result: {result}")


async def full_config_example():
    """Full Langfuse configuration with all options."""
    print("\n=== Full Langfuse Configuration ===\n")

    agent = Agent(
        id='research-assistant',
        model={'provider': 'openai', 'model': 'gpt-4o-mini'},
        tools=[
            ToolSpec.from_function(get_weather),
            ToolSpec.from_function(search_news),
        ],

        # Full observability configuration
        observability=ObservabilityConfig(
            provider='langfuse',
            enabled=True,

            # API keys loaded from .env file
            public_key=os.getenv('LANGFUSE_PUBLIC_KEY'),
            secret_key=os.getenv('LANGFUSE_SECRET_KEY'),
            host=os.getenv('LANGFUSE_HOST', 'https://cloud.langfuse.com'),

            # User and session tracking
            user_id='user-12345',
            session_id='session-abc',
            trace_name='research-assistant-run',

            # Metadata and tags for filtering in Langfuse UI
            tags=['production', 'research', 'v1'],
            metadata={
                'team': 'ml-platform',
                'environment': 'demo',
                'feature_flags': 'extended_context',
            },
            release='1.0.0',

            # Sampling (set to 1.0 for demo, use lower for production)
            sample_rate=1.0,

            # Capture controls (all enabled by default)
            capture_input=True,
            capture_output=True,
            capture_tool_io=True,
        )
    )

    result = await agent.run({
        'message': 'What is the weather in Tokyo?'
    })
    print(f"Result: {result}")


async def per_request_context_example():
    """Demonstrate per-request observability context overrides."""
    print("\n=== Per-Request Context Override ===\n")

    # Create agent with base config
    agent = Agent(
        id='support-agent',
        model={'provider': 'openai', 'model': 'gpt-4o-mini'},
        tools=[ToolSpec.from_function(get_weather)],
        observability=ObservabilityConfig(
            provider='langfuse',
            tags=['support'],
            release='1.0.0',
        )
    )

    # Simulate different user requests
    users = [
        {'user_id': 'user-001', 'tier': 'premium'},
        {'user_id': 'user-002', 'tier': 'free'},
    ]

    for user in users:
        print(f"\nProcessing request for {user['user_id']} ({user['tier']} tier)...")

        result = await agent.run(
            {'message': 'What is the weather in London?'},
            # Per-request context override
            observability_context={
                'user_id': user['user_id'],
                'session_id': f"session-{user['user_id']}",
                'tags': [f"tier-{user['tier']}"],
                'metadata': {'user_tier': user['tier']},
            }
        )
        print(f"Result: {result}")


async def streaming_example():
    """Langfuse tracing with streaming responses."""
    print("\n=== Streaming with Langfuse ===\n")

    agent = Agent(
        id='streaming-agent',
        model={'provider': 'openai', 'model': 'gpt-4o-mini'},
        tools=[ToolSpec.from_function(get_weather)],
        observability=ObservabilityConfig(
            provider='langfuse',
            user_id='streaming-demo-user',
            tags=['streaming', 'demo'],
        )
    )

    print("Streaming response: ", end="", flush=True)
    async for event in agent.run_stream({'message': 'What is the weather in New York?'}):
        if event.get('type') == 'text-delta':
            print(event.get('delta', ''), end="", flush=True)
    print("\n")


async def sampling_example():
    """Demonstrate sampling for high-traffic scenarios."""
    print("\n=== Sampling Example (10% sample rate) ===\n")

    agent = Agent(
        id='high-traffic-agent',
        model={'provider': 'openai', 'model': 'gpt-4o-mini'},
        observability=ObservabilityConfig(
            provider='langfuse',
            sample_rate=0.1,  # Only trace 10% of requests
            tags=['high-traffic', 'sampled'],
        )
    )

    # In production, only ~10% of these would be traced
    for i in range(5):
        result = await agent.run({'message': f'Request {i+1}: Hello!'})
        print(f"Request {i+1} completed")


async def dict_config_example():
    """Using dict-based configuration (quick setup)."""
    print("\n=== Dict-Based Configuration ===\n")

    agent = Agent(
        id='quick-setup-agent',
        model={'provider': 'openai', 'model': 'gpt-4o-mini'},
        # Config as dict instead of ObservabilityConfig
        observability={
            'provider': 'langfuse',
            'user_id': 'dict-config-user',
            'tags': ['quick-setup'],
        }
    )

    result = await agent.run({'message': 'Hello, world!'})
    print(f"Result: {result}")


async def main():
    """Run all examples."""
    # Check for required environment variables
    if not os.getenv('LANGFUSE_PUBLIC_KEY') or not os.getenv('LANGFUSE_SECRET_KEY'):
        print("Warning: LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY not set.")
        print("Langfuse tracing will be disabled. Set these environment variables to enable tracing.")
        print()

    # Run examples
    # await basic_example()
    await full_config_example()
    # await per_request_context_example()
    # await streaming_example()
    # await dict_config_example()

    print("\n=== All examples completed ===")
    print("\nView your traces in the Langfuse dashboard!")


if __name__ == '__main__':
    asyncio.run(main())
