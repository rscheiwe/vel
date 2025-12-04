"""
Structured Output Streaming Example

Demonstrates how to use `output_type` with `run_stream()` to get
progressive structured data as it streams - not just at the end.

Two modes:
1. Array mode (List[X]) - emits `data-object-element` for each validated item
2. Object mode (single model) - emits `data-object-partial` for field updates

Both emit `data-object-complete` with the final validated output.
"""

import asyncio
from typing import List
from pydantic import BaseModel, Field
from vel import Agent


# =============================================================================
# Schema Definitions
# =============================================================================

class AIAgentIdea(BaseModel):
    """Schema for an AI agent idea."""
    name: str = Field(description="A catchy name for the AI agent")
    tagline: str = Field(description="A short tagline describing the agent")
    use_case: str = Field(description="Primary use case for this agent")
    unique_value: str = Field(description="What makes this agent unique")


class WeatherResponse(BaseModel):
    """Schema for a weather response."""
    city: str = Field(description="City name")
    country: str = Field(description="Country name")
    temperature: float = Field(description="Temperature in Celsius")
    conditions: str = Field(description="Weather conditions (sunny, cloudy, etc)")
    humidity: int = Field(description="Humidity percentage")
    wind_speed: float = Field(description="Wind speed in km/h")


# =============================================================================
# Array Streaming Example
# =============================================================================

async def array_streaming_example():
    """
    Demonstrates streaming an array of objects.

    When output_type is List[X], Vel emits:
    - `data-object-element` for each validated array item as it completes
    - `data-object-complete` with the full validated array at the end
    """
    print("\n" + "=" * 60)
    print("ARRAY STREAMING EXAMPLE")
    print("=" * 60)

    agent = Agent(
        id='agent-idea-generator',
        model={'provider': 'openai', 'model': 'gpt-4o-mini'},
        output_type=List[AIAgentIdea],  # Array mode - streams elements one-by-one
        instruction="You are a creative AI product strategist. Generate innovative AI agent ideas."
    )

    elements_received = []

    print("\nGenerating 3 AI agent ideas (streaming)...\n")

    async for event in agent.run_stream({'message': 'Generate 3 unique AI agent ideas'}):
        event_type = event.get('type')

        if event_type == 'text-delta':
            # Raw JSON tokens - useful for showing typing indicator
            print(event.get('delta', ''), end='', flush=True)

        elif event_type == 'data-object-element':
            # A complete array element has been validated!
            data = event.get('data', {})
            index = data.get('index', 0)
            element = data.get('element', {})
            elements_received.append(element)

            print(f"\n\n✅ Element {index} validated:")
            print(f"   Name: {element.get('name')}")
            print(f"   Tagline: {element.get('tagline')}")

        elif event_type == 'data-object-complete':
            # Final validated array
            data = event.get('data', {})
            final_array = data.get('object', [])
            mode = data.get('mode')

            print(f"\n\n🎉 Streaming complete! Mode: {mode}")
            print(f"   Total elements: {len(final_array)}")

    print(f"\n📊 Received {len(elements_received)} elements progressively")
    return elements_received


# =============================================================================
# Object Streaming Example
# =============================================================================

async def object_streaming_example():
    """
    Demonstrates streaming a single object with partial updates.

    When output_type is a single Pydantic model, Vel emits:
    - `data-object-partial` as object fields are parsed
    - `data-object-complete` with the final validated object
    """
    print("\n" + "=" * 60)
    print("OBJECT STREAMING EXAMPLE")
    print("=" * 60)

    agent = Agent(
        id='weather-agent',
        model={'provider': 'openai', 'model': 'gpt-4o-mini'},
        output_type=WeatherResponse,  # Object mode - streams partial updates
        instruction="You provide weather information. Always respond with realistic weather data."
    )

    partial_updates = []

    print("\nGetting weather for Tokyo (streaming)...\n")

    async for event in agent.run_stream({'message': 'What is the current weather in Tokyo, Japan?'}):
        event_type = event.get('type')

        if event_type == 'text-delta':
            # Raw JSON tokens
            print(event.get('delta', ''), end='', flush=True)

        elif event_type == 'data-object-partial':
            # Partial object update - fields parsed so far
            data = event.get('data', {})
            partial = data.get('partial', {})
            partial_updates.append(partial)

            # Show what fields we have so far
            fields = list(partial.keys())
            print(f"\n   📝 Partial update: {fields}")

        elif event_type == 'data-object-complete':
            # Final validated object
            data = event.get('data', {})
            weather = data.get('object', {})
            mode = data.get('mode')

            print(f"\n\n🎉 Streaming complete! Mode: {mode}")
            print(f"   City: {weather.get('city')}")
            print(f"   Country: {weather.get('country')}")
            print(f"   Temperature: {weather.get('temperature')}°C")
            print(f"   Conditions: {weather.get('conditions')}")
            print(f"   Humidity: {weather.get('humidity')}%")
            print(f"   Wind: {weather.get('wind_speed')} km/h")

    print(f"\n📊 Received {len(partial_updates)} partial updates")
    return partial_updates


# =============================================================================
# Comparison: With vs Without Structured Output Streaming
# =============================================================================

async def comparison_example():
    """
    Shows the difference between regular structured output (validation at end)
    and streaming structured output (progressive events).
    """
    print("\n" + "=" * 60)
    print("COMPARISON: Progressive vs End Validation")
    print("=" * 60)

    # With structured output streaming, you get data as it arrives
    print("\n📊 With streaming: UI can update progressively")
    print("   - Show each agent card as it's validated")
    print("   - Update fields as they're parsed")
    print("   - Better UX for slow generations")

    print("\n📊 Without streaming: Wait for full response")
    print("   - All or nothing - wait for complete JSON")
    print("   - Validation errors only at the end")
    print("   - Spinner until done")


# =============================================================================
# Frontend Integration Example (pseudo-code)
# =============================================================================

def frontend_example():
    """
    Shows how to integrate with Vercel AI SDK's useChat hook.
    This is JavaScript/React code for reference.
    """
    print("\n" + "=" * 60)
    print("FRONTEND INTEGRATION (useChat)")
    print("=" * 60)

    code = '''
// React component with useChat
import { useChat } from '@ai-sdk/react';
import { useState } from 'react';

function AgentGenerator() {
  const [agents, setAgents] = useState([]);

  const { messages, sendMessage, status } = useChat({
    api: '/api/generate-agents',
    onData: (data) => {
      // Handle structured output events
      if (data.type === 'data-object-element') {
        // Add each agent as it's validated
        setAgents(prev => [...prev, data.data.element]);
      }

      if (data.type === 'data-object-partial') {
        // Update partial object (for single object mode)
        setPartialData(data.data.partial);
      }

      if (data.type === 'data-object-complete') {
        // Final validated data
        console.log('Complete:', data.data.object);
      }
    }
  });

  return (
    <div>
      <button onClick={() => sendMessage('Generate 5 agent ideas')}>
        Generate
      </button>

      {/* Cards appear one-by-one as they stream */}
      {agents.map((agent, i) => (
        <AgentCard key={i} agent={agent} />
      ))}
    </div>
  );
}
'''
    print(code)


# =============================================================================
# Main
# =============================================================================

async def main():
    """Run all examples."""
    print("\n🚀 Structured Output Streaming Examples")
    print("=" * 60)

    # Show frontend integration code
    frontend_example()

    # Run array streaming example
    await array_streaming_example()

    # Run object streaming example
    await object_streaming_example()

    # Show comparison
    await comparison_example()

    print("\n" + "=" * 60)
    print("✅ All examples complete!")
    print("=" * 60)


if __name__ == '__main__':
    asyncio.run(main())
