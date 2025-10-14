"""Test both streaming and non-streaming modes"""
import asyncio
from dotenv import load_dotenv
from vel import Agent, ToolSpec, register_tool

# Load environment variables from .env file
load_dotenv()

# Register a custom weather tool
def get_weather_handler(input: dict, ctx: dict) -> dict:
    """Dummy weather tool - returns fake weather data"""
    city = input.get('city', 'Unknown')
    print(f"  [TOOL CALLED] get_weather(city='{city}')")
    return {
        'temp_f': 72.5,
        'condition': 'sunny',
        'city': city
    }

weather_tool = ToolSpec(
    name='get_weather',
    input_schema={
        'type': 'object',
        'properties': {
            'city': {'type': 'string', 'description': 'The city to get weather for'}
        },
        'required': ['city']
    },
    output_schema={
        'type': 'object',
        'properties': {
            'temp_f': {'type': 'number'},
            'condition': {'type': 'string'},
            'city': {'type': 'string'}
        },
        'required': ['temp_f', 'condition', 'city']
    },
    handler=get_weather_handler
)

register_tool(weather_tool)

async def test_streaming():
    """Test streaming mode - should see text-delta events"""
    print("=== TESTING STREAMING MODE ===\n")
    agent = Agent(
        id='chat-general:v1',
        model={'provider':'openai','model':'gpt-4o'},
        tools=['get_weather'],
        policies={'max_steps':8}
    )

    print("Streaming events:")
    async for event in agent.run_stream({'message':'What is the weather in San Francisco?'}):
        print(f"  {event}")
    print()

async def test_non_streaming():
    """Test non-streaming mode - should get final answer only"""
    print("=== TESTING NON-STREAMING MODE ===\n")
    agent = Agent(
        id='chat-general:v1',
        model={'provider':'openai','model':'gpt-4o'},
        tools=['get_weather'],
        policies={'max_steps':8}
    )

    print("Calling agent.run()...")
    answer = await agent.run({'message':'What is the weather in New York?'})
    print(f"Final answer: {answer}\n")

async def main():
    await test_streaming()
    # await test_non_streaming()

if __name__=='__main__':
    asyncio.run(main())
