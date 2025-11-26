"""
Dynamic Tool Creation Example

Demonstrates how to create and use tools dynamically without global registration.
Shows:
- Creating tools from Python functions using ToolSpec.from_function()
- Passing ToolSpec instances directly to agents
- Runtime tool creation (simulating UI-created tools)
- Mixing global registry tools with dynamic tools
- Flexible output validation
"""
import asyncio
from vel import Agent, ToolSpec


# Example 1: Simple dynamic tool (no output validation)
def add_numbers(x: int, y: int) -> dict:
    """Add two numbers together."""
    return {'result': x + y, 'operation': 'addition'}


# Example 2: Tool with strict output validation
def get_weather(city: str) -> dict:
    """Get weather information for a city."""
    return {
        'temperature': 72,
        'condition': 'sunny',
        'humidity': 65,
        'city': city
    }


# Example 3: Tool that might return unexpected data (benefits from flexible validation)
def fetch_data(key: str) -> dict:
    """Fetch arbitrary data by key."""
    # Returns different shapes based on key
    if key == 'user':
        return {'name': 'Alice', 'age': 30, 'role': 'admin'}
    elif key == 'stats':
        return {'views': 1000, 'likes': 50, 'extra_field': 'surprise!'}
    else:
        return {'error': 'Key not found', 'requested_key': key}


# Example 4: Async tool
async def async_search(query: str) -> dict:
    """Simulate async search operation."""
    await asyncio.sleep(0.1)  # Simulate network delay
    return {
        'query': query,
        'results': [
            {'title': 'Result 1', 'url': 'http://example.com/1'},
            {'title': 'Result 2', 'url': 'http://example.com/2'}
        ]
    }


async def main():
    print("=" * 70)
    print("Dynamic Tool Creation Example")
    print("=" * 70)
    print()

    # ========================================
    # Pattern 1: Simple tool (flexible output, no validation)
    # ========================================
    print("Pattern 1: Simple tool with flexible output")
    print("-" * 70)

    add_tool = ToolSpec.from_function(add_numbers)
    # output_schema = {} (no validation, accepts anything)

    agent = Agent(
        id='calculator-agent',
        model={'provider': 'openai', 'model': 'gpt-4o-mini'},
        tools=[add_tool]  # ✅ Pass directly, no registration!
    )

    result = await agent.run({'message': 'What is 15 + 27?'})
    print(f"Result: {result}")
    print()

    # ========================================
    # Pattern 2: Tool with strict output validation
    # ========================================
    print("Pattern 2: Tool with strict output validation")
    print("-" * 70)

    weather_tool = ToolSpec.from_function(
        get_weather,
        validate_output=True,
        output_schema={
            'type': 'object',
            'properties': {
                'temperature': {'type': 'number'},
                'condition': {'type': 'string', 'enum': ['sunny', 'cloudy', 'rainy', 'snowy']},
                'humidity': {'type': 'number', 'minimum': 0, 'maximum': 100},
                'city': {'type': 'string'}
            },
            'required': ['temperature', 'condition', 'city'],
            'additionalProperties': False  # Strict: no extra fields
        }
    )

    agent = Agent(
        id='weather-agent',
        model={'provider': 'openai', 'model': 'gpt-4o-mini'},
        tools=[weather_tool]
    )

    result = await agent.run({'message': 'What is the weather in San Francisco?'})
    print(f"Result: {result}")
    print()

    # ========================================
    # Pattern 3: Flexible tool (accepts any output shape)
    # ========================================
    print("Pattern 3: Flexible tool (different output shapes)")
    print("-" * 70)

    fetch_tool = ToolSpec.from_function(fetch_data)
    # output_schema = {} (flexible, accepts any shape)

    agent = Agent(
        id='data-agent',
        model={'provider': 'openai', 'model': 'gpt-4o-mini'},
        tools=[fetch_tool]
    )

    result = await agent.run({'message': 'Fetch user data'})
    print(f"Result: {result}")
    print()

    # ========================================
    # Pattern 4: Runtime tool creation (simulating UI)
    # ========================================
    print("Pattern 4: Runtime tool creation (UI simulation)")
    print("-" * 70)

    def create_tool_from_ui(user_code: str, tool_name: str, tool_desc: str):
        """
        Simulate UI allowing user to create custom tools at runtime.
        """
        # Execute user code in isolated namespace
        namespace = {}
        exec(user_code, namespace)
        handler_fn = namespace.get('handler')

        # Wrap in ToolSpec
        return ToolSpec.from_function(
            handler_fn,
            name=tool_name,
            description=tool_desc
        )

    # User creates tool via UI
    user_tool = create_tool_from_ui(
        user_code='''
def handler(base: int, exponent: int) -> dict:
    """Calculate base raised to exponent."""
    return {'result': base ** exponent}
''',
        tool_name='power',
        tool_desc='Calculate base raised to exponent'
    )

    # Agent uses it immediately (no restart needed!)
    agent = Agent(
        id='power-agent',
        model={'provider': 'openai', 'model': 'gpt-4o-mini'},
        tools=[user_tool]
    )

    result = await agent.run({'message': 'What is 2 to the power of 8?'})
    print(f"Result: {result}")
    print()

    # ========================================
    # Pattern 5: Mix global registry + dynamic tools
    # ========================================
    print("Pattern 5: Mix global registry + dynamic tools")
    print("-" * 70)

    # Assume 'websearch' is in global registry (if available)
    agent = Agent(
        id='hybrid-agent',
        model={'provider': 'openai', 'model': 'gpt-4o-mini'},
        tools=[
            # 'websearch',       # Global registry (string)
            add_tool,          # Dynamic tool (ToolSpec)
            weather_tool       # Dynamic tool (ToolSpec)
        ]
    )

    result = await agent.run({'message': 'What is 10 + 5, and what is the weather?'})
    print(f"Result: {result}")
    print()

    # ========================================
    # Pattern 6: Async tool
    # ========================================
    print("Pattern 6: Async tool")
    print("-" * 70)

    search_tool = ToolSpec.from_function(async_search)

    agent = Agent(
        id='search-agent',
        model={'provider': 'openai', 'model': 'gpt-4o-mini'},
        tools=[search_tool]
    )

    result = await agent.run({'message': 'Search for AI agents'})
    print(f"Result: {result}")
    print()

    print("=" * 70)
    print("Summary")
    print("=" * 70)
    print("✅ All tools created dynamically (no global registration)")
    print("✅ Tools scoped to agent instance (no global state pollution)")
    print("✅ Output validation flexible by default")
    print("✅ Can enable strict validation when needed")
    print("✅ Supports sync and async tools")
    print("✅ Enables runtime tool creation (for UIs)")


if __name__ == '__main__':
    asyncio.run(main())
