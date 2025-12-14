"""
Test to debug tool calling behavior.

This test checks if tools are being called more than expected.
"""
import asyncio
import pytest
from vel import Agent, ToolSpec


# Simple mock tool that tracks how many times it's called
call_count = 0

async def mock_websearch(input: dict, ctx: dict = None) -> dict:
    """Mock websearch tool that tracks calls."""
    global call_count
    call_count += 1
    query = input.get('query', str(input))
    print(f"\n[TOOL CALLED] websearch called {call_count} time(s) with query: {query}")
    return {
        "results": [
            {"title": "Result 1", "snippet": f"This is a test result about {query}"},
            {"title": "Result 2", "snippet": f"Another result for {query}"},
        ]
    }


@pytest.fixture
def reset_call_count():
    """Reset the global call counter before each test."""
    global call_count
    call_count = 0
    yield
    call_count = 0


@pytest.mark.asyncio
async def test_tool_called_once_for_simple_query(reset_call_count):
    """
    Test that a simple query only triggers one tool call.

    This is the behavior we expect - the agent should:
    1. Call websearch once
    2. Use the results to answer
    3. NOT call websearch again
    """
    global call_count

    websearch_tool = ToolSpec.from_function(mock_websearch)

    agent = Agent(
        id='test-agent',
        model={'provider': 'openai', 'model': 'gpt-4o'},
        tools=[websearch_tool],
    )

    # Simple query that should only need one search
    events = []
    async for event in agent.run_stream({'message': 'What is the weather in New York?'}):
        events.append(event)
        print(f"Event: {event.get('type')}")

    print(f"\nTotal websearch calls: {call_count}")
    print(f"Total events: {len(events)}")

    # We expect websearch to be called exactly once
    assert call_count == 1, f"Expected 1 websearch call, got {call_count}"


@pytest.mark.asyncio
async def test_tool_loop_with_policies(reset_call_count):
    """
    Test with reset_tool_choice policy enabled.
    """
    global call_count

    websearch_tool = ToolSpec.from_function(mock_websearch)

    agent = Agent(
        id='test-agent',
        model={'provider': 'openai', 'model': 'gpt-4o'},
        tools=[websearch_tool],
        policies={
            'max_steps': 8,
            'reset_tool_choice': True,
        }
    )

    events = []
    async for event in agent.run_stream({'message': 'What is the weather in New York?'}):
        events.append(event)
        print(f"Event: {event.get('type')}")

    print(f"\nTotal websearch calls: {call_count}")

    # We expect websearch to be called exactly once
    assert call_count == 1, f"Expected 1 websearch call, got {call_count}"


@pytest.mark.asyncio
async def test_tool_loop_without_policies(reset_call_count):
    """
    Test WITHOUT any policies - baseline behavior.
    """
    global call_count

    websearch_tool = ToolSpec(
        name='mock_websearch',
        description='Search the web for information',
        input_schema={
            'type': 'object',
            'properties': {
                'query': {'type': 'string', 'description': 'The search query'}
            },
            'required': ['query']
        },
        output_schema={},
        handler=mock_websearch
    )

    agent = Agent(
        id='test-agent',
        model={'provider': 'openai', 'model': 'gpt-4o'},
        tools=[websearch_tool],
        # No policies - default behavior
    )

    events = []
    tool_calls = []
    async for event in agent.run_stream({'message': 'What is the weather in New York?'}):
        events.append(event)
        if event.get('type') == 'tool-input-available':
            tool_calls.append(event)
        if event.get('type') == 'error':
            print(f"ERROR: {event}")
        print(f"Event: {event.get('type')}")

    print(f"\nTotal websearch calls: {call_count}")
    print(f"Tool call events: {len(tool_calls)}")

    # Log for debugging - don't assert, just observe
    if call_count > 1:
        print(f"WARNING: Tool was called {call_count} times!")


@pytest.mark.asyncio
async def test_compare_with_and_without_prompt_context_manager(reset_call_count):
    """
    Test if using PromptContextManager affects tool calling behavior.

    This tests our recent changes to support dynamic prompts.
    """
    from vel import PromptTemplate
    from vel.prompts import PromptContextManager
    from vel.core import ContextManager

    global call_count

    websearch_tool = ToolSpec.from_function(mock_websearch)

    # Test 1: Default ContextManager (no prompt)
    print("\n--- Test with default ContextManager ---")
    call_count = 0

    agent1 = Agent(
        id='test-agent-default-ctx',
        model={'provider': 'openai', 'model': 'gpt-4o'},
        tools=[websearch_tool],
    )

    async for event in agent1.run_stream({'message': 'Search for Python tutorials'}):
        if event.get('type') == 'tool-input-available':
            print(f"Tool call: {event.get('tool_name')}")

    calls_without_prompt = call_count
    print(f"Calls with default ContextManager: {calls_without_prompt}")

    # Test 2: PromptContextManager (with prompt template)
    print("\n--- Test with PromptContextManager ---")
    call_count = 0

    template = PromptTemplate(
        id='test-prompt',
        system='You are a helpful assistant.'
    )

    agent2 = Agent(
        id='test-agent-prompt-ctx',
        model={'provider': 'openai', 'model': 'gpt-4o'},
        tools=[websearch_tool],
        prompt=template,
    )

    async for event in agent2.run_stream({'message': 'Search for Python tutorials'}):
        if event.get('type') == 'tool-input-available':
            print(f"Tool call: {event.get('tool_name')}")

    calls_with_prompt = call_count
    print(f"Calls with PromptContextManager: {calls_with_prompt}")

    # Compare
    print(f"\n--- Comparison ---")
    print(f"Default ContextManager: {calls_without_prompt} calls")
    print(f"PromptContextManager: {calls_with_prompt} calls")

    if calls_with_prompt != calls_without_prompt:
        print("WARNING: Different behavior between context managers!")


if __name__ == '__main__':
    # Run a quick manual test
    print("Running manual test...")
    asyncio.run(test_tool_loop_without_policies(None))
