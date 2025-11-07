"""
Demonstrates stateless message array usage - client manages conversation history.

This pattern is useful when:
- Building multi-tenant applications where history is stored per-user in a DB
- Integrating Vel as a library where the parent app manages state
- You want full control over conversation history and persistence

Two patterns are shown:
1. Single-turn with history (most common)
2. Multi-turn tool calling with history (agent uses tools within a single request)
"""
import asyncio
from dotenv import load_dotenv
from vel import Agent

load_dotenv()


async def example_single_turn():
    """
    Client manages full conversation history.
    Each request includes complete message history.
    """
    print("=== STATELESS MODE: Single Turn ===\n")

    agent = Agent(
        id='chat-agent',
        model={'provider': 'openai', 'model': 'gpt-4o'}
    )

    # Simulate conversation history stored in client's database
    conversation_history = []

    # First request
    print("User: My name is Alice")
    conversation_history.append({'role': 'user', 'content': 'My name is Alice'})

    response1 = await agent.run({'messages': conversation_history})
    print(f"Agent: {response1}\n")

    # Store assistant response in client DB
    conversation_history.append({'role': 'assistant', 'content': response1})

    # Second request - client sends full history
    print("User: What is my name?")
    conversation_history.append({'role': 'user', 'content': 'What is my name?'})

    response2 = await agent.run({'messages': conversation_history})
    print(f"Agent: {response2}\n")

    # Store assistant response in client DB
    conversation_history.append({'role': 'assistant', 'content': response2})

    print(f"Final conversation history has {len(conversation_history)} messages")
    print()


async def example_streaming():
    """
    Streaming with stateless messages array.
    """
    print("=== STATELESS MODE: Streaming ===\n")

    agent = Agent(
        id='chat-agent',
        model={'provider': 'openai', 'model': 'gpt-4o'}
    )

    # Conversation history from client DB
    conversation_history = [
        {'role': 'user', 'content': 'My name is Bob'},
        {'role': 'assistant', 'content': 'Nice to meet you, Bob! How can I help you today?'}
    ]

    # New user message
    print("User: Write a haiku about my name")
    conversation_history.append({'role': 'user', 'content': 'Write a haiku about my name'})

    # Stream response
    print("Agent: ", end="", flush=True)
    full_response = ""

    async for event in agent.run_stream({'messages': conversation_history}):
        if event.get('type') == 'text-delta':
            delta = event.get('delta', '')
            print(delta, end="", flush=True)
            full_response += delta

    print("\n")

    # Store in client DB
    conversation_history.append({'role': 'assistant', 'content': full_response})


async def example_with_api_endpoint():
    """
    Typical pattern when Vel agent is behind a REST API endpoint.

    The client sends:
    POST /chat
    {
        "messages": [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi!"},
            {"role": "user", "content": "Tell me a joke"}
        ]
    }
    """
    print("=== API ENDPOINT PATTERN ===\n")

    # Simulating what happens in your FastAPI/Flask endpoint
    async def chat_endpoint(request_body: dict):
        """Your API endpoint handler"""
        agent = Agent(
            id='chat-agent',
            model={'provider': 'openai', 'model': 'gpt-4o'}
        )

        # Client sends full message history
        messages = request_body.get('messages', [])

        # Pass directly to agent - Vel handles the rest
        response = await agent.run({'messages': messages})

        return {'response': response}

    # Client request
    request = {
        'messages': [
            {'role': 'user', 'content': 'Hello'},
            {'role': 'assistant', 'content': 'Hi there! How can I help you?'},
            {'role': 'user', 'content': 'Tell me a joke'}
        ]
    }

    print("Request:", request)
    response = await chat_endpoint(request)
    print(f"\nResponse: {response}\n")


async def example_comparison():
    """
    Side-by-side comparison of stateless vs session-based patterns.
    """
    print("=== COMPARISON: Stateless vs Session-Based ===\n")

    agent = Agent(
        id='chat-agent',
        model={'provider': 'openai', 'model': 'gpt-4o-mini'}
    )

    # PATTERN 1: Stateless (client manages history)
    print("PATTERN 1: Stateless - Client Manages History")
    print("Client stores history in their own database\n")

    history = [
        {'role': 'user', 'content': 'Hello'},
        {'role': 'assistant', 'content': 'Hi! How can I help?'},
        {'role': 'user', 'content': 'Count to 3'}
    ]

    response1 = await agent.run({'messages': history})
    print(f"Response: {response1}\n")

    # PATTERN 2: Session-based (Vel manages history)
    print("\nPATTERN 2: Session-Based - Vel Manages History")
    print("Vel stores history internally using session_id\n")

    session_id = 'my-session'

    # First call
    await agent.run({'message': 'Hello'}, session_id=session_id)

    # Second call - Vel remembers first call
    response2 = await agent.run({'message': 'Count to 3'}, session_id=session_id)
    print(f"Response: {response2}\n")

    print("\nUse stateless when:")
    print("- You have your own database for history")
    print("- Building multi-tenant apps")
    print("- Need full control over persistence")
    print("\nUse session-based when:")
    print("- Quick prototyping")
    print("- Simple single-user applications")
    print("- Want Vel to handle history automatically")


async def main():
    print("Stateless Messages Array Examples")
    print("=" * 60)
    print()

    await example_single_turn()
    print("=" * 60)
    print()

    await example_streaming()
    print("=" * 60)
    print()

    await example_with_api_endpoint()
    print("=" * 60)
    print()

    await example_comparison()
    print("=" * 60)


if __name__ == '__main__':
    asyncio.run(main())
