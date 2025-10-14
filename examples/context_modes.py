"""
Demonstrates three context manager modes:
1. Default (full memory) - remembers all conversation history
2. Stateless (no memory) - each call is independent
3. Limited history - only remembers last N messages
"""
import asyncio
from dotenv import load_dotenv
from vel import Agent, ContextManager, StatelessContextManager

load_dotenv()

async def test_default_memory():
    """Default mode - full conversation memory with sessions"""
    print("=== DEFAULT MODE (Full Memory with Sessions) ===\n")

    agent = Agent(
        id='chat-agent',
        model={'provider': 'openai', 'model': 'gpt-4o'},
        session_storage='memory'  # Store sessions in memory
    )

    session_id = 'alice-session'

    # First call
    print("User: My name is Alice")
    answer1 = await agent.run({'message': 'My name is Alice'}, session_id=session_id)
    print(f"Agent: {answer1}\n")

    # Second call - should remember Alice
    print("User: What is my name?")
    answer2 = await agent.run({'message': 'What is my name?'}, session_id=session_id)
    print(f"Agent: {answer2}\n")
    print()


async def test_stateless_mode():
    """Stateless mode - no memory between calls"""
    print("=== STATELESS MODE (No Memory) ===\n")

    agent = Agent(
        id='chat-agent',
        model={'provider': 'openai', 'model': 'gpt-4o'},
        context_manager=StatelessContextManager(),
        session_storage='memory'
    )

    session_id = 'bob-session'

    # First call
    print("User: My name is Bob")
    answer1 = await agent.run({'message': 'My name is Bob'}, session_id=session_id)
    print(f"Agent: {answer1}\n")

    # Second call - should NOT remember Bob (stateless)
    print("User: What is my name?")
    answer2 = await agent.run({'message': 'What is my name?'}, session_id=session_id)
    print(f"Agent: {answer2}\n")
    print()


async def test_limited_history():
    """Limited history - only remembers last N messages"""
    print("=== LIMITED HISTORY MODE (max_history=4) ===\n")

    agent = Agent(
        id='chat-agent',
        model={'provider': 'openai', 'model': 'gpt-4o'},
        context_manager=ContextManager(max_history=4),  # Keep last 4 messages (~2 turns)
        session_storage='memory'
    )

    session_id = 'charlie-session'

    # First call
    print("User: My name is Charlie")
    answer1 = await agent.run({'message': 'My name is Charlie'}, session_id=session_id)
    print(f"Agent: {answer1}\n")

    # Second call - fills up the 2-message buffer
    print("User: I like pizza")
    answer2 = await agent.run({'message': 'I like pizza'}, session_id=session_id)
    print(f"Agent: {answer2}\n")

    # Third call - should remember pizza but forget Charlie (oldest message dropped)
    print("User: What do I like to eat?")
    answer3 = await agent.run({'message': 'What do I like to eat?'}, session_id=session_id)
    print(f"Agent: {answer3}\n")

    # Fourth call - should NOT remember Charlie anymore
    print("User: What is my name?")
    answer4 = await agent.run({'message': 'What is my name?'}, session_id=session_id)
    print(f"Agent: {answer4}\n")
    print()


async def main():
    print("Testing different context manager modes...\n")
    print("=" * 60)
    print()

    await test_default_memory()
    print("=" * 60)
    print()

    await test_stateless_mode()
    print("=" * 60)
    print()

    await test_limited_history()
    print("=" * 60)


if __name__ == '__main__':
    asyncio.run(main())
