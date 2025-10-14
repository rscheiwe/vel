"""Test Anthropic Claude provider"""
import asyncio
from dotenv import load_dotenv
from vel import Agent

load_dotenv()

async def test_claude_streaming():
    """Test Claude with streaming"""
    print("=== TESTING CLAUDE STREAMING ===\n")

    agent = Agent(
        id='claude-agent',
        model={'provider': 'anthropic', 'model': 'claude-sonnet-4-20250514'}
    )

    print("Streaming response:")
    async for event in agent.run_stream({'message': 'Write a haiku about AI agents'}):
        if event['type'] == 'text-delta':
            print(event['delta'], end='', flush=True)
        elif event['type'] == 'finish-message':
            print(f"\n\nFinished: {event['finishReason']}")
    print()

async def test_claude_non_streaming():
    """Test Claude without streaming"""
    print("=== TESTING CLAUDE NON-STREAMING ===\n")

    agent = Agent(
        id='claude-agent',
        model={'provider': 'anthropic', 'model': 'claude-3-5-haiku-20241022'}
    )

    print("Calling agent.run()...")
    answer = await agent.run({'message': 'What is 2+2? Answer briefly.'})
    print(f"Answer: {answer}\n")

async def main():
    await test_claude_streaming()
    await test_claude_non_streaming()

if __name__ == '__main__':
    asyncio.run(main())
