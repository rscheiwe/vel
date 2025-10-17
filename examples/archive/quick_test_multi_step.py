"""
Quick Test - Multi-Step Agent

Simple test to verify all tools are working correctly.
Tests each tool independently and then together in an agent.

Run: python examples/quick_test_multi_step.py
"""
import asyncio
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vel import Agent

# Import all tools
from multi_step_tools import (
    web_search_tool,
    news_search_tool,
    analyze_tool,
    decide_tool,
    provide_answer_tool
)


async def test_individual_tools():
    """Test each tool individually"""
    print("\n" + "="*60)
    print("Testing Individual Tools")
    print("="*60)

    # Test web search
    print("\n1. Testing websearch tool...")
    result = await web_search_tool._handler(
        {'query': 'python programming', 'limit': 3},
        {}
    )
    print(f"   State: {result.get('state')}")
    print(f"   Results: {len(result.get('results', []))} found")
    assert result.get('state') == 'ready', "❌ websearch failed"
    print("   ✓ websearch works")

    # Test news search
    print("\n2. Testing news tool...")
    result = await news_search_tool._handler(
        {'topic': 'artificial intelligence', 'limit': 3},
        {}
    )
    print(f"   State: {result.get('state')}")
    print(f"   Items: {len(result.get('items', []))} found")
    assert result.get('state') == 'ready', "❌ news failed"
    print("   ✓ news works")

    # Test analyze
    print("\n3. Testing analyze tool...")
    result = await analyze_tool._handler(
        {'problem': 'build a web app', 'approach': 'systematic'},
        {}
    )
    print(f"   State: {result.get('state')}")
    print(f"   Breakdown: {result.get('breakdown', '')[:60]}...")
    assert result.get('state') == 'ready', "❌ analyze failed"
    print("   ✓ analyze works")

    # Test decide
    print("\n4. Testing decide tool...")
    result = await decide_tool._handler(
        {
            'options': ['React', 'Vue', 'Angular'],
            'criteria': ['performance', 'learning curve'],
            'context': 'Building a dashboard'
        },
        {}
    )
    print(f"   State: {result.get('state')}")
    print(f"   Decision: {result.get('decision')}")
    assert result.get('state') == 'ready', "❌ decide failed"
    print("   ✓ decide works")

    # Test provide answer
    print("\n5. Testing provideAnswer tool...")
    result = await provide_answer_tool._handler(
        {
            'answer': 'This is a test answer.',
            'steps': [
                {'step': 'Test step', 'reasoning': 'Testing', 'result': 'Success'}
            ],
            'confidence': 0.95
        },
        {}
    )
    print(f"   State: {result.get('state')}")
    print(f"   Answer length: {len(result.get('answer', ''))}")
    print(f"   Confidence: {result.get('confidence')}")
    assert result.get('state') == 'ready', "❌ provideAnswer failed"
    print("   ✓ provideAnswer works")

    print("\n" + "="*60)
    print("✓ All tools working correctly!")
    print("="*60)


async def test_agent():
    """Test agent with all tools"""
    print("\n" + "="*60)
    print("Testing Multi-Step Agent")
    print("="*60 + "\n")

    # Check if OPENAI_API_KEY is set
    if not os.getenv('OPENAI_API_KEY'):
        print("⚠️  Skipping agent test - OPENAI_API_KEY not set")
        print("   (This is optional - individual tool tests already passed)")
        return

    agent = Agent(
        id='test-agent',
        model={'provider': 'openai', 'model': 'gpt-4o'},
        tools=['websearch', 'news', 'analyze', 'decide', 'provideAnswer'],
        policies={'max_steps': 3},  # Limit for quick test
        generation_config={
            'temperature': 0.7,
            'max_tokens': 500
        }
    )

    query = "What are the top 3 programming languages in 2024?"
    print(f"Query: {query}\n")
    print("Events:")
    print("-" * 60)

    step_count = 0
    tool_calls = []

    async for event in agent.run_stream({'message': query}):
        event_type = event.get('type')

        if event_type == 'start-step':
            step_count += 1
            print(f"[Step {step_count} Start]")

        elif event_type == 'finish-step':
            print(f"[Step {step_count} Finish]")

        elif event_type == 'tool-input-available':
            tool_name = event.get('toolName')
            tool_calls.append(tool_name)
            print(f"  → Tool: {tool_name}")

        elif event_type == 'tool-output-available':
            output = event.get('output', {})
            state = output.get('state', 'unknown')
            print(f"  ← State: {state}")

        elif event_type == 'text-delta':
            # Agent is thinking
            pass

        elif event_type == 'finish-message':
            finish_reason = event.get('finishReason')
            print(f"\n[Finished: {finish_reason}]")

        elif event_type == 'error':
            error_msg = event.get('error')
            print(f"\n[Error: {error_msg}]")

    print("-" * 60)
    print(f"\nSummary:")
    print(f"  Total Steps: {step_count}")
    print(f"  Tools Used: {', '.join(tool_calls)}")

    print("\n" + "="*60)
    print("✓ Agent test complete!")
    print("="*60)


async def main():
    """Run all tests"""
    try:
        print("\n" + "="*60)
        print(" Quick Test - Multi-Step Agent Tools")
        print("="*60)

        # Test individual tools
        await test_individual_tools()

        # Test agent
        await test_agent()

        print("\n" + "="*60)
        print(" ✓ All tests passed!")
        print("="*60 + "\n")

    except AssertionError as e:
        print(f"\n❌ Test failed: {e}\n")
        raise
    except Exception as e:
        print(f"\n❌ Error: {e}\n")
        import traceback
        traceback.print_exc()
        raise


if __name__ == '__main__':
    asyncio.run(main())
