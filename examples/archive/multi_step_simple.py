"""
Simple Multi-Step Agent Example

Tests basic multi-step flow:
- Agent receives question
- Uses websearch tool
- Uses news tool (optional)
- Provides final answer

For debugging: This is the simplest multi-step pattern.
"""
import asyncio
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vel import Agent

# Import tools
from multi_step_tools import (
    web_search_tool,
    news_search_tool,
    provide_answer_tool
)


async def main():
    """Simple query that should trigger websearch and news"""

    agent = Agent(
        id='simple-multi-step',
        model={'provider': 'openai', 'model': 'gpt-4o'},
        tools=['websearch', 'news', 'provideAnswer'],
        policies={'max_steps': 5},
        generation_config={'temperature': 0.7}
    )

    query = "What are the latest trends in artificial intelligence?"

    print(f"\n{'='*70}")
    print(f"Simple Multi-Step Agent Test")
    print(f"{'='*70}")
    print(f"Query: {query}\n")

    step_count = 0

    async for event in agent.run_stream({'message': query}):
        event_type = event.get('type')

        if event_type == 'start-step':
            step_count += 1
            print(f"\n→ Step {step_count}")

        elif event_type == 'tool-input-available':
            tool_name = event.get('toolName')
            tool_input = event.get('input', {})
            print(f"  🔧 {tool_name}: {tool_input}")

        elif event_type == 'tool-output-available':
            state = event.get('output', {}).get('state')
            print(f"  ✓ Output state: {state}")

        elif event_type == 'text-delta':
            print(event.get('delta', ''), end='', flush=True)

        elif event_type == 'finish-message':
            print(f"\n\n{'='*70}")
            print(f"✓ Complete - {step_count} steps")
            print(f"{'='*70}\n")

        elif event_type == 'error':
            print(f"\n❌ Error: {event.get('error')}\n")


if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nInterrupted")
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()
