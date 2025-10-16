"""
Multi-Step Agent with Analysis Tool

Tests analysis-focused multi-step flow:
- Agent receives complex problem
- Uses analyze tool to break down the problem
- Uses websearch to gather information
- Provides structured final answer

For debugging: Focus on complex problem decomposition.
"""
import asyncio
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vel import Agent

# Import tools
from multi_step_tools import (
    web_search_tool,
    analyze_tool,
    provide_answer_tool
)


async def main():
    """Complex query requiring problem analysis"""

    agent = Agent(
        id='analysis-multi-step',
        model={'provider': 'openai', 'model': 'gpt-4o'},
        tools=['websearch', 'analyze', 'provideAnswer'],
        policies={'max_steps': 6},
        generation_config={'temperature': 0.7}
    )

    query = ("How should I approach building a full-stack AI-powered web application? "
             "What technology stack would you recommend?")

    print(f"\n{'='*70}")
    print(f"Multi-Step Agent - Analysis Pattern")
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

            print(f"  🔧 {tool_name}")

            if tool_name == 'analyze':
                print(f"     Problem: {tool_input.get('problem', 'N/A')[:60]}...")
                print(f"     Approach: {tool_input.get('approach', 'N/A')}")
            elif tool_name == 'websearch':
                print(f"     Query: {tool_input.get('query', 'N/A')}")
            elif tool_name == 'provideAnswer':
                print(f"     Finalizing with {len(tool_input.get('steps', []))} steps")

        elif event_type == 'tool-output-available':
            tool_output = event.get('output', {})
            state = tool_output.get('state', 'unknown')
            print(f"  ✓ State: {state}")

            if 'breakdown' in tool_output:
                breakdown = tool_output['breakdown']
                components = tool_output.get('components', [])
                print(f"     Breakdown: {breakdown[:60]}...")
                print(f"     Components: {len(components)}")
            elif 'results' in tool_output:
                results = tool_output['results']
                print(f"     Results found: {len(results)}")
            elif 'answer' in tool_output:
                answer = tool_output['answer']
                print(f"\n  📝 Final Answer:\n  {'-'*66}")
                # Print with proper wrapping
                for line in answer.split('\n')[:5]:  # First 5 lines
                    print(f"  {line[:64]}")
                print(f"  {'-'*66}")
                print(f"  Confidence: {tool_output.get('confidence', 0):.1%}")

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
