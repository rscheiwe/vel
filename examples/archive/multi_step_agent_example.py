"""
Multi-Step Agent Example - AI SDK Pattern Compatible

Demonstrates how to build a multi-step agent in Vel that matches
Vercel AI SDK's multi-step agent pattern with proper event emissions.

This example shows:
1. Step-start and finish-step events for multi-step reasoning
2. Tools with 'state' field in output (AI SDK compatible)
3. Final answer tool that terminates the agent loop
4. Frontend-compatible event streaming

Run: python examples/multi_step_agent_example.py
"""
import asyncio
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vel import Agent
# Import tools (registers them)
from multi_step_tools import web_search_tool, provide_answer_tool

async def main():
    # Create multi-step agent with AI SDK compatible tools
    agent = Agent(
        id='multi-step-agent',
        model={'provider': 'openai', 'model': 'gpt-4o'},
        tools=['websearch', 'provideAnswer'],
        policies={'max_steps': 8},  # Maximum 8 steps like AI SDK example
        generation_config={
            'temperature': 0.7,
            'max_tokens': 2000
        }
    )

    print("Multi-Step Agent Example (AI SDK Pattern)")
    print("=" * 50)
    print()

    # Example query
    query = "What are the latest trends in artificial intelligence?"

    print(f"Query: {query}")
    print()
    print("Streaming events:")
    print("-" * 50)

    # Stream events
    step_count = 0
    async for event in agent.run_stream({'message': query}):
        event_type = event.get('type')

        # Track steps
        if event_type == 'start-step':
            step_count += 1
            print(f"\n[Step {step_count} Start]")

        elif event_type == 'finish-step':
            print(f"[Step {step_count} Finish]\n")

        # Show tool calls
        elif event_type == 'tool-input-available':
            tool_name = event.get('toolName')
            tool_input = event.get('input')
            print(f"  Tool Call: {tool_name}")
            print(f"  Input: {tool_input}")

        # Show tool outputs
        elif event_type == 'tool-output-available':
            tool_output = event.get('output', {})
            state = tool_output.get('state', 'unknown')
            print(f"  Tool Output State: {state}")

            # Show answer if it's the final tool
            if 'answer' in tool_output:
                print(f"\n  Final Answer:")
                print(f"  {tool_output['answer']}")
                print(f"\n  Confidence: {tool_output.get('confidence', 0)}")
                print(f"  Steps taken: {len(tool_output.get('steps', []))}")

        # Show text streaming
        elif event_type == 'text-delta':
            delta = event.get('delta', '')
            print(delta, end='', flush=True)

        # Show completion
        elif event_type == 'finish-message':
            finish_reason = event.get('finishReason', 'unknown')
            print(f"\n\n[Finished: {finish_reason}]")

        # Show errors
        elif event_type == 'error':
            error_msg = event.get('error', '')
            print(f"\n[Error: {error_msg}]")

    print()
    print("=" * 50)
    print(f"Total steps: {step_count}")

if __name__ == '__main__':
    asyncio.run(main())
