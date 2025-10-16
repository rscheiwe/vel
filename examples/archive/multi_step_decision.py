"""
Multi-Step Agent with Decision Tool

Tests decision-making multi-step flow:
- Agent receives decision question with options
- Uses decide tool to evaluate options against criteria
- Provides structured recommendation

For debugging: Focus on decision evaluation patterns.
"""
import asyncio
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vel import Agent

# Import tools
from multi_step_tools import (
    decide_tool,
    provide_answer_tool
)


async def main():
    """Decision-making query with explicit options and criteria"""

    agent = Agent(
        id='decision-multi-step',
        model={'provider': 'openai', 'model': 'gpt-4o'},
        tools=['decide', 'provideAnswer'],
        policies={'max_steps': 4},
        generation_config={'temperature': 0.7}
    )

    query = ("I need to choose between React, Vue, and Svelte for my next project. "
             "The criteria are: learning curve, performance, and ecosystem. "
             "What would you recommend?")

    print(f"\n{'='*70}")
    print(f"Multi-Step Agent - Decision Pattern")
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

            if tool_name == 'decide':
                context = tool_input.get('context', 'N/A')
                options = tool_input.get('options', [])
                criteria = tool_input.get('criteria', [])

                print(f"     Context: {context[:60]}...")
                print(f"     Options: {len(options)} - {', '.join(options)}")
                print(f"     Criteria: {', '.join(criteria)}")

            elif tool_name == 'provideAnswer':
                print(f"     Finalizing recommendation")

        elif event_type == 'tool-output-available':
            tool_output = event.get('output', {})
            state = tool_output.get('state', 'unknown')
            print(f"  ✓ State: {state}")

            if 'decision' in tool_output:
                decision = tool_output['decision']
                evaluation = tool_output.get('evaluation', [])
                reasoning = tool_output.get('reasoning', 'N/A')

                print(f"     Decision: {decision}")
                print(f"     Reasoning: {reasoning[:60]}...")
                print(f"     Evaluations: {len(evaluation)}")

                # Show evaluation scores
                for eval_item in evaluation:
                    option = eval_item.get('option', 'N/A')
                    score = eval_item.get('score', 0)
                    print(f"       • {option}: {score:.1f}/10")

            elif 'answer' in tool_output:
                answer = tool_output['answer']
                print(f"\n  📝 Final Recommendation:\n  {'-'*66}")
                # Print with proper wrapping
                for line in answer.split('\n')[:8]:
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
