"""
Complex Multi-Step Agent Example

Tests all tools working together in a complex reasoning chain:
- Agent receives nuanced, multi-faceted question
- Uses websearch for current information
- Uses news for latest developments
- Uses analyze to break down the problem
- Uses decide to evaluate options
- Provides comprehensive final answer with citations

For debugging: Full reasoning chain with all tools.
"""
import asyncio
import sys
import os
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


async def main():
    """Complex multi-faceted query requiring all tools"""

    agent = Agent(
        id='complex-multi-step',
        model={'provider': 'openai', 'model': 'gpt-4o'},
        tools=['websearch', 'news', 'analyze', 'decide', 'provideAnswer'],
        policies={'max_steps': 10},
        generation_config={
            'temperature': 0.7,
            'max_tokens': 2000
        }
    )

    # Complex cryptocurrency investment question (similar to AI SDK example)
    query = ("Should I invest in cryptocurrency right now? "
             "Consider the current market trends, regulatory environment, "
             "and compare Bitcoin, Ethereum, and Solana. "
             "What would be the safest approach for a beginner?")

    print(f"\n{'='*70}")
    print(f"Complex Multi-Step Agent - All Tools")
    print(f"{'='*70}")
    print(f"Query: {query}\n")

    step_count = 0
    current_step_tools = []

    async for event in agent.run_stream({'message': query}):
        event_type = event.get('type')

        if event_type == 'step-start':
            step_count += 1
            current_step_tools = []
            print(f"\n┌─ Step {step_count} {'─'*60}")

        elif event_type == 'step-finish':
            tools_used = ', '.join(current_step_tools) if current_step_tools else 'none'
            print(f"└─ Step {step_count} complete (tools: {tools_used}) {'─'*40}")

        elif event_type == 'tool-input-available':
            tool_name = event.get('toolName')
            tool_input = event.get('input', {})
            current_step_tools.append(tool_name)

            print(f"│")
            print(f"│ 🔧 Tool: {tool_name}")

            # Show formatted input based on tool type
            if tool_name == 'websearch':
                print(f"│    Query: {tool_input.get('query', 'N/A')}")
                print(f"│    Limit: {tool_input.get('limit', 5)}")

            elif tool_name == 'news':
                print(f"│    Topic: {tool_input.get('topic', 'N/A')}")
                print(f"│    Limit: {tool_input.get('limit', 5)}")

            elif tool_name == 'analyze':
                print(f"│    Problem: {tool_input.get('problem', 'N/A')[:55]}...")
                print(f"│    Approach: {tool_input.get('approach', 'N/A')}")

            elif tool_name == 'decide':
                options = tool_input.get('options', [])
                criteria = tool_input.get('criteria', [])
                print(f"│    Context: {tool_input.get('context', 'N/A')[:50]}...")
                print(f"│    Options: {len(options)} - {', '.join(options[:3])}")
                print(f"│    Criteria: {', '.join(criteria)}")

            elif tool_name == 'provideAnswer':
                steps = tool_input.get('steps', [])
                confidence = tool_input.get('confidence', 0)
                print(f"│    Finalizing comprehensive answer...")
                print(f"│    Steps taken: {len(steps)}")
                print(f"│    Confidence: {confidence:.1%}")

        elif event_type == 'tool-output-available':
            tool_output = event.get('output', {})
            state = tool_output.get('state', 'unknown')
            print(f"│    ✓ State: {state}")

            # Show relevant output details
            if 'results' in tool_output:  # websearch
                results = tool_output['results']
                print(f"│      Found {len(results)} web results")
                for i, result in enumerate(results[:2], 1):
                    title = result.get('title', 'N/A')[:50]
                    print(f"│      {i}. {title}...")

            elif 'items' in tool_output:  # news
                items = tool_output['items']
                print(f"│      Found {len(items)} news items")
                for i, item in enumerate(items[:2], 1):
                    title = item.get('title', 'N/A')[:50]
                    print(f"│      {i}. {title}...")

            elif 'breakdown' in tool_output:  # analyze
                breakdown = tool_output['breakdown']
                components = tool_output.get('components', [])
                print(f"│      Breakdown: {breakdown[:50]}...")
                print(f"│      Components identified: {len(components)}")

            elif 'decision' in tool_output:  # decide
                decision = tool_output['decision']
                evaluation = tool_output.get('evaluation', [])
                print(f"│      Decision: {decision}")
                print(f"│      Options evaluated: {len(evaluation)}")
                for eval_item in evaluation[:3]:
                    opt = eval_item.get('option', 'N/A')
                    score = eval_item.get('score', 0)
                    print(f"│        • {opt}: {score:.1f}/10")

            elif 'answer' in tool_output:  # provideAnswer
                answer = tool_output['answer']
                citations = tool_output.get('citations', [])

                print(f"│")
                print(f"│ 📝 Final Comprehensive Answer:")
                print(f"│ {'─'*65}")

                # Print answer with wrapping
                lines = answer.split('\n')
                for line in lines[:10]:  # First 10 lines
                    if len(line) <= 60:
                        print(f"│ {line}")
                    else:
                        # Wrap long lines
                        words = line.split()
                        current_line = "│ "
                        for word in words:
                            if len(current_line) + len(word) + 1 <= 66:
                                current_line += word + " "
                            else:
                                print(current_line.rstrip())
                                current_line = "│ " + word + " "
                        if current_line.strip() != "│":
                            print(current_line.rstrip())

                if len(lines) > 10:
                    print(f"│ ... ({len(lines) - 10} more lines)")

                print(f"│ {'─'*65}")
                print(f"│ Confidence: {tool_output.get('confidence', 0):.1%}")
                print(f"│ Total Steps: {len(tool_output.get('steps', []))}")

                # Show citations
                if citations:
                    print(f"│")
                    print(f"│ 📚 Citations:")
                    for citation in citations[:5]:
                        num = citation.get('number', '?')
                        title = citation.get('title', 'N/A')[:48]
                        print(f"│   [{num}] {title}...")

        elif event_type == 'text-delta':
            # Show any additional text streaming
            delta = event.get('delta', '')
            print(delta, end='', flush=True)

        elif event_type == 'finish-message':
            finish_reason = event.get('finishReason', 'unknown')
            print(f"\n\n{'='*70}")
            print(f"✓ Finished ({finish_reason}) - Total steps: {step_count}")
            print(f"{'='*70}\n")

        elif event_type == 'error':
            error_msg = event.get('error', '')
            print(f"\n│ ❌ Error: {error_msg}\n")


if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()
