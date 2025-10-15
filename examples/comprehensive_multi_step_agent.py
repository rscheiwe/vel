"""
Comprehensive Multi-Step Agent Example

Demonstrates all multi-step tools working together:
- websearch: Web search for current information
- news: News search for recent headlines
- analyze: Break down complex problems
- decide: Make decisions between options
- provideAnswer: Final structured answer with citations

This example matches the Vercel AI SDK multi-step agent pattern.
"""
import asyncio
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vel import Agent

# Import all tools (registers them automatically)
from multi_step_tools import (
    web_search_tool,
    news_search_tool,
    analyze_tool,
    decide_tool,
    provide_answer_tool
)


async def run_agent(query: str, show_raw_events: bool = False):
    """
    Run the multi-step agent with a query.

    Args:
        query: User's question
        show_raw_events: If True, show all raw events; otherwise show formatted output
    """
    # Create multi-step agent
    agent = Agent(
        id='multi-step-agent',
        model={'provider': 'openai', 'model': 'gpt-4o'},
        tools=['websearch', 'news', 'analyze', 'decide', 'provideAnswer'],
        policies={'max_steps': 8},  # Maximum 8 steps like AI SDK
        generation_config={
            'temperature': 0.7,
            'max_tokens': 2000
        }
    )

    print(f"\n{'='*70}")
    print(f"Query: {query}")
    print(f"{'='*70}\n")

    # Track state
    step_count = 0
    current_step_tools = []

    # Stream events
    async for event in agent.run_stream({'message': query}):
        event_type = event.get('type')

        if show_raw_events:
            print(f"[RAW] {event}")
            continue

        # Track steps
        if event_type == 'step-start':
            step_count += 1
            current_step_tools = []
            print(f"\n┌─ Step {step_count} ─────────────────────────────────")

        elif event_type == 'step-finish':
            print(f"└─ Step {step_count} complete ─────────────────────────")

        # Show tool calls
        elif event_type == 'tool-input-available':
            tool_name = event.get('toolName')
            tool_input = event.get('input', {})
            current_step_tools.append(tool_name)

            print(f"│")
            print(f"│ 🔧 Tool: {tool_name}")

            # Format input nicely based on tool
            if tool_name == 'websearch':
                print(f"│    Query: {tool_input.get('query', 'N/A')}")
                print(f"│    Limit: {tool_input.get('limit', 5)}")

            elif tool_name == 'news':
                print(f"│    Topic: {tool_input.get('topic', 'N/A')}")
                print(f"│    Limit: {tool_input.get('limit', 5)}")

            elif tool_name == 'analyze':
                print(f"│    Problem: {tool_input.get('problem', 'N/A')}")
                print(f"│    Approach: {tool_input.get('approach', 'N/A')}")

            elif tool_name == 'decide':
                print(f"│    Context: {tool_input.get('context', 'N/A')}")
                print(f"│    Options: {len(tool_input.get('options', []))} options")
                print(f"│    Criteria: {', '.join(tool_input.get('criteria', []))}")

            elif tool_name == 'provideAnswer':
                print(f"│    Finalizing answer...")
                print(f"│    Steps: {len(tool_input.get('steps', []))}")
                print(f"│    Confidence: {tool_input.get('confidence', 0):.1%}")

        # Show tool outputs
        elif event_type == 'tool-output-available':
            tool_output = event.get('output', {})
            state = tool_output.get('state', 'unknown')

            print(f"│    State: {state}")

            # Show specific output based on what's available
            if 'results' in tool_output:  # websearch
                results = tool_output['results']
                print(f"│    Results: {len(results)} found")
                for i, result in enumerate(results[:3], 1):
                    print(f"│      {i}. {result.get('title', 'N/A')[:60]}...")

            elif 'items' in tool_output:  # news
                items = tool_output['items']
                print(f"│    News items: {len(items)} found")
                for i, item in enumerate(items[:3], 1):
                    print(f"│      {i}. {item.get('title', 'N/A')[:60]}...")

            elif 'breakdown' in tool_output:  # analyze
                print(f"│    Breakdown: {tool_output['breakdown'][:80]}...")
                print(f"│    Components: {len(tool_output.get('components', []))}")

            elif 'decision' in tool_output:  # decide
                print(f"│    Decision: {tool_output['decision'][:60]}...")
                print(f"│    Evaluations: {len(tool_output.get('evaluation', []))}")

            elif 'answer' in tool_output:  # provideAnswer
                answer = tool_output['answer']
                print(f"│")
                print(f"│ 📝 Final Answer:")
                print(f"│ {'-'*65}")

                # Print answer with proper line wrapping
                lines = answer.split('\n')
                for line in lines:
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

                print(f"│ {'-'*65}")
                print(f"│ Confidence: {tool_output.get('confidence', 0):.1%}")
                print(f"│ Total Steps: {len(tool_output.get('steps', []))}")

                # Show citations if available
                citations = tool_output.get('citations', [])
                if citations:
                    print(f"│")
                    print(f"│ 📚 Citations:")
                    for citation in citations[:5]:  # Show first 5
                        num = citation.get('number', '?')
                        title = citation.get('title', 'N/A')[:50]
                        print(f"│   [{num}] {title}...")

        # Show text streaming (if any)
        elif event_type == 'text-delta':
            delta = event.get('delta', '')
            print(delta, end='', flush=True)

        # Show completion
        elif event_type == 'finish-message':
            finish_reason = event.get('finishReason', 'unknown')
            print(f"\n\n{'='*70}")
            print(f"✓ Finished ({finish_reason}) - Total steps: {step_count}")
            print(f"{'='*70}")

        # Show errors
        elif event_type == 'error':
            error_msg = event.get('error', '')
            print(f"\n❌ Error: {error_msg}")


async def main():
    """
    Run examples demonstrating different types of queries.
    """
    print("\n" + "="*70)
    print(" Multi-Step Agent - Comprehensive Example")
    print(" Demonstrates all tools: websearch, news, analyze, decide, provideAnswer")
    print("="*70)

    # Example 1: Simple question (should use websearch + news + provideAnswer)
    print("\n\n📌 Example 1: Simple Question")
    await run_agent(
        "What are the latest trends in artificial intelligence?"
    )

    # Example 2: Complex question requiring analysis (should use analyze + websearch + provideAnswer)
    print("\n\n📌 Example 2: Complex Question with Analysis")
    await run_agent(
        "How should I approach building a full-stack AI-powered web application? "
        "What technology stack would you recommend?"
    )

    # Example 3: Decision-making question (should use decide + provideAnswer)
    print("\n\n📌 Example 3: Decision-Making Question")
    await run_agent(
        "I need to choose between React, Vue, and Svelte for my next project. "
        "The criteria are: learning curve, performance, and ecosystem. "
        "What would you recommend?"
    )

    print("\n" + "="*70)
    print(" All examples complete!")
    print("="*70 + "\n")


if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()
