#!/usr/bin/env python3
"""
Extended Thinking Examples

Demonstrates multi-pass reasoning (Analyze -> Critique -> Refine -> Conclude)
with Vel's ReflectionController.

Requirements:
    pip install vel
    export OPENAI_API_KEY=sk-...

Usage:
    python examples/extended_thinking.py
"""

import asyncio
import os
from vel import Agent, ToolSpec
from vel.thinking import ThinkingConfig


# =============================================================================
# Example 1: Basic Extended Thinking
# =============================================================================

async def basic_thinking():
    """
    Basic usage - enable extended thinking for deeper reasoning.

    The agent will:
    1. Analyze the question
    2. Critique its analysis
    3. Refine based on critiques
    4. Provide a final answer
    """
    print("\n" + "="*60)
    print("Example 1: Basic Extended Thinking")
    print("="*60 + "\n")

    agent = Agent(
        id='basic-thinker',
        model={'provider': 'openai', 'model': 'gpt-4o'},
        thinking=ThinkingConfig(mode='reflection')
    )

    question = "What are the key factors that led to the Renaissance?"
    print(f"Question: {question}\n")
    print("-" * 40)

    current_stage = None

    async for event in agent.run_stream({'message': question}):
        event_type = event.get('type')

        # Track stage changes
        if event_type == 'data-thinking-stage':
            stage = event['data']['stage']
            if stage != current_stage:
                current_stage = stage
                print(f"\n[Stage: {stage.upper()}]")
                if 'confidence' in event['data']:
                    print(f"  Confidence: {event['data']['confidence']:.0%}")

        # Show reasoning content
        elif event_type == 'reasoning-delta':
            print(event.get('delta', ''), end='', flush=True)

        # Show final answer
        elif event_type == 'text-delta':
            print(event.get('delta', ''), end='', flush=True)

        # Show completion stats
        elif event_type == 'data-thinking-complete':
            data = event['data']
            print(f"\n\n[Thinking Complete]")
            print(f"  Steps: {data['steps']}")
            print(f"  Iterations: {data['iterations']}")
            print(f"  Final Confidence: {data['final_confidence']:.0%}")


# =============================================================================
# Example 2: Runtime Override
# =============================================================================

async def runtime_override():
    """
    Enable thinking on-demand for specific queries.

    Useful when you want thinking for complex questions
    but not for simple ones.
    """
    print("\n" + "="*60)
    print("Example 2: Runtime Override")
    print("="*60 + "\n")

    # Agent without default thinking
    agent = Agent(
        id='flexible-agent',
        model={'provider': 'openai', 'model': 'gpt-4o'}
    )

    # Simple question - no thinking needed
    print("Simple question (no thinking):")
    print("-" * 40)
    async for event in agent.run_stream({'message': 'What is 2 + 2?'}):
        if event.get('type') == 'text-delta':
            print(event.get('delta', ''), end='', flush=True)
    print("\n")

    # Complex question - enable thinking
    print("Complex question (with thinking):")
    print("-" * 40)
    async for event in agent.run_stream(
        {'message': 'Should a startup prioritize growth or profitability in its first year?'},
        thinking=ThinkingConfig(mode='reflection', max_refinements=2)
    ):
        event_type = event.get('type')
        if event_type == 'reasoning-delta':
            print(f"[thinking] {event.get('delta', '')}", end='', flush=True)
        elif event_type == 'text-delta':
            print(event.get('delta', ''), end='', flush=True)
        elif event_type == 'reasoning-end':
            print("\n" + "-" * 40)
            print("[Final Answer]")


# =============================================================================
# Example 3: Cost-Optimized (Different Models)
# =============================================================================

async def cost_optimized():
    """
    Use a cheaper model for thinking, stronger model for final answer.

    This reduces costs while maintaining quality on the final output.
    """
    print("\n" + "="*60)
    print("Example 3: Cost-Optimized (Different Models)")
    print("="*60 + "\n")

    agent = Agent(
        id='cost-optimized',
        model={'provider': 'openai', 'model': 'gpt-4o'},  # Strong model for answer
        thinking=ThinkingConfig(
            mode='reflection',
            thinking_model={'provider': 'openai', 'model': 'gpt-4o-mini'},  # Cheap model for thinking
            max_refinements=2
        )
    )

    question = "Explain the trade-offs between microservices and monolithic architecture."
    print(f"Question: {question}")
    print(f"Thinking model: gpt-4o-mini")
    print(f"Answer model: gpt-4o")
    print("-" * 40 + "\n")

    async for event in agent.run_stream({'message': question}):
        event_type = event.get('type')
        if event_type == 'reasoning-delta':
            print(event.get('delta', ''), end='', flush=True)
        elif event_type == 'reasoning-end':
            print("\n" + "-" * 40)
            print("[Final Answer (gpt-4o)]")
        elif event_type == 'text-delta':
            print(event.get('delta', ''), end='', flush=True)
        elif event_type == 'data-thinking-complete':
            print(f"\n\n[Model: {event['data'].get('thinking_model', 'unknown')}]")


# =============================================================================
# Example 4: With Tools During Thinking
# =============================================================================

async def thinking_with_tools():
    """
    Allow tool usage during thinking phases.

    The agent can gather information during analysis and refinement,
    leading to more grounded reasoning.
    """
    print("\n" + "="*60)
    print("Example 4: Thinking With Tools")
    print("="*60 + "\n")

    # Define a simple tool
    def get_current_date() -> dict:
        """Get the current date."""
        from datetime import datetime
        return {'date': datetime.now().strftime('%Y-%m-%d'), 'day': datetime.now().strftime('%A')}

    def lookup_fact(topic: str) -> dict:
        """Look up a fact about a topic."""
        facts = {
            'python': 'Python was created by Guido van Rossum and released in 1991.',
            'javascript': 'JavaScript was created by Brendan Eich in just 10 days in 1995.',
            'rust': 'Rust was first released in 2010 and focuses on memory safety.',
        }
        return {'fact': facts.get(topic.lower(), f'No fact found for {topic}')}

    agent = Agent(
        id='research-thinker',
        model={'provider': 'openai', 'model': 'gpt-4o'},
        tools=[
            ToolSpec.from_function(get_current_date),
            ToolSpec.from_function(lookup_fact),
        ],
        thinking=ThinkingConfig(
            mode='reflection',
            thinking_tools=True,  # Enable tools during thinking
            max_refinements=2
        )
    )

    question = "Compare Python and Rust for building web backends."
    print(f"Question: {question}")
    print(f"Tools available: get_current_date, lookup_fact")
    print("-" * 40 + "\n")

    async for event in agent.run_stream({'message': question}):
        event_type = event.get('type')

        if event_type == 'tool-input-available':
            print(f"\n[Tool Call: {event['toolName']}({event['input']})]")
        elif event_type == 'tool-output-available':
            print(f"[Tool Result: {event['output']}]\n")
        elif event_type == 'reasoning-delta':
            print(event.get('delta', ''), end='', flush=True)
        elif event_type == 'text-delta':
            print(event.get('delta', ''), end='', flush=True)


# =============================================================================
# Example 5: High Confidence Threshold
# =============================================================================

async def high_confidence():
    """
    Set a high confidence threshold for thorough reasoning.

    The agent will refine multiple times until it reaches 90% confidence
    or hits the max refinements limit.
    """
    print("\n" + "="*60)
    print("Example 5: High Confidence Threshold")
    print("="*60 + "\n")

    agent = Agent(
        id='thorough-thinker',
        model={'provider': 'openai', 'model': 'gpt-4o'},
        thinking=ThinkingConfig(
            mode='reflection',
            confidence_threshold=0.90,  # Require 90% confidence
            max_refinements=4  # Allow more iterations
        )
    )

    question = "What is the most ethical approach to AI development?"
    print(f"Question: {question}")
    print(f"Confidence threshold: 90%")
    print(f"Max refinements: 4")
    print("-" * 40 + "\n")

    iteration_count = 0

    async for event in agent.run_stream({'message': question}):
        event_type = event.get('type')

        if event_type == 'data-thinking-stage':
            stage = event['data']['stage']
            if stage == 'refining':
                iteration_count = event['data'].get('iteration', 0)
                confidence = event['data'].get('confidence', 0)
                print(f"\n[Refining - Iteration {iteration_count}, Confidence: {confidence:.0%}]")
        elif event_type == 'reasoning-delta':
            # Only show first 200 chars per stage to keep output manageable
            delta = event.get('delta', '')
            print(delta[:200] if len(delta) > 200 else delta, end='', flush=True)
        elif event_type == 'text-delta':
            print(event.get('delta', ''), end='', flush=True)
        elif event_type == 'data-thinking-complete':
            data = event['data']
            print(f"\n\n[Complete: {data['iterations']} iterations, {data['final_confidence']:.0%} confidence]")


# =============================================================================
# Example 6: Minimal Output (Silent Thinking)
# =============================================================================

async def silent_thinking():
    """
    Hide thinking content, only show final answer.

    Useful when you want the benefits of multi-pass reasoning
    without exposing the internal process to users.
    """
    print("\n" + "="*60)
    print("Example 6: Silent Thinking")
    print("="*60 + "\n")

    agent = Agent(
        id='silent-thinker',
        model={'provider': 'openai', 'model': 'gpt-4o'},
        thinking=ThinkingConfig(
            mode='reflection',
            show_analysis=False,
            show_critiques=False,
            show_refinements=False  # Hide all thinking content
        )
    )

    question = "What makes a good leader?"
    print(f"Question: {question}")
    print("(Thinking in background...)")
    print("-" * 40 + "\n")

    async for event in agent.run_stream({'message': question}):
        event_type = event.get('type')

        # Only reasoning-start/end are emitted, no content
        if event_type == 'reasoning-end':
            print("[Thinking complete, generating answer...]\n")
        elif event_type == 'text-delta':
            print(event.get('delta', ''), end='', flush=True)
        elif event_type == 'data-thinking-complete':
            data = event['data']
            print(f"\n\n[Stats: {data['steps']} steps, {data['final_confidence']:.0%} confidence]")


# =============================================================================
# Example 7: Progress Tracking UI
# =============================================================================

async def progress_tracking():
    """
    Build a progress indicator using stage events.

    Shows how to create a user-friendly UI that indicates
    which thinking phase is currently active.
    """
    print("\n" + "="*60)
    print("Example 7: Progress Tracking UI")
    print("="*60 + "\n")

    agent = Agent(
        id='progress-demo',
        model={'provider': 'openai', 'model': 'gpt-4o'},
        thinking=ThinkingConfig(mode='reflection', max_refinements=2)
    )

    question = "How do neural networks learn?"
    print(f"Question: {question}\n")

    stage_icons = {
        'analyzing': '🔍',
        'critiquing': '🤔',
        'refining': '✨',
        'concluding': '📝'
    }

    stage_labels = {
        'analyzing': 'Analyzing the question...',
        'critiquing': 'Reviewing reasoning...',
        'refining': 'Refining analysis...',
        'concluding': 'Preparing answer...'
    }

    async for event in agent.run_stream({'message': question}):
        event_type = event.get('type')

        if event_type == 'data-thinking-stage':
            stage = event['data']['stage']
            step = event['data']['step']
            icon = stage_icons.get(stage, '•')
            label = stage_labels.get(stage, stage)

            # Build progress bar
            progress = '█' * step + '░' * (6 - step)

            if 'confidence' in event['data']:
                conf = event['data']['confidence']
                print(f"\r{icon} [{progress}] {label} ({conf:.0%} confident)    ", end='', flush=True)
            else:
                print(f"\r{icon} [{progress}] {label}                         ", end='', flush=True)

        elif event_type == 'reasoning-end':
            print(f"\r✅ [██████] Thinking complete!                              ")
            print("-" * 40)
            print("Answer:\n")

        elif event_type == 'text-delta':
            print(event.get('delta', ''), end='', flush=True)


# =============================================================================
# Main
# =============================================================================

async def main():
    """Run all examples."""

    # Check for API key
    if not os.getenv('OPENAI_API_KEY'):
        print("Error: OPENAI_API_KEY environment variable not set")
        print("Please run: export OPENAI_API_KEY=sk-...")
        return

    examples = [
        ("Basic Extended Thinking", basic_thinking),
        ("Runtime Override", runtime_override),
        ("Cost-Optimized", cost_optimized),
        ("Thinking With Tools", thinking_with_tools),
        ("High Confidence Threshold", high_confidence),
        ("Silent Thinking", silent_thinking),
        ("Progress Tracking UI", progress_tracking),
    ]

    print("\n" + "="*60)
    print("Extended Thinking Examples")
    print("="*60)
    print("\nAvailable examples:")
    for i, (name, _) in enumerate(examples, 1):
        print(f"  {i}. {name}")
    print(f"  {len(examples)+1}. Run all")
    print()

    try:
        choice = input("Enter example number (or 'q' to quit): ").strip()

        if choice.lower() == 'q':
            return

        choice_num = int(choice)

        if choice_num == len(examples) + 1:
            # Run all
            for name, func in examples:
                await func()
                print("\n" + "="*60 + "\n")
                input("Press Enter to continue...")
        elif 1 <= choice_num <= len(examples):
            await examples[choice_num - 1][1]()
        else:
            print("Invalid choice")

    except ValueError:
        print("Invalid input")
    except KeyboardInterrupt:
        print("\nInterrupted")


if __name__ == '__main__':
    asyncio.run(main())
