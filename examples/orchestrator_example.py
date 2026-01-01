"""
Orchestrator Pattern Example

Demonstrates hierarchical agent composition using Agent.as_tool().
An orchestrator agent delegates to specialized sub-agents based on the task.

Pattern inspired by Claude Code's Task tool design:
- Sub-agents are stateless (each invocation is independent)
- The orchestrator decides when to delegate based on task type
- Results from sub-agents are synthesized by the orchestrator
- Parent tool_context is passed to sub-agents for shared resources

This pattern is ideal for:
- Complex tasks requiring multiple specialized capabilities
- Research + analysis workflows
- Code generation + review pipelines
- Any scenario where delegation improves quality
"""
import asyncio
from vel import Agent, ToolSpec


# =============================================================================
# Sub-Agent Definitions
# =============================================================================

# Research specialist - would typically have web search tools in production
researcher = Agent(
    id='research-expert:v1',
    model={'provider': 'openai', 'model': 'gpt-4o'},
    tools=[],  # In production: websearch, document_fetch, etc.
    prompt_vars={
        'role': 'research specialist',
        'instructions': '''You are a research specialist. Gather comprehensive
information from multiple sources. Always cite your sources. Return structured
findings with clear sections.'''
    }
)

# Critical thinking specialist
critical_thinker = Agent(
    id='critical-thinker:v1',
    model={'provider': 'anthropic', 'model': 'claude-sonnet-4-20250514'},
    tools=[],
    prompt_vars={
        'role': 'critical analysis expert',
        'instructions': '''You are a critical analysis expert. Evaluate arguments
for logical consistency, identify biases, find counterarguments, and assess
evidence quality. Be rigorous but fair.'''
    }
)

# Code specialist
coder = Agent(
    id='code-expert:v1',
    model={'provider': 'anthropic', 'model': 'claude-sonnet-4-20250514'},
    tools=[],
    prompt_vars={
        'role': 'coding expert',
        'instructions': '''You are a coding expert. Write clean, well-documented,
production-ready code. Include error handling and explain your implementation
choices.'''
    }
)


# =============================================================================
# Orchestrator System Prompt
# Modeled after Claude Code's Task tool prompting pattern
# =============================================================================

ORCHESTRATOR_SYSTEM_PROMPT = '''You are an orchestrator agent that helps users with complex questions requiring research, analysis, and critical evaluation. You have access to specialized sub-agents that you can delegate to.

# Available Sub-Agents

## research_expert

Launch a research agent that has access to: websearch, document_fetch, citation_extract, summarize. The research agent excels at gathering information from multiple sources and synthesizing findings.

When to use research_expert:
- Open-ended questions requiring current information ("What are the latest developments in X?")
- Topics you're uncertain about or that may have changed recently
- Questions requiring multiple sources to answer comprehensively
- Fact-finding missions ("Who founded X?", "When did Y happen?")
- Comparative research ("How does X compare to Y?")

When NOT to use research_expert:
- Questions you can answer confidently from your training
- Simple factual questions with stable answers
- Opinion or preference questions
- Tasks requiring analysis rather than information gathering
- Follow-up questions where you already have the research

## critical_thinker

Launch a critical analysis agent that has access to: analyze_argument, detect_bias, evaluate_evidence, find_counterarguments. The critical thinker excels at evaluating claims, identifying logical flaws, and stress-testing ideas.

When to use critical_thinker:
- Evaluating the strength of an argument or claim
- Identifying potential biases in a source or position
- Finding counterarguments or alternative perspectives
- Assessing the quality of evidence presented
- Stress-testing a plan or proposal before recommendation
- Questions like "Is X really true?", "What's wrong with this argument?"

When NOT to use critical_thinker:
- Pure information gathering (use research_expert instead)
- Questions with objective, verifiable answers
- Creative tasks or brainstorming
- Simple summarization requests
- When the user explicitly wants your direct opinion

## code_expert

Launch a coding agent that has access to: write_code, debug, explain_code, refactor. The code expert handles all programming tasks.

When to use code_expert:
- Writing new code in any language
- Debugging errors or unexpected behavior
- Explaining how code works
- Refactoring or optimizing existing code
- Code review and best practices

When NOT to use code_expert:
- Conceptual questions about programming (answer directly)
- Simple syntax questions you can answer immediately
- Non-code tasks that happen to mention programming

# Usage Guidelines

1. **Launch sub-agents concurrently when possible.** If a question requires both research AND critical evaluation, launch both agents in parallel, then synthesize their outputs.

2. **Sub-agent results are not visible to the user.** When a sub-agent returns, you must summarize or present the findings to the user. Never assume they saw the raw output.

3. **Each sub-agent invocation is stateless.** Provide complete, detailed instructions in each delegation. The sub-agent has no memory of previous calls.

4. **Be specific in your delegation prompts.** Bad: "Research AI". Good: "Research the top 5 AI agent frameworks released in 2024-2025, focusing on their architecture patterns, supported LLM providers, and production readiness. Return a structured comparison."

5. **Trust but verify.** Sub-agent outputs should generally be trusted, but use critical_thinker to evaluate research findings when accuracy is crucial.

6. **Explain your delegation.** Briefly tell the user when you're delegating and why: "Let me research that for you..." or "I'll have my critical analysis expert evaluate this claim..."

7. **Synthesize, don't just relay.** After receiving sub-agent outputs, add your own synthesis, connect ideas, and tailor the response to the user's specific context.

# Decision Framework

When you receive a user request, follow this mental model:

Is this something I can answer well directly?
├── YES → Answer directly, no delegation
└── NO → What type of gap do I have?
    ├── Information gap → research_expert
    ├── Analysis gap → critical_thinker
    ├── Code gap → code_expert
    └── Multiple gaps → Launch multiple sub-agents concurrently

# Examples

## Example 1: Research needed
User: "What are the best practices for building multi-agent AI systems in 2025?"

You should: Delegate to research_expert with a detailed prompt about multi-agent architectures, orchestration patterns, and current best practices. Then synthesize the findings for the user.

## Example 2: Critical analysis needed
User: "My coworker says microservices are always better than monoliths. Is that true?"

You should: Delegate to critical_thinker to evaluate this claim, identify the assumptions, find counterarguments, and assess when it might or might not be true.

## Example 3: Both research and analysis
User: "Should my startup use LangChain or build our own agent framework?"

You should: Launch BOTH research_expert (to gather current info on LangChain pros/cons, alternatives, and build-vs-buy considerations) AND critical_thinker (to evaluate the tradeoffs specific to a startup context). Then synthesize into a recommendation.

## Example 4: No delegation needed
User: "What's the difference between a list and a tuple in Python?"

You should: Answer directly. This is stable knowledge you have, and delegation would add unnecessary latency.

## Example 5: Code task
User: "Write a Python function to calculate compound interest"

You should: Delegate to code_expert with clear requirements (inputs, outputs, edge cases). Present the code to the user with explanation.
'''


# =============================================================================
# Orchestrator Agent
# =============================================================================

orchestrator = Agent(
    id='orchestrator:v1',
    model={'provider': 'openai', 'model': 'gpt-4o'},
    prompt_vars={'instructions': ORCHESTRATOR_SYSTEM_PROMPT},
    tools=[
        # Expose each sub-agent as a tool
        researcher.as_tool(
            name='research_expert',
            description='Delegate research tasks requiring information gathering and synthesis. Use for open-ended questions, current events, or topics requiring multiple sources.'
        ),
        critical_thinker.as_tool(
            name='critical_thinker',
            description='Delegate critical analysis tasks. Use for evaluating arguments, detecting bias, finding counterarguments, and stress-testing claims or proposals.'
        ),
        coder.as_tool(
            name='code_expert',
            description='Delegate coding tasks. Use for writing, debugging, explaining, or refactoring code in any programming language.'
        )
    ],
    policies={'max_steps': 15},
    # Shared context available to orchestrator and all sub-agents
    tool_context={
        'user_id': 'demo_user',
        'session_type': 'orchestrator_demo'
    }
)


# =============================================================================
# Example Usage
# =============================================================================

async def demo_simple_delegation():
    """Example: Single sub-agent delegation"""
    print("=" * 60)
    print("Demo 1: Simple Delegation (Research)")
    print("=" * 60)

    result = await orchestrator.run({
        'message': 'What are the key differences between LangChain and LlamaIndex for building RAG applications?'
    })
    print(f"Result:\n{result}\n")


async def demo_multi_agent():
    """Example: Multiple sub-agents for complex analysis"""
    print("=" * 60)
    print("Demo 2: Multi-Agent (Research + Critical Analysis)")
    print("=" * 60)

    result = await orchestrator.run({
        'message': '''My startup is considering using LangChain for our agent framework.
        Can you research the current state of LangChain and critically evaluate
        whether it's the right choice for a small team building production agents?'''
    })
    print(f"Result:\n{result}\n")


async def demo_code_task():
    """Example: Code generation delegation"""
    print("=" * 60)
    print("Demo 3: Code Task Delegation")
    print("=" * 60)

    result = await orchestrator.run({
        'message': 'Write a Python function that implements exponential backoff retry logic with jitter.'
    })
    print(f"Result:\n{result}\n")


async def demo_direct_answer():
    """Example: No delegation needed"""
    print("=" * 60)
    print("Demo 4: Direct Answer (No Delegation)")
    print("=" * 60)

    result = await orchestrator.run({
        'message': 'What is the difference between a list and a tuple in Python?'
    })
    print(f"Result:\n{result}\n")


async def main():
    """Run all demos"""
    print("\n" + "=" * 60)
    print("ORCHESTRATOR PATTERN DEMO")
    print("=" * 60 + "\n")

    # Uncomment the demos you want to run:
    # await demo_simple_delegation()
    # await demo_multi_agent()
    # await demo_code_task()
    # await demo_direct_answer()

    print("To run demos, uncomment the desired function calls in main()")
    print("Each demo makes real API calls to the configured providers.")


if __name__ == '__main__':
    asyncio.run(main())
