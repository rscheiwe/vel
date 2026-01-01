#!/usr/bin/env python3
"""
Scratchpad Demo — Ephemeral Working Memory for Multi-Step Tasks

This example demonstrates:
1. Enabling scratchpad with a single flag
2. Agent using scratchpad tools during execution
3. Automatic summary injection between runs
4. Clearing context for new conversations

The Scratchpad is ephemeral in-memory storage for a single agent run:
- Execution plans
- Research findings
- Observations from tool calls
- Reasoning steps

Unlike Fact Store (long-term) or ReasoningBank (learned strategies), the
Scratchpad is for short-term working memory during multi-step tool execution.
"""

import asyncio
import os

from vel import Agent
from vel.tools import ToolSpec
from vel.tools.scratchpad import ScratchpadConfig


def print_section(title: str):
    """Print a formatted section header."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")


def print_subsection(title: str):
    """Print a formatted subsection header."""
    print(f"\n--- {title} ---\n")


# =============================================================================
# Mock Tools for Demo
# =============================================================================

def search_documentation(query: str) -> str:
    """Search technical documentation for information."""
    # Simulated search results
    results = {
        "quantum computing": "Quantum computing uses qubits. Key players: IBM (127 qubits), Google (72 qubits), IonQ (32 qubits).",
        "ibm quantum": "IBM Quantum: 127-qubit Eagle processor, Qiskit framework, cloud access via IBM Cloud.",
        "google quantum": "Google Quantum AI: Sycamore processor, achieved quantum supremacy in 2019.",
        "ionq": "IonQ: Trapped ion technology, 32 algorithmic qubits, publicly traded (NYSE: IONQ).",
    }

    for key, value in results.items():
        if key in query.lower():
            return value
    return f"No results found for: {query}"


def get_company_funding(company: str) -> str:
    """Get funding information for a company."""
    funding = {
        "ibm": "IBM: Fortune 500 company, $60B+ revenue, significant R&D investment in quantum.",
        "google": "Google/Alphabet: $280B+ revenue, quantum division well-funded internally.",
        "ionq": "IonQ: Raised $600M+, valued at $2B, backed by Amazon, Samsung, Breakthrough Energy.",
    }

    for key, value in funding.items():
        if key in company.lower():
            return value
    return f"No funding data for: {company}"


# =============================================================================
# Demo Functions
# =============================================================================

async def demo_basic_scratchpad():
    """Demo 1: Basic scratchpad usage with scratchpad=True."""
    print_section("Demo 1: Basic Scratchpad Usage")

    # Create agent with scratchpad enabled (simplest form)
    agent = Agent(
        id="researcher",
        model=get_model_config(),
        tools=[
            ToolSpec.from_function(search_documentation),
            ToolSpec.from_function(get_company_funding),
        ],
        scratchpad=True,  # Enable scratchpad with defaults
    )

    print("Agent created with scratchpad=True")
    print("Available scratchpad tools: write_to_scratchpad, read_from_scratchpad,")
    print("  save_plan, record_finding, record_observation, search_scratchpad,")
    print("  checkpoint_scratchpad, list_scratchpad_checkpoints")

    # Run the agent
    print_subsection("Running Agent")

    result = await agent.run({
        "message": """Research the top 3 quantum computing companies.

        Use your scratchpad to:
        1. Save your execution plan first
        2. Record each finding as you discover it
        3. Read your scratchpad before giving your final answer
        """
    })

    print("Agent response:")
    print(result[:500] + "..." if len(result) > 500 else result)

    # Check that summary was captured
    print_subsection("Captured Summary")
    if agent._scratchpad_summary:
        print("Summary from this run (will be injected into next run):")
        print(agent._scratchpad_summary)
    else:
        print("(No summary captured - agent may not have used scratchpad)")


async def demo_multi_run_continuity():
    """Demo 2: Context continuity across multiple runs."""
    print_section("Demo 2: Multi-Run Context Continuity")

    agent = Agent(
        id="research-assistant",
        model=get_model_config(),
        tools=[
            ToolSpec.from_function(search_documentation),
            ToolSpec.from_function(get_company_funding),
        ],
        scratchpad=ScratchpadConfig(
            max_entries=50,
            summary_max_chars=800,  # Larger summary for more context
        ),
    )

    # Run 1: Initial research
    print_subsection("Run 1: Initial Research")
    result1 = await agent.run({
        "message": "Research IBM's quantum computing efforts. Save your plan and record findings."
    })
    print(f"Run 1 response: {result1[:300]}...")
    print(f"\nSummary captured: {bool(agent._scratchpad_summary)}")

    # Run 2: Follow-up (previous context injected automatically)
    print_subsection("Run 2: Follow-up Question")
    result2 = await agent.run({
        "message": "Now compare IBM's approach to Google's quantum computing. Build on your previous research."
    })
    print(f"Run 2 response: {result2[:300]}...")

    # Run 3: Another follow-up
    print_subsection("Run 3: Investment Analysis")
    result3 = await agent.run({
        "message": "Based on your research, which company seems better positioned? Why?"
    })
    print(f"Run 3 response: {result3[:300]}...")

    # Clear context and start fresh
    print_subsection("Clearing Context")
    agent.clear_scratchpad_context()
    print("Scratchpad context cleared")
    print(f"Summary after clear: {agent._scratchpad_summary}")


async def demo_scratchpad_config():
    """Demo 3: Custom scratchpad configuration."""
    print_section("Demo 3: Custom Configuration")

    # Create agent with custom scratchpad config
    config = ScratchpadConfig(
        max_entries=25,           # Smaller scratchpad
        max_content_length=10000,  # Shorter entries
        summary_max_chars=300,     # Compact summaries
        include_search=True,       # Include search tool
        include_checkpoint=False,  # Disable checkpoint tools (fewer tools)
    )

    print(f"Custom config:")
    print(f"  max_entries: {config.max_entries}")
    print(f"  max_content_length: {config.max_content_length}")
    print(f"  summary_max_chars: {config.summary_max_chars}")
    print(f"  include_search: {config.include_search}")
    print(f"  include_checkpoint: {config.include_checkpoint}")

    agent = Agent(
        id="compact-agent",
        model=get_model_config(),
        scratchpad=config,
    )

    # Show injected tools
    print_subsection("Scratchpad Tools (via config)")
    from vel.tools.scratchpad import Scratchpad, get_scratchpad_tools
    sp = Scratchpad(config)
    tools = get_scratchpad_tools(sp)
    print(f"Number of tools: {len(tools)}")
    for tool in tools:
        print(f"  - {tool.name}: {tool.description[:50]}...")


async def demo_scratchpad_standalone():
    """Demo 4: Using Scratchpad class directly (without Agent)."""
    print_section("Demo 4: Standalone Scratchpad Usage")

    from vel.tools.scratchpad import Scratchpad, EntryType

    # Create scratchpad directly
    sp = Scratchpad()

    # Write operations
    print_subsection("Writing to Scratchpad")

    sp.set_plan("1. Research competitors\n2. Analyze market\n3. Make recommendation")
    print("Set execution plan")

    sp.add_finding("Market is growing 25% YoY", source="industry_report")
    sp.add_finding("Main competitor has 40% market share", source="market_analysis")
    sp.add_finding("New regulations expected Q3", source="legal_team")
    print("Added 3 findings")

    sp.add_observation("API returned 150 results", tool_name="search_api")
    print("Added observation")

    sp.add_reasoning("Given the growth rate and regulatory changes, timing is important")
    sp.add_reasoning("Competitor's market share suggests room for disruption")
    print("Added reasoning steps")

    # Read all
    print_subsection("Reading All Contents")
    print(sp.read())

    # Search
    print_subsection("Searching for 'market'")
    results = sp.search("market")
    print(f"Found {len(results)} entries matching 'market'")
    for r in results:
        print(f"  - [{r.entry_type.value}] {r.content[:50]}...")

    # Summary
    print_subsection("Generated Summary")
    print(sp.get_summary())

    # Stats
    print_subsection("Statistics")
    stats = sp.get_stats()
    print(f"Total entries: {stats.total_entries}")
    print(f"By type: {stats.entries_by_type}")
    print(f"Total content length: {stats.total_content_length} chars")


def get_model_config():
    """Get model configuration based on available API keys."""
    if os.environ.get("ANTHROPIC_API_KEY"):
        return {"provider": "anthropic", "model": "claude-sonnet-4"}
    elif os.environ.get("OPENAI_API_KEY"):
        return {"provider": "openai", "model": "gpt-4o"}
    else:
        # Return OpenAI config but warn user
        print("Warning: No API key found. Set OPENAI_API_KEY or ANTHROPIC_API_KEY")
        return {"provider": "openai", "model": "gpt-4o"}


async def main():
    print_section("Scratchpad Demo")
    print("""
The Scratchpad is ephemeral working memory for multi-step agent execution.

Key features:
- Lives for a single agent run (not persisted)
- Automatic summary injection between runs
- Structured entry types (plan, finding, observation, reasoning, error)
- Thread-safe for concurrent tool calls
- Zero external dependencies

Use cases:
- Multi-step research tasks
- Complex reasoning chains
- Tool output aggregation
- Execution plan tracking
""")

    # Check for API key
    if not os.environ.get("ANTHROPIC_API_KEY") and not os.environ.get("OPENAI_API_KEY"):
        print("="*60)
        print("NOTE: No API key found. Demos 1-3 require an API key.")
        print("Set OPENAI_API_KEY or ANTHROPIC_API_KEY to run full demos.")
        print("Running Demo 4 (standalone usage) only...")
        print("="*60)
        await demo_scratchpad_standalone()
        return

    # Run all demos
    await demo_basic_scratchpad()
    await demo_multi_run_continuity()
    await demo_scratchpad_config()
    await demo_scratchpad_standalone()

    print_section("Demo Complete")
    print("""
Summary:
1. Enable with scratchpad=True or ScratchpadConfig(...)
2. Agent automatically gets scratchpad tools
3. Summary from each run is injected into the next
4. Call agent.clear_scratchpad_context() to start fresh
""")


if __name__ == "__main__":
    asyncio.run(main())
