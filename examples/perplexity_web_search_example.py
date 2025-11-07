"""
Example: Using Perplexity Web Search Tool

This example demonstrates how to use the web_search tool with Perplexity API
in a Vel agent.

Setup:
    1. Install Perplexity API client (uses OpenAI-compatible API):
       pip install openai

    2. Get your API key from: https://www.perplexity.ai/settings/api

    3. Set environment variable:
       export PERPLEXITY_API_KEY=pplx-...

Usage:
    python examples/perplexity_web_search_example.py
"""
import os
import asyncio
from vel import Agent

# Import the web_search tool (registers it automatically)
from examples.multi_step_tools.web_search import web_search_tool


async def main():
    # Check API key
    if not os.getenv('PERPLEXITY_API_KEY'):
        print("❌ Error: PERPLEXITY_API_KEY environment variable not set")
        print("Get your API key from: https://www.perplexity.ai/settings/api")
        print("Then run: export PERPLEXITY_API_KEY=pplx-...")
        return

    # Create agent with web search capability
    agent = Agent(
        id='research-agent:v1',
        model={
            'provider': 'openai',
            'model': 'gpt-4o'
        },
        tools=['websearch'],  # Use the Perplexity web search tool
        policies={'max_steps': 10}
    )

    print("🔍 Research Agent with Perplexity Web Search\n")

    # Example queries
    queries = [
        "What are the latest developments in AI agents?",
        "Compare Perplexity vs OpenAI search capabilities"
    ]

    for query in queries:
        print(f"Query: {query}")
        print("-" * 60)

        result = await agent.run({
            'message': query
        })

        print(f"Answer: {result}\n")


if __name__ == '__main__':
    asyncio.run(main())
