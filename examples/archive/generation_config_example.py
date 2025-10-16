"""
Examples demonstrating generation_config usage across all providers.

The generation_config parameter allows fine-grained control over model behavior,
matching the flexibility of Vercel AI SDK's streamText() function.
"""

import asyncio
from vel import Agent

async def deterministic_code_generation():
    """Example: Deterministic code generation with seed and low temperature"""
    agent = Agent(
        id='code-generator',
        model={'provider': 'openai', 'model': 'gpt-4o'},
        generation_config={
            'temperature': 0,  # Fully deterministic
            'max_tokens': 2000,  # Enough for code
            'seed': 42,  # Reproducible outputs
            'stop': ['```\n\n']  # Stop after code block
        }
    )

    # This should generate identical code every time due to seed
    result1 = await agent.run({'message': 'Write a Python function to calculate factorial'})
    print("First run:", result1[:100], "...")

    result2 = await agent.run({'message': 'Write a Python function to calculate factorial'})
    print("Second run:", result2[:100], "...")
    print("Results identical:", result1 == result2)


async def creative_storytelling():
    """Example: Creative storytelling with high temperature"""
    agent = Agent(
        id='storyteller',
        model={'provider': 'anthropic', 'model': 'claude-sonnet-4-20250514'},
        generation_config={
            'temperature': 0.9,  # Very creative
            'max_tokens': 4000,  # Long stories
            'top_p': 0.95,  # Diverse vocabulary
            'top_k': 50  # Consider top 50 tokens
        }
    )

    result = await agent.run({'message': 'Tell me a sci-fi story about AI'})
    print("Story:", result[:200], "...")


async def concise_answers():
    """Example: Limit output length for concise answers"""
    agent = Agent(
        id='concise-assistant',
        model={'provider': 'google', 'model': 'gemini-1.5-pro'},
        generation_config={
            'max_tokens': 100,  # Short responses
            'temperature': 0.7,
            'stop_sequences': ['\n\n']  # Stop at double newline
        }
    )

    result = await agent.run({'message': 'Explain quantum computing'})
    print("Concise answer:", result)


async def per_run_override():
    """Example: Override generation config per run"""
    agent = Agent(
        id='flexible-agent',
        model={'provider': 'openai', 'model': 'gpt-4o'},
        generation_config={
            'temperature': 0.7,  # Default temperature
            'max_tokens': 500
        }
    )

    # Use default config
    creative_result = await agent.run({'message': 'Write a poem'})
    print("Creative (0.7):", creative_result[:100], "...")

    # Override for this run only - deterministic
    factual_result = await agent.run(
        {'message': 'What is 2+2?'},
        generation_config={'temperature': 0}  # Override to 0 for this run
    )
    print("Factual (0.0):", factual_result)


async def streaming_with_config():
    """Example: Streaming with generation config"""
    agent = Agent(
        id='streaming-agent',
        model={'provider': 'anthropic', 'model': 'claude-sonnet-4-20250514'},
        generation_config={
            'temperature': 0.8,
            'max_tokens': 1000
        }
    )

    print("Streaming response:")
    async for event in agent.run_stream({'message': 'Explain neural networks'}):
        if event.get('type') == 'text-delta':
            print(event.get('delta'), end='', flush=True)
    print()


async def provider_specific_parameters():
    """Example: Using provider-specific parameters"""

    # OpenAI-specific: presence_penalty and frequency_penalty
    openai_agent = Agent(
        id='openai-agent',
        model={'provider': 'openai', 'model': 'gpt-4o'},
        generation_config={
            'temperature': 0.7,
            'presence_penalty': 0.6,  # Encourage new topics
            'frequency_penalty': 0.3,  # Reduce repetition
            'logit_bias': {  # Discourage certain tokens (example)
                # Token IDs would go here
            }
        }
    )

    # Anthropic-specific: top_k
    anthropic_agent = Agent(
        id='anthropic-agent',
        model={'provider': 'anthropic', 'model': 'claude-sonnet-4-20250514'},
        generation_config={
            'temperature': 0.8,
            'top_k': 40,  # Anthropic supports top_k
            'top_p': 0.9
        }
    )

    # Gemini-specific: max_output_tokens (native parameter name)
    gemini_agent = Agent(
        id='gemini-agent',
        model={'provider': 'google', 'model': 'gemini-1.5-pro'},
        generation_config={
            'temperature': 0.7,
            'max_output_tokens': 2048,  # Gemini's native parameter name
            'top_k': 40,  # Gemini also supports top_k
            'top_p': 0.95
        }
    )


async def main():
    """Run all examples"""
    print("=== Deterministic Code Generation ===")
    await deterministic_code_generation()

    print("\n=== Creative Storytelling ===")
    await creative_storytelling()

    print("\n=== Concise Answers ===")
    await concise_answers()

    print("\n=== Per-Run Override ===")
    await per_run_override()

    print("\n=== Streaming with Config ===")
    await streaming_with_config()


if __name__ == '__main__':
    asyncio.run(main())
