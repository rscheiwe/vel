"""
Example of using Vel's Translation API

This demonstrates how external libraries can use Vel's event translation
to get standardized stream protocol events from any provider.
"""
import asyncio
import os


async def basic_example():
    """Basic usage of the translation API."""
    from vel import get_translator

    print("=== Basic Translation Example ===\n")

    # Get translator for OpenAI
    translator = get_translator("openai")
    print(f"Using translator for: {translator.provider_name}")

    # Prepare messages
    messages = [
        {"role": "user", "content": "Tell me a very short joke about Python"}
    ]

    # Stream events
    print("\nStreaming response:\n")
    full_text = ""

    async for event in translator.translate_stream(messages, model="gpt-4"):
        if event.type == "text-start":
            print(f"[Text block started: {event.block_id}]")

        elif event.type == "text-delta":
            print(event.delta, end="", flush=True)
            full_text += event.delta

        elif event.type == "text-end":
            print(f"\n[Text block ended: {event.block_id}]")

        elif event.type == "finish-message":
            print(f"\n[Message finished: {event.finish_reason}]")

    print(f"\nFull response: {full_text}")


async def dict_format_example():
    """Example using dictionary format instead of objects."""
    from vel import get_translator

    print("\n\n=== Dictionary Format Example ===\n")

    translator = get_translator("openai")

    messages = [
        {"role": "user", "content": "What is 2+2?"}
    ]

    # Stream as dictionaries (easier for JSON serialization)
    async for event_dict in translator.translate_stream_to_dicts(messages, "gpt-4"):
        print(f"Event: {event_dict}")


async def multi_provider_example():
    """Example showing how to work with multiple providers."""
    from vel import available_providers

    print("\n\n=== Multi-Provider Support ===\n")

    providers = available_providers()
    print(f"Available providers: {providers}\n")

    # You could iterate over all providers
    # for provider_name in providers:
    #     translator = get_translator(provider_name)
    #     # Use same code for all providers


async def tool_calling_example():
    """Example with tool calling."""
    from vel import get_translator

    print("\n\n=== Tool Calling Example ===\n")

    translator = get_translator("openai")

    messages = [
        {"role": "user", "content": "What's the weather in San Francisco?"}
    ]

    tools = {
        "get_weather": {
            "input": {
                "type": "object",
                "properties": {
                    "city": {"type": "string"},
                },
                "required": ["city"]
            }
        }
    }

    async for event in translator.translate_stream(messages, "gpt-4", tools=tools):
        if event.type == "text-delta":
            print(event.delta, end="", flush=True)

        elif event.type == "tool-input-start":
            print(f"\n[Tool call starting: {event.tool_name}]")

        elif event.type == "tool-input-delta":
            print(f"[Tool args delta: {event.input_delta}]", end="")

        elif event.type == "tool-input-available":
            print(f"\n[Tool ready: {event.tool_name}({event.input})]")

        elif event.type == "finish-message":
            print(f"\n[Finished: {event.finish_reason}]")


async def main():
    """Run all examples."""
    # Check if API key is set
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not set")
        print("Set it with: export OPENAI_API_KEY=sk-...")
        return

    # Run examples
    await basic_example()
    await dict_format_example()
    await multi_provider_example()

    # Uncomment to test tool calling
    # await tool_calling_example()


if __name__ == "__main__":
    asyncio.run(main())
