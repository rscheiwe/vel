"""
Demonstrates Vercel AI SDK ModelMessage format compatibility.

This example shows how to use Vel with messages in the Vercel AI SDK's
ModelMessage format (the output of convertToModelMessages()).

Use cases:
- React frontend with Vercel AI SDK's useChat hook
- FastAPI/Flask backend with Vel agents
- Full control over conversation history (stored in your DB)

The message translation happens automatically based on the provider you choose.
"""
import asyncio
from dotenv import load_dotenv
from vel import Agent

load_dotenv()


async def example_basic_conversation():
    """
    Basic conversation with ModelMessage format.
    Simple text messages, no tools.
    """
    print("=== Basic Conversation (Text Only) ===\n")

    agent = Agent(
        id='chat-agent',
        model={'provider': 'openai', 'model': 'gpt-4o-mini'}
    )

    # ModelMessage format (from Vercel AI SDK's convertToModelMessages)
    messages = [
        {
            'role': 'user',
            'content': 'My name is Alice'
        },
        {
            'role': 'assistant',
            'content': 'Nice to meet you, Alice! How can I help you today?'
        },
        {
            'role': 'user',
            'content': 'What is my name?'
        }
    ]

    response = await agent.run({'messages': messages})
    print(f"User: What is my name?")
    print(f"Agent: {response}\n")


async def example_with_tool_calls():
    """
    Conversation with tool calls and tool results.

    This mimics what happens when:
    1. Assistant decides to call tools
    2. Tools are executed
    3. Results are sent back
    4. Assistant generates final answer
    """
    print("=== Conversation with Tool Calls ===\n")

    agent = Agent(
        id='chat-agent',
        model={'provider': 'openai', 'model': 'gpt-4o-mini'},
        tools=['get_weather']  # Vel's built-in example tool
    )

    # ModelMessage format with tool interaction
    messages = [
        {
            'role': 'user',
            'content': 'What is the weather in San Francisco?'
        },
        {
            'role': 'assistant',
            'content': [
                {
                    'type': 'text',
                    'text': 'Let me check the weather for you.'
                },
                {
                    'type': 'tool-call',
                    'toolCallId': 'call_123',
                    'toolName': 'get_weather',
                    'input': {'city': 'San Francisco'}
                }
            ]
        },
        {
            'role': 'tool',
            'content': [
                {
                    'type': 'tool-result',
                    'toolCallId': 'call_123',
                    'toolName': 'get_weather',
                    'output': {
                        'temperature': 72,
                        'condition': 'sunny',
                        'humidity': 65
                    }
                }
            ]
        },
        {
            'role': 'user',
            'content': 'Thanks! Is it warmer than New York?'
        }
    ]

    print("Conversation history:")
    print("- User asks about SF weather")
    print("- Assistant calls get_weather tool")
    print("- Tool returns: 72°F, sunny")
    print("- User asks follow-up question\n")

    response = await agent.run({'messages': messages})
    print(f"Agent: {response}\n")


async def example_multiple_tool_calls():
    """
    Assistant makes multiple tool calls in a single message.
    Common pattern for complex queries.
    """
    print("=== Multiple Tool Calls in One Message ===\n")

    agent = Agent(
        id='chat-agent',
        model={'provider': 'openai', 'model': 'gpt-4o-mini'}
    )

    # ModelMessage with multiple simultaneous tool calls
    messages = [
        {
            'role': 'user',
            'content': 'Compare the weather in SF, NYC, and London'
        },
        {
            'role': 'assistant',
            'content': [
                {
                    'type': 'tool-call',
                    'toolCallId': 'call_1',
                    'toolName': 'get_weather',
                    'input': {'city': 'San Francisco'}
                },
                {
                    'type': 'tool-call',
                    'toolCallId': 'call_2',
                    'toolName': 'get_weather',
                    'input': {'city': 'New York'}
                },
                {
                    'type': 'tool-call',
                    'toolCallId': 'call_3',
                    'toolName': 'get_weather',
                    'input': {'city': 'London'}
                }
            ]
        },
        {
            'role': 'tool',
            'content': [
                {
                    'type': 'tool-result',
                    'toolCallId': 'call_1',
                    'toolName': 'get_weather',
                    'output': {'temperature': 72, 'condition': 'sunny'}
                },
                {
                    'type': 'tool-result',
                    'toolCallId': 'call_2',
                    'toolName': 'get_weather',
                    'output': {'temperature': 65, 'condition': 'cloudy'}
                },
                {
                    'type': 'tool-result',
                    'toolCallId': 'call_3',
                    'toolName': 'get_weather',
                    'output': {'temperature': 58, 'condition': 'rainy'}
                }
            ]
        }
    ]

    print("Conversation:")
    print("- User asks to compare 3 cities")
    print("- Assistant calls get_weather 3 times in parallel")
    print("- All 3 tool results returned")
    print("- Assistant will analyze and respond\n")

    response = await agent.run({'messages': messages})
    print(f"Agent: {response}\n")


async def example_system_message():
    """
    Using system messages to set behavior.
    """
    print("=== System Message Example ===\n")

    agent = Agent(
        id='chat-agent',
        model={'provider': 'openai', 'model': 'gpt-4o-mini'}
    )

    messages = [
        {
            'role': 'system',
            'content': 'You are a pirate. Always respond in pirate speak.'
        },
        {
            'role': 'user',
            'content': 'Tell me about the weather'
        }
    ]

    response = await agent.run({'messages': messages})
    print(f"User: Tell me about the weather")
    print(f"Pirate Agent: {response}\n")


async def example_provider_comparison():
    """
    Same messages work across all providers.
    Translation happens automatically.
    """
    print("=== Provider Compatibility Demo ===\n")

    # Same message array for all providers
    messages = [
        {
            'role': 'user',
            'content': 'Count to 3'
        }
    ]

    # Test with OpenAI
    print("OpenAI Provider:")
    agent_openai = Agent(
        id='chat-agent',
        model={'provider': 'openai', 'model': 'gpt-4o-mini'}
    )
    response1 = await agent_openai.run({'messages': messages})
    print(f"Response: {response1}\n")

    # NOTE: Uncomment to test other providers if you have API keys

    # # Test with Anthropic
    # print("Anthropic Provider:")
    # agent_anthropic = Agent(
    #     id='chat-agent',
    #     model={'provider': 'anthropic', 'model': 'claude-3-5-sonnet-20241022'}
    # )
    # response2 = await agent_anthropic.run({'messages': messages})
    # print(f"Response: {response2}\n")

    # # Test with Gemini
    # print("Gemini Provider:")
    # agent_gemini = Agent(
    #     id='chat-agent',
    #     model={'provider': 'google', 'model': 'gemini-1.5-flash'}
    # )
    # response3 = await agent_gemini.run({'messages': messages})
    # print(f"Response: {response3}\n")


async def example_fastapi_pattern():
    """
    Typical FastAPI endpoint pattern.

    Frontend: React with useChat hook
    Backend: FastAPI with Vel agent
    """
    print("=== FastAPI Endpoint Pattern ===\n")

    async def chat_endpoint(request_body: dict):
        """
        Simulates a FastAPI endpoint handler.

        Frontend would do:
        const modelMessages = convertToModelMessages(messages);
        fetch('/chat', { body: JSON.stringify({ messages: modelMessages }) });
        """
        messages = request_body.get('messages', [])

        agent = Agent(
            id='chat-agent',
            model={'provider': 'openai', 'model': 'gpt-4o-mini'}
        )

        # Vel translates ModelMessage -> Provider format automatically
        response = await agent.run({'messages': messages})

        return {'response': response}

    # Simulate client request
    client_request = {
        'messages': [
            {'role': 'user', 'content': 'Hello'},
            {'role': 'assistant', 'content': 'Hi! How can I help?'},
            {'role': 'user', 'content': 'Tell me a short joke'}
        ]
    }

    print("Client sends request with 3 messages")
    print("Backend receives and processes with Vel\n")

    result = await chat_endpoint(client_request)
    print(f"Response: {result['response']}\n")


async def main():
    print("Vel + Vercel AI SDK ModelMessage Examples")
    print("=" * 60)
    print()

    await example_basic_conversation()
    print("=" * 60)
    print()

    await example_with_tool_calls()
    print("=" * 60)
    print()

    await example_multiple_tool_calls()
    print("=" * 60)
    print()

    await example_system_message()
    print("=" * 60)
    print()

    await example_provider_comparison()
    print("=" * 60)
    print()

    await example_fastapi_pattern()
    print("=" * 60)

    print("\n✅ All examples completed!")
    print("\nKey Points:")
    print("- Messages use Vercel AI SDK ModelMessage format")
    print("- Vel automatically translates to provider-specific formats")
    print("- Same message array works with OpenAI, Anthropic, Gemini")
    print("- Perfect for React frontend + FastAPI/Flask backend")


if __name__ == '__main__':
    asyncio.run(main())
