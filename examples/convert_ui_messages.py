"""
Demonstrates convert_to_model_messages() - Python version of Vercel AI SDK's converter.

Use cases:
1. Stored UIMessages from React frontend in database
2. Python-based chat UIs (Streamlit, Gradio)
3. Converting saved conversation history
4. Testing and debugging

The converter handles:
- Text messages
- Images and files
- Tool executions (splits input/output into separate messages)
- UI-only elements (filters them out)
"""
import asyncio
from dotenv import load_dotenv
from vel import Agent
from vel.utils import convert_to_model_messages

load_dotenv()


def example_basic_conversion():
    """
    Basic UIMessage to ModelMessage conversion.
    """
    print("=== Basic Conversion ===\n")

    # UIMessage format (from useChat or stored in DB)
    ui_messages = [
        {
            'id': 'msg-1',
            'role': 'user',
            'parts': [
                {'type': 'text', 'text': 'My name is Alice'}
            ]
        },
        {
            'id': 'msg-2',
            'role': 'assistant',
            'parts': [
                {'type': 'text', 'text': 'Nice to meet you, Alice!'}
            ]
        },
        {
            'id': 'msg-3',
            'role': 'user',
            'parts': [
                {'type': 'text', 'text': 'What is my name?'}
            ]
        }
    ]

    print("UIMessage format (3 messages):")
    for msg in ui_messages:
        role = msg['role']
        text = msg['parts'][0]['text']
        print(f"  {role}: {text}")

    # Convert to ModelMessage format
    model_messages = convert_to_model_messages(ui_messages)

    print("\nModelMessage format (3 messages):")
    for msg in model_messages:
        role = msg['role']
        content = msg['content']
        print(f"  {role}: {content}")

    print("\n✅ Same messages, just format changed\n")


def example_tool_execution_split():
    """
    Tool execution with both input and output splits into 2 messages.
    """
    print("=== Tool Execution Split ===\n")

    # UIMessage with executed tool (input + output in same message)
    ui_messages = [
        {
            'role': 'user',
            'parts': [
                {'type': 'text', 'text': 'What are the latest AI trends?'}
            ]
        },
        {
            'role': 'assistant',
            'parts': [
                {'type': 'step-start'},  # UI-only element
                {'type': 'text', 'text': 'Let me search for that.'},
                {
                    'type': 'tool-websearch',  # Custom tool type
                    'toolCallId': 'call_abc123',
                    'state': 'output-available',  # Both input and output present
                    'input': {'query': 'latest AI trends 2025', 'limit': 3},
                    'output': {
                        'results': [
                            {'title': 'AI Trend 1', 'url': 'https://example.com/1'},
                            {'title': 'AI Trend 2', 'url': 'https://example.com/2'}
                        ]
                    }
                },
                {'type': 'step-finish'}  # UI-only element
            ]
        }
    ]

    print("UIMessage: 2 messages (user + assistant with executed tool)")
    print("  Assistant message has: text + tool-websearch (with input AND output)\n")

    # Convert
    model_messages = convert_to_model_messages(ui_messages)

    print(f"ModelMessage: {len(model_messages)} messages")
    print("  1. user: 'What are the latest AI trends?'")
    print("  2. assistant: 'Let me search...' + tool-call (input only)")
    print("  3. tool: tool-result (output only)")
    print("\n✅ Tool execution split into tool-call and tool-result\n")


def example_multimodal_content():
    """
    Messages with images and files.
    """
    print("=== Multimodal Content ===\n")

    # UIMessage with image
    ui_messages = [
        {
            'role': 'user',
            'parts': [
                {'type': 'text', 'text': 'What is in this image?'},
                {
                    'type': 'image',
                    'image': 'iVBORw0KGgoAAAANS...',  # base64 encoded
                    'mimeType': 'image/png'
                }
            ]
        }
    ]

    model_messages = convert_to_model_messages(ui_messages)

    print("UIMessage: User message with text + image")
    print("ModelMessage: Same structure preserved\n")
    print(f"  role: {model_messages[0]['role']}")
    print(f"  content: [{model_messages[0]['content'][0]['type']}, {model_messages[0]['content'][1]['type']}]")
    print("\n✅ Multimodal content preserved\n")


def example_ui_element_filtering():
    """
    UI-only elements are filtered out.
    """
    print("=== UI Element Filtering ===\n")

    ui_messages = [
        {
            'role': 'assistant',
            'parts': [
                {'type': 'step-start'},  # Filtered out
                {'type': 'text', 'text': 'Processing your request...'},
                {'type': 'step-finish'}  # Filtered out
            ]
        }
    ]

    print("UIMessage: 3 parts (step-start + text + step-finish)")

    model_messages = convert_to_model_messages(ui_messages)

    print(f"ModelMessage: {len(model_messages)} message with only text")
    print(f"  content: '{model_messages[0]['content']}'")
    print("\n✅ UI-only elements filtered\n")


async def example_with_vel_agent():
    """
    Using converted messages with Vel agent.
    """
    print("=== Using with Vel Agent ===\n")

    # Simulate UIMessages retrieved from database
    ui_messages_from_db = [
        {
            'id': 'msg-1',
            'role': 'user',
            'parts': [
                {'type': 'text', 'text': 'Hello'}
            ],
            'createdAt': '2025-01-01T00:00:00Z'
        },
        {
            'id': 'msg-2',
            'role': 'assistant',
            'parts': [
                {'type': 'text', 'text': 'Hi! How can I help?'}
            ],
            'createdAt': '2025-01-01T00:00:01Z'
        },
        {
            'id': 'msg-3',
            'role': 'user',
            'parts': [
                {'type': 'text', 'text': 'Tell me a joke'}
            ],
            'createdAt': '2025-01-01T00:00:02Z'
        }
    ]

    print("Scenario: Retrieved 3 UIMessages from database")
    print("Converting to ModelMessage format...\n")

    # Convert
    model_messages = convert_to_model_messages(ui_messages_from_db)

    # Use with Vel agent
    agent = Agent(
        id='chat-agent',
        model={'provider': 'openai', 'model': 'gpt-4o-mini'}
    )

    response = await agent.run({'messages': model_messages})

    print(f"Agent response: {response}\n")
    print("✅ Converted UIMessages work perfectly with Vel\n")


def example_multiple_tool_calls():
    """
    Multiple tool calls in single message.
    """
    print("=== Multiple Tool Calls ===\n")

    ui_messages = [
        {
            'role': 'assistant',
            'parts': [
                {
                    'type': 'tool-websearch',
                    'toolCallId': 'call_1',
                    'state': 'output-available',
                    'input': {'query': 'AI trends'},
                    'output': {'results': ['trend1']}
                },
                {
                    'type': 'tool-news',
                    'toolCallId': 'call_2',
                    'state': 'output-available',
                    'input': {'topic': 'AI'},
                    'output': {'articles': ['article1']}
                }
            ]
        }
    ]

    print("UIMessage: 1 message with 2 tool executions")

    model_messages = convert_to_model_messages(ui_messages)

    print(f"ModelMessage: {len(model_messages)} messages")
    print("  1. assistant: [tool-call-1, tool-call-2]")
    print("  2. tool: [tool-result-1, tool-result-2]")
    print("\n✅ Multiple tools handled correctly\n")


def example_practical_fastapi_pattern():
    """
    Practical FastAPI pattern with UIMessages from database.
    """
    print("=== Practical FastAPI Pattern ===\n")

    # Simulate FastAPI endpoint
    async def chat_endpoint(user_id: str, new_message: str, db_connection):
        """
        FastAPI endpoint that:
        1. Loads UIMessages from database
        2. Converts to ModelMessages
        3. Sends to Vel agent
        4. Returns response
        """
        # 1. Load conversation history from DB (UIMessage format)
        ui_messages_from_db = db_connection.get_messages(user_id)

        # 2. Add new user message
        ui_messages_from_db.append({
            'role': 'user',
            'parts': [{'type': 'text', 'text': new_message}]
        })

        # 3. Convert to ModelMessage format
        model_messages = convert_to_model_messages(ui_messages_from_db)

        # 4. Send to Vel agent
        agent = Agent(
            id='chat-agent',
            model={'provider': 'openai', 'model': 'gpt-4o-mini'}
        )

        response = await agent.run({'messages': model_messages})

        # 5. Save assistant response back to DB (UIMessage format)
        db_connection.save_message(user_id, {
            'role': 'assistant',
            'parts': [{'type': 'text', 'text': response}]
        })

        return {'response': response}

    # Mock database
    class MockDB:
        def get_messages(self, user_id):
            return [
                {'role': 'user', 'parts': [{'type': 'text', 'text': 'Hello'}]},
                {'role': 'assistant', 'parts': [{'type': 'text', 'text': 'Hi!'}]}
            ]

        def save_message(self, user_id, message):
            pass

    print("FastAPI endpoint workflow:")
    print("  1. Load UIMessages from DB")
    print("  2. Add new user message")
    print("  3. Convert to ModelMessages")
    print("  4. Send to Vel agent")
    print("  5. Save response to DB")
    print("\n✅ Clean separation: UI format in DB, Model format for LLM\n")


async def main():
    print("Python convert_to_model_messages() Examples")
    print("=" * 60)
    print()

    example_basic_conversion()
    print("=" * 60)
    print()

    example_tool_execution_split()
    print("=" * 60)
    print()

    example_multimodal_content()
    print("=" * 60)
    print()

    example_ui_element_filtering()
    print("=" * 60)
    print()

    await example_with_vel_agent()
    print("=" * 60)
    print()

    example_multiple_tool_calls()
    print("=" * 60)
    print()

    example_practical_fastapi_pattern()
    print("=" * 60)

    print("\n✅ All examples completed!")
    print("\nKey Takeaways:")
    print("- convert_to_model_messages() is Python equivalent of Vercel AI SDK's converter")
    print("- Splits tool executions into separate call/result messages")
    print("- Filters UI-only elements (step-start, step-finish)")
    print("- Preserves multimodal content (images, files)")
    print("- Perfect for apps that store UIMessages in DB")


if __name__ == '__main__':
    asyncio.run(main())
