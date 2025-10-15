"""
Message Reducer Example

Demonstrates how to use MessageReducer to aggregate streaming events
into Vercel AI SDK compatible message format for storage and frontend use.

This example shows:
1. Creating a MessageReducer instance
2. Adding user messages
3. Processing streaming events from an agent
4. Getting the complete message structure
5. Storing messages in a database (simulated)

Run: python examples/message_reducer_example.py
"""
import asyncio
import json
from dotenv import load_dotenv
from vel import Agent, MessageReducer

# Load environment variables
load_dotenv()


async def basic_example():
    """Basic example: Simple text response"""
    print("=" * 70)
    print("Basic Example: Simple Text Response")
    print("=" * 70)

    # Create reducer
    reducer = MessageReducer()

    # Add user message
    user_msg = reducer.add_user_message("Hello! How are you?")
    print("\n✓ User message created:")
    print(json.dumps(user_msg, indent=2))

    # Create agent
    agent = Agent(
        id='chat-agent',
        model={'provider': 'openai', 'model': 'gpt-4o'}
    )

    # Stream response and aggregate
    print("\n✓ Streaming assistant response...")
    async for event in agent.run_stream({'message': 'Hello! How are you?'}):
        reducer.process_event(event)

    # Get complete messages
    messages = reducer.get_messages()
    print("\n✓ Complete message structure:")
    print(json.dumps(messages, indent=2))

    print("\n" + "=" * 70)


async def tool_calling_example():
    """Example with tool calls"""
    print("\n" + "=" * 70)
    print("Tool Calling Example: Weather Query")
    print("=" * 70)

    # Create reducer
    reducer = MessageReducer()

    # Add user message
    user_msg = reducer.add_user_message("What's the weather in San Francisco?")
    print("\n✓ User message created:")
    print(json.dumps(user_msg, indent=2))

    # Note: The Agent uses a default ToolRegistry with a built-in get_weather tool
    # that returns {'temp_f': 72.0}. For custom tools, you would register them
    # before creating the Agent, or provide a custom ToolRegistry instance.

    # Create agent with tools
    agent = Agent(
        id='weather-agent',
        model={'provider': 'openai', 'model': 'gpt-4o'},
        tools=['get_weather']  # Uses built-in get_weather tool
    )

    # Stream response and aggregate
    print("\n✓ Streaming assistant response with tool calls...")
    async for event in agent.run_stream({'message': "What's the weather in San Francisco?"}):
        reducer.process_event(event)
        # Show event types as they come in
        event_type = event.get('type')
        if event_type in ['step-start', 'tool-input-available', 'tool-output-available', 'text-start']:
            print(f"   → Event: {event_type}")

    # Get complete messages
    messages = reducer.get_messages()
    print("\n✓ Complete message structure:")
    print(json.dumps(messages, indent=2))

    # Show parts breakdown
    assistant_msg = messages[1]
    print("\n✓ Parts breakdown:")
    for i, part in enumerate(assistant_msg['parts'], 1):
        print(f"   Part {i}: {part['type']}")

    print("\n" + "=" * 70)


async def multi_turn_example():
    """Example: Multi-turn conversation with reducer"""
    print("\n" + "=" * 70)
    print("Multi-Turn Example: Conversation with Memory")
    print("=" * 70)

    # Create agent
    agent = Agent(
        id='chat-agent',
        model={'provider': 'openai', 'model': 'gpt-4o'},
        session_persistence='transient'
    )

    session_id = 'user-123'
    all_messages = []

    # Turn 1
    print("\n--- Turn 1 ---")
    reducer1 = MessageReducer()
    user_msg1 = reducer1.add_user_message("My name is Alice")
    print(f"User: {user_msg1['parts'][0]['text']}")

    async for event in agent.run_stream({'message': 'My name is Alice'}, session_id=session_id):
        reducer1.process_event(event)

    messages1 = reducer1.get_messages()
    all_messages.extend(messages1)

    # Get assistant response text
    assistant_text1 = next(
        (part['text'] for part in messages1[1]['parts'] if part['type'] == 'text'),
        'No text response'
    )
    print(f"Assistant: {assistant_text1}")

    # Turn 2
    print("\n--- Turn 2 ---")
    reducer2 = MessageReducer()
    user_msg2 = reducer2.add_user_message("What is my name?")
    print(f"User: {user_msg2['parts'][0]['text']}")

    async for event in agent.run_stream({'message': 'What is my name?'}, session_id=session_id):
        reducer2.process_event(event)

    messages2 = reducer2.get_messages()
    all_messages.extend(messages2)

    # Get assistant response text
    assistant_text2 = next(
        (part['text'] for part in messages2[1]['parts'] if part['type'] == 'text'),
        'No text response'
    )
    print(f"Assistant: {assistant_text2}")

    # Show complete conversation
    print("\n✓ Complete conversation structure (all 4 messages):")
    print(json.dumps(all_messages, indent=2))

    print("\n" + "=" * 70)


async def database_storage_example():
    """Example: Storing messages in a database (simulated)"""
    print("\n" + "=" * 70)
    print("Database Storage Example")
    print("=" * 70)

    # Simulated database
    database = []

    def store_messages(messages, conversation_id):
        """Simulate storing messages in a database"""
        for msg in messages:
            db_record = {
                'conversation_id': conversation_id,
                'message_id': msg['id'],
                'role': msg['role'],
                'parts': msg['parts'],
                'metadata': msg['metadata'],
                'created_at': 'timestamp_here'
            }
            database.append(db_record)

    # Create reducer and agent
    reducer = MessageReducer()
    agent = Agent(
        id='chat-agent',
        model={'provider': 'openai', 'model': 'gpt-4o'}
    )

    conversation_id = 'conv-123'

    # Add user message with metadata
    user_msg = reducer.add_user_message(
        "Tell me a short joke",
        metadata={'source': 'web-app', 'user_id': 'user-456'}
    )

    # Stream and aggregate
    async for event in agent.run_stream({'message': 'Tell me a short joke'}):
        reducer.process_event(event)

    # Get messages with metadata
    messages = reducer.get_messages(
        user_metadata={'source': 'web-app', 'user_id': 'user-456'},
        assistant_metadata={'model': 'gpt-4o', 'tokens': 150}
    )

    # Store in database
    store_messages(messages, conversation_id)

    print("\n✓ Messages stored in database:")
    for record in database:
        print(f"\n  Conversation: {record['conversation_id']}")
        print(f"  Message ID: {record['message_id']}")
        print(f"  Role: {record['role']}")
        print(f"  Parts: {len(record['parts'])} parts")
        print(f"  Metadata: {record['metadata']}")

    print("\n✓ Full database contents:")
    print(json.dumps(database, indent=2))

    print("\n" + "=" * 70)


async def custom_ids_example():
    """Example: Using custom message IDs"""
    print("\n" + "=" * 70)
    print("Custom IDs Example")
    print("=" * 70)

    reducer = MessageReducer()

    # Use custom message IDs (e.g., from your frontend)
    custom_user_id = 'frontend-msg-001'
    custom_assistant_id = 'frontend-msg-002'

    user_msg = reducer.add_user_message(
        "Hello",
        message_id=custom_user_id
    )

    agent = Agent(
        id='chat-agent',
        model={'provider': 'openai', 'model': 'gpt-4o'}
    )

    async for event in agent.run_stream({'message': 'Hello'}):
        reducer.process_event(event)

    # Get assistant message with custom ID
    assistant_msg = reducer.get_assistant_message(message_id=custom_assistant_id)

    print(f"\n✓ User message ID: {user_msg['id']}")
    print(f"✓ Assistant message ID: {assistant_msg['id']}")

    messages = [user_msg, assistant_msg]
    print("\n✓ Messages with custom IDs:")
    print(json.dumps(messages, indent=2))

    print("\n" + "=" * 70)


async def main():
    """Run all examples"""
    print("\n" + "=" * 70)
    print(" MESSAGE REDUCER EXAMPLES")
    print(" Aggregate streaming events into Vercel AI SDK message format")
    print("=" * 70)

    try:
        # Run examples
        await basic_example()
        await tool_calling_example()
        await multi_turn_example()
        await database_storage_example()
        await custom_ids_example()

        print("\n" + "=" * 70)
        print(" All examples completed successfully! ✓")
        print("=" * 70 + "\n")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    asyncio.run(main())
