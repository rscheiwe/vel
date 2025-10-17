"""
Test MessageReducer with reasoning events (o1/o3 models)

Shows how reasoning events are converted to AI SDK message format.
"""
import asyncio
import json
from vel import MessageReducer

async def test_reasoning_reducer():
    """Test that reasoning events are properly converted to message parts"""

    reducer = MessageReducer()

    # Add user message
    reducer.add_user_message("What is the square root of 169?")

    # Simulate streaming events from o1 model
    events = [
        {'type': 'start'},
        {'type': 'start-step'},
        {'type': 'reasoning-start', 'id': 'rs_084b321b9f935e570068f123d553d881989b96899a08889f58'},
        # Note: OpenAI encrypts reasoning, so no reasoning-delta events
        {'type': 'reasoning-end', 'id': 'rs_084b321b9f935e570068f123d553d881989b96899a08889f58'},
        {'type': 'text-start', 'id': 'msg_084b321b9f935e570068f123d6e7848198a9663aa96c84ccef'},
        {'type': 'text-delta', 'id': 'msg_084b321b9f935e570068f123d6e7848198a9663aa96c84ccef', 'delta': 'The '},
        {'type': 'text-delta', 'id': 'msg_084b321b9f935e570068f123d6e7848198a9663aa96c84ccef', 'delta': 'square '},
        {'type': 'text-delta', 'id': 'msg_084b321b9f935e570068f123d6e7848198a9663aa96c84ccef', 'delta': 'root '},
        {'type': 'text-delta', 'id': 'msg_084b321b9f935e570068f123d6e7848198a9663aa96c84ccef', 'delta': 'of '},
        {'type': 'text-delta', 'id': 'msg_084b321b9f935e570068f123d6e7848198a9663aa96c84ccef', 'delta': '169 '},
        {'type': 'text-delta', 'id': 'msg_084b321b9f935e570068f123d6e7848198a9663aa96c84ccef', 'delta': 'is '},
        {'type': 'text-delta', 'id': 'msg_084b321b9f935e570068f123d6e7848198a9663aa96c84ccef', 'delta': '13.'},
        {'type': 'text-end', 'id': 'msg_084b321b9f935e570068f123d6e7848198a9663aa96c84ccef'},
        {'type': 'finish-message', 'finishReason': 'stop'},
        {'type': 'finish-step'},
        {'type': 'finish'}
    ]

    # Process all events
    for event in events:
        reducer.process_event(event)

    # Get messages
    messages = reducer.get_messages()

    print("=" * 70)
    print("AI SDK Message Format with Reasoning")
    print("=" * 70)
    print()
    print(json.dumps(messages, indent=2))
    print()

    # Verify structure
    assistant_msg = messages[1]
    parts = assistant_msg['parts']

    print("=" * 70)
    print("Parts Analysis")
    print("=" * 70)
    print(f"Total parts: {len(parts)}")
    print()

    for i, part in enumerate(parts):
        print(f"Part {i + 1}: type='{part['type']}'")
        if part['type'] == 'reasoning':
            print(f"  - text: '{part['text']}' (empty = encrypted)")
            print(f"  - state: {part['state']}")
            print(f"  - providerMetadata: {part.get('providerMetadata')}")
        elif part['type'] == 'text':
            print(f"  - text: '{part['text'][:50]}...' ({len(part['text'])} chars)")
            print(f"  - state: {part['state']}")
    print()

if __name__ == '__main__':
    asyncio.run(test_reasoning_reducer())
