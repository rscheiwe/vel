"""
Custom Data Events Example

Demonstrates the use of custom data-* events for application-specific streaming:
- data-notification: UI notifications (transient)
- data-progress: Progress updates (transient)
- data-stage-data: Stage transitions (persistent)
- data-metrics: Real-time metrics (persistent)

Matches Vercel AI SDK V5 custom data pattern.
"""
import asyncio
from vel import Agent, MessageReducer, DataEvent


class CustomAgent:
    """
    Example agent that emits custom data-* events during processing.

    In a real implementation, these would be emitted by providers or tools.
    This example simulates the pattern for demonstration.
    """

    def __init__(self, agent: Agent):
        self.agent = agent

    async def run_with_custom_events(self, input_dict: dict):
        """Simulate agent run with custom data events"""

        # Emit transient notification (won't be saved to history)
        yield DataEvent(
            type='data-notification',
            data={
                'message': 'Processing your request...',
                'level': 'info'
            },
            transient=True
        ).to_dict()

        await asyncio.sleep(0.5)

        # Emit stage transition (persistent - saved to history)
        yield DataEvent(
            type='data-stage-data',
            data={
                'stage': 'analyzing',
                'description': 'Analyzing your question'
            },
            transient=False
        ).to_dict()

        await asyncio.sleep(0.5)

        # Emit progress update (transient)
        yield DataEvent(
            type='data-progress',
            data={
                'percent': 50,
                'message': 'Halfway there...'
            },
            transient=True
        ).to_dict()

        await asyncio.sleep(0.5)

        # Emit metrics (persistent)
        yield DataEvent(
            type='data-metrics',
            data={
                'processing_time_ms': 1500,
                'tokens_used': 250
            },
            transient=False
        ).to_dict()

        # Now run the actual agent
        async for event in self.agent.run_stream(input_dict):
            yield event


async def basic_example():
    """Basic example: Custom data events with reducer"""
    print("="*70)
    print("Basic Example: Custom Data Events")
    print("="*70)

    # Create agent
    agent = Agent(
        id='custom-data-agent',
        model={'provider': 'openai', 'model': 'gpt-4o'}
    )

    custom_agent = CustomAgent(agent)

    # Create reducer
    reducer = MessageReducer()
    reducer.add_user_message("Hello! How are you?")

    print("\n🔄 Streaming with custom data events...\n")

    async for event in custom_agent.run_with_custom_events({'message': 'Hello! How are you?'}):
        event_type = event.get('type', '')
        is_transient = event.get('transient', False)

        # Show all events
        if event_type.startswith('data-'):
            data = event.get('data', {})
            transient_label = " [TRANSIENT]" if is_transient else " [PERSISTENT]"
            print(f"  📊 {event_type}{transient_label}")
            print(f"     Data: {data}\n")

        elif event_type == 'text-delta':
            print(event.get('delta', ''), end='', flush=True)

        # Process through reducer
        reducer.process_event(event)

    # Get messages
    messages = reducer.get_messages()
    assistant_msg = messages[1]

    print(f"\n\n{'='*70}")
    print("Message Parts (only persistent events saved):")
    print("="*70)

    for i, part in enumerate(assistant_msg['parts'], 1):
        part_type = part.get('type', '')
        print(f"\n  Part {i}: {part_type}")

        if part_type.startswith('data-'):
            print(f"    Data: {part.get('data', {})}")
        elif part_type == 'text':
            print(f"    Text: {part.get('text', '')[:60]}...")

    print(f"\n{'='*70}\n")


async def multi_step_with_custom_data():
    """Example: Multi-step agent with stage tracking via custom data"""
    print("\n" + "="*70)
    print("Multi-Step Example: Stage Tracking with data-stage-data")
    print("="*70)

    # In a real implementation, tools would emit data-stage-data events
    # to track which stage of processing they're in

    print("""
This pattern is useful for multi-step agents to communicate:
- Current stage (analyzing, deciding, executing, etc.)
- Progress within each stage
- Metadata about the step

Example data-stage-data events:

  {
    "type": "data-stage-data",
    "data": {
      "stage": "analyzing",
      "step": 1,
      "total_steps": 5,
      "description": "Breaking down the problem"
    },
    "transient": false  // Saved to history
  }

These events are added to the message parts array and can be:
1. Stored in the database for debugging/analytics
2. Replayed to reconstruct the agent's thought process
3. Used by frontends to show step-by-step progress

Transient events (transient: true) are NOT saved to message history:
- Good for real-time UI updates (loading spinners, etc.)
- Won't clutter your database
- Still sent to frontend via onData callback

""")


async def frontend_integration_pattern():
    """Example: How frontend would handle custom data events"""
    print("="*70)
    print("Frontend Integration Pattern")
    print("="*70)

    print("""
**Backend (Vel):**

```python
from vel import Agent, DataEvent

agent = Agent(id='agent', model={'provider': 'openai', 'model': 'gpt-4o'})

async def run_with_notifications(input_dict):
    # Transient notification
    yield DataEvent(
        type='data-notification',
        data={'message': 'Processing...', 'level': 'info'},
        transient=True
    ).to_dict()

    # Persistent stage transition
    yield DataEvent(
        type='data-stage-data',
        data={'stage': 'analyzing'},
        transient=False
    ).to_dict()

    # Stream agent response
    async for event in agent.run_stream(input_dict):
        yield event
```

**Frontend (React + Vercel AI SDK):**

```typescript
import { useChat } from 'ai/react';

const { messages, sendMessage } = useChat({
  api: '/api/chat',

  // Handle transient data events (NOT in message history)
  onData: (dataPart) => {
    if (dataPart.type === 'data-notification') {
      toast.info(dataPart.data.message);
    }

    if (dataPart.type === 'data-progress') {
      setProgress(dataPart.data.percent);
    }
  }
});

// Access persistent data events from message.parts
messages.forEach(msg => {
  msg.parts.forEach(part => {
    if (part.type === 'data-stage-data') {
      console.log('Stage:', part.data.stage);
    }
  });
});
```

**Key Points:**
- Transient events: Real-time UI updates via `onData` callback
- Persistent events: Stored in message.parts, can be replayed
- Same event structure works across Vel and Vercel AI SDK
""")


async def main():
    """Run all examples"""
    await basic_example()
    await multi_step_with_custom_data()
    await frontend_integration_pattern()

    print("\n" + "="*70)
    print("All examples complete!")
    print("="*70 + "\n")


if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()
