---
layout: default
title: "Factor 9: Compact Errors into Context"
parent: 12-Factor Alignment
nav_order: 9
---

# Factor 9: Compact Errors into Context Window

**Principle:** Efficiently handle and communicate errors to improve resilience.

## How Vel Implements This

Vel provides structured error events and context integration:

```python
# Errors emitted as stream events
async for event in agent.run_stream({'message': 'Deploy app'}):
    if event['type'] == 'error':
        error_message = event['error']
        # Error automatically added to context
        # Agent can recover on next iteration

# Tool errors handled gracefully
async def flaky_tool_handler(input: dict, ctx: dict) -> dict:
    try:
        result = await unstable_api_call(input)
        return {'success': True, 'data': result}
    except Exception as e:
        # Return error as structured output
        return {
            'success': False,
            'error': str(e)[:200],  # Compact error message
            'retry_suggested': True
        }

# Errors compacted in context
class ErrorCompactingContextManager(ContextManager):
    def messages_for_llm(self, run_id: str, session_id: Optional[str] = None):
        messages = super().messages_for_llm(run_id, session_id)

        # Compact repeated errors
        compacted = []
        error_counts = {}

        for msg in messages:
            if 'error' in msg.get('content', '').lower():
                error_type = self.extract_error_type(msg['content'])
                error_counts[error_type] = error_counts.get(error_type, 0) + 1
                if error_counts[error_type] == 1:
                    compacted.append(msg)
                elif error_counts[error_type] == 3:
                    compacted.append({
                        'role': 'system',
                        'content': f"Note: {error_type} occurred 3+ times"
                    })
            else:
                compacted.append(msg)

        return compacted
```

## Benefits

- ✓ Structured error events
- ✓ Errors automatically added to context
- ✓ Custom error compaction strategies
- ✓ Agent can learn from and recover from errors

**See:** [Stream Protocol - Error Event](../stream-protocol#error)
