---
layout: default
title: "Factor 7: Contact Humans with Tool Calls"
parent: 12-Factor Alignment
nav_order: 7
---

# Factor 7: Contact Humans with Tool Calls

**Principle:** Integrate human intervention directly into AI workflows through tools.

## How Vel Implements This

Human-in-the-loop as a tool:

```python
from vel import ToolSpec, register_tool

async def request_human_approval(input: dict, ctx: dict) -> dict:
    """Tool that contacts a human for approval"""
    action = input['action']
    reason = input['reason']

    # Send to approval system
    approval_request = await approval_system.create({
        'run_id': ctx['run_id'],
        'session_id': ctx['session_id'],
        'action': action,
        'reason': reason,
        'status': 'pending'
    })

    # Wait for human response (webhook or polling)
    approval = await approval_system.wait_for_response(
        approval_request['id'],
        timeout=3600  # 1 hour
    )

    return {
        'approved': approval['approved'],
        'comment': approval.get('comment', ''),
        'approver': approval['approver']
    }

approval_tool = ToolSpec(
    name='request_approval',
    input_schema={
        'type': 'object',
        'properties': {
            'action': {'type': 'string'},
            'reason': {'type': 'string'}
        },
        'required': ['action', 'reason']
    },
    output_schema={
        'type': 'object',
        'properties': {
            'approved': {'type': 'boolean'},
            'comment': {'type': 'string'},
            'approver': {'type': 'string'}
        },
        'required': ['approved']
    },
    handler=request_human_approval
)

register_tool(approval_tool)

agent = Agent(
    id='deployment-agent',
    model={'provider': 'openai', 'model': 'gpt-4o'},
    tools=['request_approval', 'deploy']
)
```

## Benefits

- ✓ Human approval as a tool call
- ✓ Async tool handlers support long-running approvals
- ✓ Session persistence enables multi-hour workflows
- ✓ Clear audit trail of human decisions

## Frontend Integration

Vel's stream protocol emits tool events that frontends can use to render approval UIs:

```
tool-input-start      → Show "Requesting approval..."
tool-input-available  → Render approval card with action/reason
[human decision]      → User clicks approve/reject
tool-output-available → Display result, agent continues
```

**Quick example with Vercel AI SDK:**

```jsx
import { useChat } from 'ai/react';

function Chat() {
  const [pendingApproval, setPendingApproval] = useState(null);

  const { messages } = useChat({
    api: '/api/chat',
    onToolCall: ({ toolCall }) => {
      if (toolCall.toolName === 'request_approval') {
        setPendingApproval({
          id: toolCall.toolCallId,
          action: toolCall.args.action,
          reason: toolCall.args.reason
        });
      }
    }
  });

  const handleApprove = async (approved) => {
    await fetch(`/api/approvals/${pendingApproval.id}`, {
      method: 'POST',
      body: JSON.stringify({ approved })
    });
    setPendingApproval(null);
  };

  return (
    <div>
      {/* Chat messages */}
      {pendingApproval && (
        <ApprovalCard
          action={pendingApproval.action}
          reason={pendingApproval.reason}
          onApprove={() => handleApprove(true)}
          onReject={() => handleApprove(false)}
        />
      )}
    </div>
  );
}
```

**See:** [HITL Frontend Integration Guide](../hitl-frontend) for complete implementation with React components, streaming events, and end-to-end examples.

**See also:** [Tools - Async Tools](../tools#async-tool)
