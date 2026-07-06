---
layout: default
title: "HITL Frontend Integration"
nav_order: 13
---

# Human-in-the-Loop Frontend Integration

This guide covers integrating frontend UIs with Vel's human-in-the-loop (HITL) functionality using Vercel AI SDK and React components.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           Event Flow                                     │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Vel Agent                Stream Protocol              React UI          │
│  ─────────               ────────────────             ─────────          │
│                                                                          │
│  Agent decides    ──►    tool-input-start      ──►   Show "Calling..."  │
│  to call tool                                                            │
│                                                                          │
│  Args ready       ──►    tool-input-available  ──►   Render ApprovalCard│
│                                                                          │
│                          [Human clicks approve/reject]                   │
│                                                                          │
│  Tool executes    ◄──    POST /api/approvals   ◄──   Submit decision    │
│                                                                          │
│  Result ready     ──►    tool-output-available ──►   Show result        │
│                                                                          │
│  Agent continues  ──►    text-delta            ──►   Stream response    │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

## Stream Protocol Events

When a HITL tool like `request_approval` is called, Vel emits these events:

| Event | When | Frontend Action |
|-------|------|-----------------|
| `tool-input-start` | LLM decides to call tool | Show loading indicator |
| `tool-input-available` | Tool args ready | Render approval UI with action/reason |
| `tool-output-available` | Human responded | Display result, agent continues |

**Example event sequence:**

```json
{"type": "tool-input-start", "toolCallId": "call_123", "toolName": "request_approval"}
{"type": "tool-input-available", "toolCallId": "call_123", "toolName": "request_approval", "input": {"action": "delete", "reason": "cleanup old files"}}
// [Human approves via UI]
{"type": "tool-output-available", "toolCallId": "call_123", "output": {"approved": true, "approver": "user@example.com"}}
```

---

## Backend Setup (Python)

### 1. Define the Approval Tool

```python
from vel import ToolSpec, Agent
from typing import TypedDict
import asyncio

class ApprovalInput(TypedDict):
    action: str
    reason: str
    risk_level: str  # low, medium, high

class ApprovalOutput(TypedDict):
    approved: bool
    approver: str
    comment: str

# In-memory store (use Redis/DB in production)
pending_approvals: dict = {}
approval_events: dict = {}

async def request_approval(input: ApprovalInput, ctx: dict) -> ApprovalOutput:
    """
    Tool that pauses execution until human approves/rejects.
    """
    approval_id = ctx['tool_call_id']

    # Store pending approval
    pending_approvals[approval_id] = {
        'action': input['action'],
        'reason': input['reason'],
        'risk_level': input.get('risk_level', 'medium'),
        'status': 'pending',
        'run_id': ctx.get('run_id'),
        'session_id': ctx.get('session_id')
    }

    # Create event for async waiting
    approval_events[approval_id] = asyncio.Event()

    # Wait for human response (with timeout)
    try:
        await asyncio.wait_for(
            approval_events[approval_id].wait(),
            timeout=3600  # 1 hour
        )
    except asyncio.TimeoutError:
        return {
            'approved': False,
            'approver': 'system',
            'comment': 'Approval request timed out'
        }

    # Return the approval result
    result = pending_approvals[approval_id]
    del pending_approvals[approval_id]
    del approval_events[approval_id]

    return {
        'approved': result['approved'],
        'approver': result['approver'],
        'comment': result.get('comment', '')
    }

# Create tool using from_function
approval_tool = ToolSpec.from_function(
    request_approval,
    description="Request human approval for an action"
)
```

### 2. Create the Agent

```python
agent = Agent(
    id='deployment-agent:v1',
    model={'provider': 'openai', 'model': 'gpt-4o'},
    tools=[approval_tool, deploy_tool],
    policies={'max_steps': 10},
    session_persistence='persistent'
)
```

### 3. FastAPI Endpoints

```python
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
import json

app = FastAPI()

@app.post("/api/chat")
async def chat(request: Request):
    body = await request.json()
    message = body['messages'][-1]['content']
    session_id = body.get('session_id', 'default')

    async def generate():
        async for event in agent.run_stream(
            {'message': message},
            session_id=session_id
        ):
            # Stream as Vercel AI SDK format
            yield f"data: {json.dumps(event)}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream"
    )

@app.post("/api/approvals/{approval_id}")
async def submit_approval(approval_id: str, request: Request):
    """Endpoint for frontend to submit approval decision."""
    body = await request.json()

    if approval_id not in pending_approvals:
        return {"error": "Approval not found"}, 404

    # Update the pending approval
    pending_approvals[approval_id].update({
        'approved': body['approved'],
        'approver': body.get('approver', 'anonymous'),
        'comment': body.get('comment', ''),
        'status': 'approved' if body['approved'] else 'rejected'
    })

    # Signal the waiting coroutine
    if approval_id in approval_events:
        approval_events[approval_id].set()

    return {"status": "ok"}

@app.get("/api/approvals/pending")
async def list_pending():
    """List all pending approvals for admin UI."""
    return [
        {"id": k, **v}
        for k, v in pending_approvals.items()
        if v['status'] == 'pending'
    ]
```

---

## Frontend Integration (React)

### 1. Basic useChat Integration

```jsx
import { useChat } from 'ai/react';
import { useState, useCallback } from 'react';

function ApprovalChat() {
  const [pendingApprovals, setPendingApprovals] = useState({});

  const { messages, input, setInput, handleSubmit, status } = useChat({
    api: '/api/chat',

    // Intercept tool calls
    onToolCall: ({ toolCall }) => {
      if (toolCall.toolName === 'request_approval') {
        setPendingApprovals(prev => ({
          ...prev,
          [toolCall.toolCallId]: {
            id: toolCall.toolCallId,
            action: toolCall.args.action,
            reason: toolCall.args.reason,
            riskLevel: toolCall.args.risk_level || 'medium',
            status: 'pending'
          }
        }));
      }
    }
  });

  const handleApprovalDecision = useCallback(async (approvalId, approved, comment = '') => {
    // Update local state
    setPendingApprovals(prev => ({
      ...prev,
      [approvalId]: { ...prev[approvalId], status: approved ? 'approved' : 'rejected' }
    }));

    // Submit to backend
    await fetch(`/api/approvals/${approvalId}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ approved, comment })
    });

    // Remove from pending after delay
    setTimeout(() => {
      setPendingApprovals(prev => {
        const { [approvalId]: _, ...rest } = prev;
        return rest;
      });
    }, 2000);
  }, []);

  return (
    <div className="chat-container">
      {/* Message list */}
      <div className="messages">
        {messages.map(message => (
          <MessageRenderer key={message.id} message={message} />
        ))}
      </div>

      {/* Pending approval cards */}
      {Object.values(pendingApprovals).map(approval => (
        <ApprovalCard
          key={approval.id}
          approval={approval}
          onApprove={(comment) => handleApprovalDecision(approval.id, true, comment)}
          onReject={(comment) => handleApprovalDecision(approval.id, false, comment)}
        />
      ))}

      {/* Input */}
      <form onSubmit={handleSubmit}>
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Type a message..."
          disabled={Object.keys(pendingApprovals).length > 0}
        />
      </form>
    </div>
  );
}
```

### 2. Using onData for Custom Events

For richer HITL experiences, use the `onData` callback to handle custom events:

```jsx
const { messages } = useChat({
  api: '/api/chat',

  onData: (dataPart) => {
    // Handle HITL-specific events
    if (dataPart.type === 'data-hitl-status') {
      updateApprovalStatus(dataPart.data.approval_id, dataPart.data.status);
    }

    if (dataPart.type === 'data-hitl-timeout-warning') {
      showTimeoutWarning(dataPart.data.approval_id, dataPart.data.seconds_remaining);
    }
  }
});
```

---

## React Component Examples

These components match patterns from production chat UIs using Vercel AI elements.

### ApprovalCard Component

```jsx
import { Tool, ToolHeader, ToolContent, ToolInput } from '@/components/ai-elements';
import { CheckCircle, XCircle, Clock, AlertTriangle } from 'lucide-react';

const RISK_COLORS = {
  low: 'bg-green-50 border-green-200',
  medium: 'bg-yellow-50 border-yellow-200',
  high: 'bg-red-50 border-red-200'
};

const RISK_ICONS = {
  low: CheckCircle,
  medium: Clock,
  high: AlertTriangle
};

function ApprovalCard({ approval, onApprove, onReject }) {
  const [comment, setComment] = useState('');
  const [isSubmitting, setIsSubmitting] = useState(false);

  const RiskIcon = RISK_ICONS[approval.riskLevel] || Clock;

  const handleApprove = async () => {
    setIsSubmitting(true);
    await onApprove(comment);
    setIsSubmitting(false);
  };

  const handleReject = async () => {
    setIsSubmitting(true);
    await onReject(comment);
    setIsSubmitting(false);
  };

  if (approval.status !== 'pending') {
    return (
      <Tool>
        <ToolHeader
          state={approval.status}
          type="tool-approval"
          icon={approval.status === 'approved' ? CheckCircle : XCircle}
        />
        <ToolContent>
          <div className="text-sm text-gray-600">
            {approval.status === 'approved' ? 'Approved' : 'Rejected'}
            {approval.comment && `: ${approval.comment}`}
          </div>
        </ToolContent>
      </Tool>
    );
  }

  return (
    <Tool className={RISK_COLORS[approval.riskLevel]}>
      <ToolHeader
        state="pending"
        type="tool-approval"
        icon={RiskIcon}
      >
        Approval Required
      </ToolHeader>

      <ToolContent>
        <ToolInput input={{ action: approval.action, reason: approval.reason }} />

        <div className="mt-4 space-y-3">
          {/* Risk indicator */}
          <div className="flex items-center gap-2 text-sm">
            <RiskIcon className="w-4 h-4" />
            <span className="capitalize">{approval.riskLevel} risk</span>
          </div>

          {/* Action details */}
          <div className="bg-white rounded-md p-3 border">
            <div className="font-medium text-gray-900">{approval.action}</div>
            <div className="text-sm text-gray-600 mt-1">{approval.reason}</div>
          </div>

          {/* Comment input */}
          <textarea
            value={comment}
            onChange={(e) => setComment(e.target.value)}
            placeholder="Add a comment (optional)..."
            className="w-full p-2 border rounded-md text-sm"
            rows={2}
          />

          {/* Action buttons */}
          <div className="flex gap-2">
            <button
              onClick={handleApprove}
              disabled={isSubmitting}
              className="flex-1 bg-green-600 text-white py-2 px-4 rounded-md
                         hover:bg-green-700 disabled:opacity-50
                         flex items-center justify-center gap-2"
            >
              <CheckCircle className="w-4 h-4" />
              Approve
            </button>
            <button
              onClick={handleReject}
              disabled={isSubmitting}
              className="flex-1 bg-red-600 text-white py-2 px-4 rounded-md
                         hover:bg-red-700 disabled:opacity-50
                         flex items-center justify-center gap-2"
            >
              <XCircle className="w-4 h-4" />
              Reject
            </button>
          </div>
        </div>
      </ToolContent>
    </Tool>
  );
}
```

### ChainOfThought Integration

Show approval steps in the reasoning chain:

```jsx
import {
  ChainOfThought,
  ChainOfThoughtHeader,
  ChainOfThoughtContent,
  ChainOfThoughtStep
} from '@/components/ai-elements';
import { Brain, Search, UserCheck, Rocket } from 'lucide-react';

function AgentReasoningWithApproval({ steps, pendingApproval }) {
  return (
    <ChainOfThought defaultOpen>
      <ChainOfThoughtHeader>Agent Workflow</ChainOfThoughtHeader>
      <ChainOfThoughtContent>
        {/* Analysis step */}
        <ChainOfThoughtStep
          icon={Brain}
          label="Analysis"
          status="complete"
          description="Analyzed deployment requirements"
        />

        {/* Validation step */}
        <ChainOfThoughtStep
          icon={Search}
          label="Validation"
          status="complete"
          description="Validated configuration files"
        />

        {/* Approval step - dynamic status */}
        <ChainOfThoughtStep
          icon={UserCheck}
          label="Human Approval"
          status={pendingApproval ? 'active' : 'complete'}
          description={
            pendingApproval
              ? `Waiting for approval: ${pendingApproval.action}`
              : 'Deployment approved'
          }
        >
          {pendingApproval && (
            <div className="mt-2 p-2 bg-yellow-50 rounded text-sm">
              Action requires human review before proceeding.
            </div>
          )}
        </ChainOfThoughtStep>

        {/* Deploy step */}
        <ChainOfThoughtStep
          icon={Rocket}
          label="Deploy"
          status={pendingApproval ? 'pending' : 'complete'}
          description="Execute deployment"
        />
      </ChainOfThoughtContent>
    </ChainOfThought>
  );
}
```

### Message Part Renderer

Integrate approval into the message parts rendering pattern:

```jsx
function MessagePartRenderer({ part, isStreaming }) {
  switch (part.type) {
    case 'text':
      return <Response>{part.text}</Response>;

    case 'tool-approval':
      return (
        <ApprovalCard
          approval={{
            id: part.toolCallId,
            action: part.input?.action,
            reason: part.input?.reason,
            riskLevel: part.input?.risk_level || 'medium',
            status: part.state // 'pending' | 'approved' | 'rejected'
          }}
          onApprove={(comment) => submitApproval(part.toolCallId, true, comment)}
          onReject={(comment) => submitApproval(part.toolCallId, false, comment)}
        />
      );

    case 'tool-approval-result':
      return (
        <Tool>
          <ToolHeader state="complete" type="tool-approval" />
          <ToolOutput output={part.output} />
        </Tool>
      );

    case 'reasoning':
      return (
        <Reasoning isStreaming={isStreaming}>
          <ReasoningTrigger />
          <ReasoningContent>{part.text}</ReasoningContent>
        </Reasoning>
      );

    default:
      return null;
  }
}
```

---

## End-to-End Example

### Complete Backend (FastAPI + Vel)

```python
# app.py
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from vel import Agent, ToolSpec
import asyncio
import json

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"])

# Approval state
pending_approvals = {}
approval_events = {}

async def request_approval(action: str, reason: str, risk_level: str = "medium", ctx: dict = None) -> dict:
    """Request human approval for an action."""
    approval_id = ctx.get('tool_call_id', str(id(ctx)))

    pending_approvals[approval_id] = {
        'action': action,
        'reason': reason,
        'risk_level': risk_level,
        'status': 'pending'
    }
    approval_events[approval_id] = asyncio.Event()

    try:
        await asyncio.wait_for(approval_events[approval_id].wait(), timeout=300)
    except asyncio.TimeoutError:
        return {'approved': False, 'approver': 'timeout', 'comment': 'Request timed out'}

    result = pending_approvals.pop(approval_id)
    approval_events.pop(approval_id, None)

    return {
        'approved': result.get('approved', False),
        'approver': result.get('approver', 'unknown'),
        'comment': result.get('comment', '')
    }

async def deploy_to_environment(environment: str, version: str) -> dict:
    """Deploy application to specified environment."""
    await asyncio.sleep(2)  # Simulate deployment
    return {'status': 'success', 'environment': environment, 'version': version}

# Create agent
agent = Agent(
    id='deploy-agent:v1',
    model={'provider': 'openai', 'model': 'gpt-4o'},
    tools=[
        ToolSpec.from_function(request_approval),
        ToolSpec.from_function(deploy_to_environment)
    ],
    system_prompt="""You are a deployment assistant. When deploying to production,
    ALWAYS request approval first using the request_approval tool with risk_level='high'.
    For staging, use risk_level='medium'. For dev, risk_level='low'."""
)

@app.post("/api/chat")
async def chat(request: Request):
    body = await request.json()
    message = body['messages'][-1]['content']

    async def generate():
        async for event in agent.run_stream({'message': message}):
            yield f"data: {json.dumps(event)}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")

@app.post("/api/approvals/{approval_id}")
async def approve(approval_id: str, request: Request):
    body = await request.json()

    if approval_id in pending_approvals:
        pending_approvals[approval_id].update({
            'approved': body['approved'],
            'approver': body.get('approver', 'user'),
            'comment': body.get('comment', '')
        })
        approval_events[approval_id].set()
        return {"status": "ok"}

    return {"error": "Not found"}, 404
```

### Complete Frontend (React)

```jsx
// ApprovalChat.jsx
import { useChat } from 'ai/react';
import { useState, useCallback } from 'react';
import {
  Conversation,
  ConversationContent,
  Message,
  MessageContent,
  PromptInput,
  Tool,
  ToolHeader,
  ToolContent,
  ChainOfThought,
  ChainOfThoughtStep
} from '@/components/ai-elements';
import { CheckCircle, XCircle, UserCheck, Loader2 } from 'lucide-react';

export function ApprovalChat() {
  const [pendingApprovals, setPendingApprovals] = useState({});

  const { messages, input, setInput, handleSubmit, status } = useChat({
    api: '/api/chat',
    onToolCall: ({ toolCall }) => {
      if (toolCall.toolName === 'request_approval') {
        setPendingApprovals(prev => ({
          ...prev,
          [toolCall.toolCallId]: {
            id: toolCall.toolCallId,
            ...toolCall.args,
            status: 'pending'
          }
        }));
      }
    }
  });

  const submitApproval = useCallback(async (id, approved, comment = '') => {
    setPendingApprovals(prev => ({
      ...prev,
      [id]: { ...prev[id], status: approved ? 'approved' : 'rejected' }
    }));

    await fetch(`/api/approvals/${id}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ approved, comment })
    });

    setTimeout(() => {
      setPendingApprovals(prev => {
        const { [id]: _, ...rest } = prev;
        return rest;
      });
    }, 1500);
  }, []);

  const hasPendingApproval = Object.values(pendingApprovals).some(a => a.status === 'pending');

  return (
    <Conversation className="h-full">
      <ConversationContent>
        {messages.map(message => (
          <Message key={message.id} role={message.role}>
            <MessageContent>
              {message.content}
            </MessageContent>
          </Message>
        ))}

        {/* Inline approval cards */}
        {Object.values(pendingApprovals).map(approval => (
          <ApprovalCard
            key={approval.id}
            approval={approval}
            onApprove={(c) => submitApproval(approval.id, true, c)}
            onReject={(c) => submitApproval(approval.id, false, c)}
          />
        ))}

        {status === 'streaming' && !hasPendingApproval && (
          <div className="flex items-center gap-2 text-gray-500">
            <Loader2 className="w-4 h-4 animate-spin" />
            <span>Thinking...</span>
          </div>
        )}
      </ConversationContent>

      <PromptInput
        value={input}
        onChange={setInput}
        onSubmit={handleSubmit}
        disabled={hasPendingApproval}
        placeholder={hasPendingApproval ? "Waiting for approval..." : "Type a message..."}
      />
    </Conversation>
  );
}

function ApprovalCard({ approval, onApprove, onReject }) {
  const [comment, setComment] = useState('');

  if (approval.status !== 'pending') {
    const Icon = approval.status === 'approved' ? CheckCircle : XCircle;
    const color = approval.status === 'approved' ? 'text-green-600' : 'text-red-600';

    return (
      <Tool>
        <ToolHeader state={approval.status} type="tool-approval">
          <Icon className={`w-4 h-4 ${color}`} />
          {approval.status === 'approved' ? 'Approved' : 'Rejected'}
        </ToolHeader>
      </Tool>
    );
  }

  return (
    <Tool className="border-yellow-300 bg-yellow-50">
      <ToolHeader state="pending" type="tool-approval">
        <UserCheck className="w-4 h-4 text-yellow-600" />
        Approval Required
      </ToolHeader>
      <ToolContent>
        <div className="space-y-3">
          <div className="bg-white p-3 rounded border">
            <div className="font-medium">{approval.action}</div>
            <div className="text-sm text-gray-600">{approval.reason}</div>
            <div className="text-xs text-gray-400 mt-1">
              Risk: <span className="capitalize">{approval.risk_level}</span>
            </div>
          </div>

          <textarea
            value={comment}
            onChange={(e) => setComment(e.target.value)}
            placeholder="Optional comment..."
            className="w-full p-2 border rounded text-sm"
            rows={2}
          />

          <div className="flex gap-2">
            <button
              onClick={() => onApprove(comment)}
              className="flex-1 bg-green-600 text-white py-2 rounded hover:bg-green-700"
            >
              Approve
            </button>
            <button
              onClick={() => onReject(comment)}
              className="flex-1 bg-red-600 text-white py-2 rounded hover:bg-red-700"
            >
              Reject
            </button>
          </div>
        </div>
      </ToolContent>
    </Tool>
  );
}
```

---

## Advanced Patterns

### Timeout Handling with Progress

```jsx
function ApprovalCardWithTimeout({ approval, onApprove, onReject, timeoutSeconds = 300 }) {
  const [secondsRemaining, setSecondsRemaining] = useState(timeoutSeconds);

  useEffect(() => {
    if (approval.status !== 'pending') return;

    const timer = setInterval(() => {
      setSecondsRemaining(prev => {
        if (prev <= 1) {
          clearInterval(timer);
          return 0;
        }
        return prev - 1;
      });
    }, 1000);

    return () => clearInterval(timer);
  }, [approval.status]);

  const progress = (secondsRemaining / timeoutSeconds) * 100;

  return (
    <div className="relative">
      {/* Progress bar */}
      <div className="absolute top-0 left-0 h-1 bg-gray-200 w-full rounded-t">
        <div
          className="h-full bg-yellow-500 transition-all duration-1000"
          style={{ width: `${progress}%` }}
        />
      </div>

      <ApprovalCard
        approval={approval}
        onApprove={onApprove}
        onReject={onReject}
      />

      <div className="text-xs text-gray-500 text-center mt-1">
        {Math.floor(secondsRemaining / 60)}:{(secondsRemaining % 60).toString().padStart(2, '0')} remaining
      </div>
    </div>
  );
}
```

### Concurrent Approval Queue

```jsx
function ApprovalQueue({ approvals, onDecision }) {
  const pendingApprovals = Object.values(approvals).filter(a => a.status === 'pending');
  const [currentIndex, setCurrentIndex] = useState(0);

  const handleDecision = async (approved, comment) => {
    const current = pendingApprovals[currentIndex];
    await onDecision(current.id, approved, comment);

    if (currentIndex < pendingApprovals.length - 1) {
      setCurrentIndex(currentIndex + 1);
    }
  };

  if (pendingApprovals.length === 0) return null;

  return (
    <div className="space-y-2">
      <div className="text-sm text-gray-600">
        Approval {currentIndex + 1} of {pendingApprovals.length}
      </div>

      <ApprovalCard
        approval={pendingApprovals[currentIndex]}
        onApprove={(c) => handleDecision(true, c)}
        onReject={(c) => handleDecision(false, c)}
      />

      {/* Navigation dots */}
      <div className="flex justify-center gap-1">
        {pendingApprovals.map((_, i) => (
          <button
            key={i}
            onClick={() => setCurrentIndex(i)}
            className={`w-2 h-2 rounded-full ${
              i === currentIndex ? 'bg-blue-500' : 'bg-gray-300'
            }`}
          />
        ))}
      </div>
    </div>
  );
}
```

### Approval Audit Trail

```jsx
function ApprovalHistory({ approvals }) {
  const completedApprovals = Object.values(approvals)
    .filter(a => a.status !== 'pending')
    .sort((a, b) => new Date(b.timestamp) - new Date(a.timestamp));

  return (
    <div className="border rounded-lg p-4">
      <h3 className="font-medium mb-3">Approval History</h3>
      <div className="space-y-2">
        {completedApprovals.map(approval => (
          <div
            key={approval.id}
            className={`flex items-center gap-3 p-2 rounded ${
              approval.status === 'approved' ? 'bg-green-50' : 'bg-red-50'
            }`}
          >
            {approval.status === 'approved'
              ? <CheckCircle className="w-4 h-4 text-green-600" />
              : <XCircle className="w-4 h-4 text-red-600" />
            }
            <div className="flex-1">
              <div className="text-sm font-medium">{approval.action}</div>
              <div className="text-xs text-gray-500">
                {approval.approver} • {new Date(approval.timestamp).toLocaleString()}
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
```

---

## Summary

| Component | Purpose |
|-----------|---------|
| `ApprovalCard` | Renders approval request with action/reason and approve/reject buttons |
| `ChainOfThoughtStep` | Shows approval as a step in agent reasoning visualization |
| `MessagePartRenderer` | Integrates approval into message parts rendering pattern |
| `ApprovalQueue` | Handles multiple concurrent approval requests |
| `ApprovalHistory` | Displays audit trail of past decisions |

**Key patterns:**
1. Use `onToolCall` callback to detect approval tools
2. Store pending approvals in React state
3. Submit decisions via REST endpoint that unblocks the async tool handler
4. Use existing ai-elements (`<Tool>`, `<ToolHeader>`, etc.) for consistent styling

## Durable approval (Harness Mode)

The pattern above uses an **in-process** approval callback: the async tool
handler blocks until a REST decision unblocks it. That requires the request /
process to stay alive for the whole wait.

[Harness Mode](./harness) adds a **durable** variant for long waits, browser
round-trips, and process restarts. Instead of blocking, the run **suspends**:

1. The model calls an approval-required tool (named in
   `approval.require_for_tools`, or any tool with `requires_confirmation=True`).
2. The harness snapshots a checkpoint (messages + pending approvals + budget),
   emits the usual `tool-input-available` (so the existing approval card still
   renders) **plus** an additive `data-harness-approval-required`
   (`{approval_id, run_id, tool_call_id, tool_name, reason}`), then a
   `data-harness-suspended`, and returns. The run is now persisted.
3. The browser can disconnect. Whenever the human decides, the host calls:

   ```python
   from vel.harness import ApprovalDecision
   async for event in agent.resume(run_id, [ApprovalDecision(approval_id, "approve")]):
       ...   # approved tools execute; rejected tools get a deny result; loop continues
   ```

   — or `POST /runs/{run_id}/approvals` against the
   [`RunManager` reference server](../examples/harness/fastapi_server.py).

The decision survives a process restart (persisted in `vel_approvals` /
`vel_checkpoints`), so `resume()` works from a brand-new `Agent` instance.
`approval.mode="inline"` keeps the legacy blocking-callback behavior above.
See [Harness Mode → Durable HITL approval](./harness#durable-hitl-approval).

**See also:**
- [Factor 7: Contact Humans](./12-factor/factor-07)
- [Stream Protocol](./stream-protocol)
- [Tools](./tools)
- [Harness Mode](./harness)
