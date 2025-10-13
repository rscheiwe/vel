---
layout: default
title: 12-Factor Alignment
nav_order: 9
---

# 12-Factor Agent Alignment

How Vel aligns with the [12-Factor Agents](https://github.com/humanlayer/12-factor-agents) principles for building reliable, production-ready LLM applications.

## Overview

The 12-Factor Agents methodology, created by [Dex](https://github.com/humanlayer) and contributors, provides principles for building LLM-powered software that is reliable, scalable, and maintainable. Vel is designed from the ground up to embody these principles, giving developers full control over their AI agents while maintaining simplicity and production-readiness.

This document describes how Vel implements these principles. The original 12-Factor Agents content is available at [github.com/humanlayer/12-factor-agents](https://github.com/humanlayer/12-factor-agents) and is licensed under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).

**Core Philosophy (from 12-Factor Agents):** "The fastest way to get good AI software in the hands of customers is to take small, modular concepts from agent building, and incorporate them into existing products."

---

## Factor 1: Natural Language to Tool Calls

**Principle:** Translate natural language into structured tool calls for reliable execution.

**How Vel Implements This:**

Vel uses LLM function calling to convert natural language into structured tool invocations with JSON schema validation:

```python
from agents import Agent, ToolSpec, register_tool

def get_weather_handler(input: dict, ctx: dict) -> dict:
    city = input['city']
    return {'temp_f': 72, 'condition': 'sunny', 'city': city}

weather_tool = ToolSpec(
    name='get_weather',
    input_schema={
        'type': 'object',
        'properties': {'city': {'type': 'string'}},
        'required': ['city']
    },
    output_schema={
        'type': 'object',
        'properties': {
            'temp_f': {'type': 'number'},
            'condition': {'type': 'string'}
        },
        'required': ['temp_f', 'condition']
    },
    handler=get_weather_handler
)

register_tool(weather_tool)

agent = Agent(
    id='weather-agent',
    model={'provider': 'openai', 'model': 'gpt-4o'},
    tools=['get_weather']
)

# Natural language → Structured tool call
answer = await agent.run({'message': 'What is the weather in Tokyo?'})
```

**Benefits:**
- ✓ JSON schema validation ensures type safety
- ✓ Provider-agnostic tool format works across OpenAI, Gemini, and Claude
- ✓ Clear separation between LLM decision-making and code execution

**See:** [Tools Documentation](tools.md)

---

## Factor 2: Own Your Prompts

**Principle:** Take direct control of prompts instead of outsourcing to framework abstractions.

**How Vel Implements This:**

Vel provides the primitives but doesn't hide prompts behind abstractions. You have full control:

```python
# Direct access to messages sent to LLM
messages = agent.ctxmgr.messages_for_llm(run_id, session_id)

# Custom context manager for full prompt control
class CustomContextManager(ContextManager):
    def messages_for_llm(self, run_id: str, session_id: Optional[str] = None):
        messages = super().messages_for_llm(run_id, session_id)

        # Add custom system message
        messages.insert(0, {
            'role': 'system',
            'content': 'You are a helpful deployment assistant. Always confirm before deploying.'
        })

        # Add retrieved context (RAG)
        retrieved_docs = self.retrieve_docs(session_id)
        messages.insert(1, {
            'role': 'system',
            'content': f"Relevant context: {retrieved_docs}"
        })

        return messages

agent = Agent(
    id='my-agent',
    model={'provider': 'openai', 'model': 'gpt-4o'},
    context_manager=CustomContextManager()
)
```

**Benefits:**
- ✓ Full transparency: see exactly what's sent to the LLM
- ✓ Easy iteration: modify prompts based on performance
- ✓ Testable: create evaluations like regular code
- ✓ No hidden abstractions: prompts are first-class code

**See:** [Session Management](sessions.md#advanced-usage)

---

## Factor 3: Own Your Context Window

**Principle:** Everything is context engineering. Structure information to maximize LLM understanding.

**How Vel Implements This:**

Vel gives you complete control over context window management:

```python
# Limit context window to control token usage
agent = Agent(
    id='my-agent',
    model={'provider': 'openai', 'model': 'gpt-4o'},
    context_manager=ContextManager(max_history=20)  # Sliding window
)

# Stateless context (no history)
agent = Agent(
    id='my-agent',
    model={'provider': 'openai', 'model': 'gpt-4o'},
    context_manager=StatelessContextManager()  # Each call independent
)

# Custom context structuring
class RAGContextManager(ContextManager):
    def messages_for_llm(self, run_id: str, session_id: Optional[str] = None):
        messages = super().messages_for_llm(run_id, session_id)

        # Compress old messages into summary
        if len(messages) > 10:
            old_messages = messages[:8]
            summary = self.summarize(old_messages)
            messages = [
                {'role': 'system', 'content': f"Previous conversation: {summary}"}
            ] + messages[8:]

        return messages
```

**Benefits:**
- ✓ Configurable memory: full/stateless/limited
- ✓ Custom context managers for advanced use cases
- ✓ Direct access to message history
- ✓ Token optimization through context management

**See:** [Session Management - Context Manager Modes](sessions.md#context-manager-modes)

---

## Factor 4: Tools are Structured Outputs

**Principle:** Tools are JSON outputs that trigger deterministic code. Clean separation between LLM decisions and code execution.

**How Vel Implements This:**

Vel treats tool calls as structured outputs with validation:

```python
# Tools are structured outputs with JSON schemas
tool = ToolSpec(
    name='create_deployment',
    input_schema={
        'type': 'object',
        'properties': {
            'environment': {'type': 'string', 'enum': ['dev', 'staging', 'prod']},
            'version': {'type': 'string'}
        },
        'required': ['environment', 'version']
    },
    output_schema={
        'type': 'object',
        'properties': {
            'deployment_id': {'type': 'string'},
            'status': {'type': 'string'}
        },
        'required': ['deployment_id', 'status']
    },
    handler=create_deployment_handler
)

# LLM decides "what", your code controls "how"
async def create_deployment_handler(input: dict, ctx: dict) -> dict:
    # Your deterministic code
    environment = input['environment']
    version = input['version']

    # Complex business logic
    if environment == 'prod':
        # Send to approval queue
        approval_id = await request_approval(environment, version)
        return {'deployment_id': approval_id, 'status': 'pending_approval'}
    else:
        # Direct deployment
        deployment_id = await deploy(environment, version)
        return {'deployment_id': deployment_id, 'status': 'deployed'}
```

**Benefits:**
- ✓ LLM outputs structured JSON
- ✓ Your code executes deterministically
- ✓ Validation at input and output boundaries
- ✓ Tool execution not limited to atomic functions

**See:** [Tools Documentation](tools.md)

---

## Factor 5: Unify Execution State and Business State

**Principle:** Integrate AI execution state with business logic for coherent systems.

**How Vel Implements This:**

Vel stores execution state alongside business state:

```python
# Execution state stored in Postgres
agent = Agent(
    id='deployment-agent',
    model={'provider': 'openai', 'model': 'gpt-4o'},
    session_storage='database'  # Persistent execution state
)

# Access execution state
run_id = await agent.store.create_run(agent.id)
await agent.store.append_event(run_id, {
    'kind': 'tool_result',
    'tool': 'create_deployment',
    'result': {'deployment_id': 'dep-123', 'status': 'deployed'}
})

# Tool handlers can access and update business state
async def deployment_handler(input: dict, ctx: dict) -> dict:
    run_id = ctx['run_id']
    session_id = ctx['session_id']

    # Execute deployment
    deployment = await create_deployment(input)

    # Update business database
    await db.deployments.insert({
        'id': deployment['deployment_id'],
        'run_id': run_id,  # Link to agent execution
        'session_id': session_id,
        'status': deployment['status'],
        'created_at': datetime.now()
    })

    return deployment
```

**Benefits:**
- ✓ Execution history persisted in database
- ✓ Tool context includes run_id and session_id
- ✓ Easy correlation between agent actions and business outcomes
- ✓ Audit trail for compliance

**See:** [Session Management - Database Storage](sessions.md#database-storage)

---

## Factor 6: Launch/Pause/Resume with Simple APIs

**Principle:** Provide flexible control over agent workflows with simple, composable APIs.

**How Vel Implements This:**

Vel provides dual execution modes with async control:

```python
import asyncio
from agents import Agent

agent = Agent(
    id='my-agent',
    model={'provider': 'openai', 'model': 'gpt-4o'},
    session_storage='database'  # Persist state for resume
)

# Launch
async def launch_agent(user_input: str, session_id: str):
    """Start agent execution"""
    async for event in agent.run_stream(
        {'message': user_input},
        session_id=session_id
    ):
        if event['type'] == 'tool-input-available':
            # Pause for approval if needed
            if event['toolName'] == 'deploy_to_prod':
                await request_approval(event, session_id)
                raise PauseExecution(event)

        yield event

# Resume
async def resume_agent(approval: dict, session_id: str):
    """Resume after approval"""
    # Load previous context from database
    context = agent.ctxmgr.get_session_context(session_id)

    # Add approval to context
    context.append({
        'role': 'user',
        'content': f"Approval granted: {approval}"
    })
    agent.ctxmgr.set_session_context(session_id, context)

    # Continue execution
    async for event in agent.run_stream(
        {'message': 'Continue with approved deployment'},
        session_id=session_id
    ):
        yield event

# Cancellation support
task = asyncio.create_task(launch_agent(input, session_id))
# ... later
task.cancel()  # Graceful cancellation
```

**Benefits:**
- ✓ Streaming enables real-time pause/resume
- ✓ Database persistence enables resume after restart
- ✓ AsyncIO cancellation support
- ✓ Session context preserved across interruptions

**See:** [Getting Started - Streaming Mode](getting-started.md#streaming-mode)

---

## Factor 7: Contact Humans with Tool Calls

**Principle:** Integrate human intervention directly into AI workflows through tools.

**How Vel Implements This:**

Human-in-the-loop as a tool:

```python
from agents import ToolSpec, register_tool

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

**Benefits:**
- ✓ Human approval as a tool call
- ✓ Async tool handlers support long-running approvals
- ✓ Session persistence enables multi-hour workflows
- ✓ Clear audit trail of human decisions

**See:** [Tools - Async Tools](tools.md#async-tool)

---

## Factor 8: Own Your Control Flow

**Principle:** Maintain explicit control over agent decision-making to prevent unpredictable behavior.

**How Vel Implements This:**

Vel provides explicit control flow through policies and the reducer pattern:

```python
# Policy-based control
agent = Agent(
    id='my-agent',
    model={'provider': 'openai', 'model': 'gpt-4o'},
    tools=['deploy', 'rollback'],
    policies={
        'max_steps': 5,  # Limit tool calls per run
        'retry': {'attempts': 2}  # Retry failed tools
    }
)

# Explicit state transitions (non-streaming mode)
# The reducer pattern gives full control over state transitions
from agents.core import State, reduce

state = State(run_id='run-123')
event = {'kind': 'llm_step', 'step': {'tool': 'deploy', 'args': {...}}}

# You control the state transition
new_state, effects = reduce(state, event)

for effect in effects:
    if effect.kind == 'call_tool':
        # You decide whether to execute the tool
        if should_execute(effect.payload['tool']):
            result = await execute_tool(effect.payload['tool'], effect.payload['args'])
        else:
            result = {'error': 'Tool execution blocked by policy'}

# Custom control flow in streaming mode
async for event in agent.run_stream({'message': 'Deploy backend'}):
    if event['type'] == 'tool-input-available':
        tool_name = event['toolName']

        # Your custom control flow
        if tool_name == 'deploy' and not is_business_hours():
            # Block deployment
            print("Deployment blocked: outside business hours")
            break
```

**Benefits:**
- ✓ Policy-based limits prevent runaway execution
- ✓ Explicit state transitions via reducer (non-streaming)
- ✓ Full visibility into agent decisions
- ✓ Easy to add guards and validations

**See:** [API Reference - Agent Policies](api-reference.md#constructor)

---

## Factor 9: Compact Errors into Context Window

**Principle:** Efficiently handle and communicate errors to improve resilience.

**How Vel Implements This:**

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

**Benefits:**
- ✓ Structured error events
- ✓ Errors automatically added to context
- ✓ Custom error compaction strategies
- ✓ Agent can learn from and recover from errors

**See:** [Stream Protocol - Error Event](stream-protocol.md#error)

---

## Factor 10: Small, Focused Agents

**Principle:** Create specialized agents rather than monolithic ones for better reliability.

**How Vel Implements This:**

Vel encourages small, focused agents through composition:

```python
# Specialized agents for specific tasks
deployment_agent = Agent(
    id='deployment-agent',
    model={'provider': 'openai', 'model': 'gpt-4o'},
    tools=['deploy', 'rollback', 'get_deployment_status']
)

monitoring_agent = Agent(
    id='monitoring-agent',
    model={'provider': 'openai', 'model': 'gpt-4o'},
    tools=['get_metrics', 'get_logs', 'create_alert']
)

approval_agent = Agent(
    id='approval-agent',
    model={'provider': 'openai', 'model': 'gpt-4o'},
    tools=['request_approval', 'check_approval_status']
)

# Compose agents for complex workflows
async def deploy_with_approval(environment: str, version: str):
    # Agent 1: Get approval
    approval = await approval_agent.run({
        'message': f'Request approval to deploy {version} to {environment}'
    })

    if 'approved' in approval.lower():
        # Agent 2: Execute deployment
        result = await deployment_agent.run({
            'message': f'Deploy {version} to {environment}'
        })

        # Agent 3: Monitor deployment
        await monitoring_agent.run({
            'message': f'Monitor deployment {result} for 5 minutes'
        })
```

**Benefits:**
- ✓ Each agent has focused responsibility
- ✓ Easier to test and maintain
- ✓ Agents can be developed independently
- ✓ Composition enables complex workflows

**See:** [Getting Started - Basic Usage](getting-started.md#basic-usage)

---

## Factor 11: Trigger from Anywhere

**Principle:** Enable flexible initiation of AI tasks across platforms and interfaces.

**How Vel Implements This:**

Vel provides both SDK and REST API interfaces:

```python
# SDK: Direct Python integration
from agents import Agent

agent = Agent(
    id='my-agent',
    model={'provider': 'openai', 'model': 'gpt-4o'}
)

# Trigger from Python
answer = await agent.run({'message': 'Hello'})

# Trigger from async function
async def handle_webhook(data: dict):
    result = await agent.run({'message': data['text']})
    return result

# Trigger from FastAPI endpoint
from fastapi import FastAPI
app = FastAPI()

@app.post("/chat")
async def chat(message: str):
    return await agent.run({'message': message})

# Trigger from background task
from celery import Celery
celery = Celery('tasks')

@celery.task
async def process_message(message: str):
    return await agent.run({'message': message})
```

**REST API:**

```bash
# Trigger from anywhere via HTTP
curl -X POST http://localhost:8000/runs \
  -H "Content-Type: application/json" \
  -d '{
    "agent_id": "my-agent",
    "provider": "openai",
    "model": "gpt-4o",
    "input": {"message": "hello"}
  }'

# Trigger from Slack, webhooks, cron jobs, etc.
```

**Benefits:**
- ✓ Python SDK for direct integration
- ✓ REST API for cross-platform access
- ✓ Streaming and non-streaming endpoints
- ✓ Easy integration with existing systems

**See:** [Getting Started - REST API](getting-started.md#rest-api)

---

## Factor 12: Make Agent a Stateless Reducer

**Principle:** Design agents as stateless reducers for predictable, reproducible behavior.

**How Vel Implements This:**

Vel implements the reducer pattern for non-streaming execution:

```python
# The reducer is a pure function: (State, Event) -> (State, Effects)
from agents.core import State, reduce

# Initial state
state = State(run_id='run-123')

# Event sequence
events = [
    {'kind': 'start'},
    {'kind': 'llm_step', 'step': {'tool': 'get_weather', 'args': {'city': 'NYC'}}},
    {'kind': 'tool_result', 'result': {'temp_f': 65, 'condition': 'cloudy'}},
    {'kind': 'llm_step', 'step': {'done': True, 'answer': 'The weather is cloudy, 65°F'}},
]

# Deterministic state transitions
for event in events:
    state, effects = reduce(state, event)

    # State is immutable, effects are generated
    for effect in effects:
        if effect.kind == 'call_llm':
            # Execute LLM call
            pass
        elif effect.kind == 'call_tool':
            # Execute tool
            pass
        elif effect.kind == 'halt':
            # Agent completed
            break

# Same inputs always produce same outputs
# Easy to test, debug, and reason about
```

**Stateless Execution:**

```python
# Agent execution is stateless with external state storage
agent = Agent(
    id='my-agent',
    model={'provider': 'openai', 'model': 'gpt-4o'},
    session_storage='database'  # State stored externally
)

# Each call is independent
# State loaded from database
# New state saved after completion
# Agent itself is stateless
```

**Benefits:**
- ✓ Predictable behavior: same input → same output
- ✓ Easy to test with pure functions
- ✓ Reproducible debugging
- ✓ State stored externally (database)
- ✓ Horizontally scalable (agents are stateless)

**See:** [Architecture - Reducer](../README.md#architecture)

---

## Summary: Why Vel Embodies 12-Factor Principles

| Factor | How Vel Implements It |
|--------|----------------------|
| 1. Natural Language to Tool Calls | JSON schema-validated tools with provider-agnostic format |
| 2. Own Your Prompts | Custom context managers, direct message access |
| 3. Own Your Context Window | Configurable context managers (full/stateless/limited) |
| 4. Tools are Structured Outputs | ToolSpec with input/output schemas, validation |
| 5. Unify Execution & Business State | Database persistence, tool context, event storage |
| 6. Launch/Pause/Resume | Streaming mode, database sessions, async cancellation |
| 7. Contact Humans | Tools as async functions, long-running approval flows |
| 8. Own Your Control Flow | Policies, reducer pattern, explicit state transitions |
| 9. Compact Errors | Structured error events, custom context managers |
| 10. Small, Focused Agents | Single-purpose agents, composition patterns |
| 11. Trigger from Anywhere | Python SDK + REST API, streaming/non-streaming |
| 12. Stateless Reducer | Pure reducer function, external state storage |

## Key Takeaways

**Vel is designed for production:**
- ✓ No hidden abstractions
- ✓ Full control over prompts, context, and control flow
- ✓ Explicit state management
- ✓ Provider-agnostic architecture
- ✓ Composable, modular design

**Vel gives you the primitives, not the framework:**
- You own your prompts
- You own your context window
- You own your control flow
- You compose small agents into larger systems

**Result:** Reliable, maintainable, production-ready AI agents.

## Learn More

- [12-Factor Agents Project](https://github.com/humanlayer/12-factor-agents) - Original methodology by Dex and contributors
- [HumanLayer](https://www.humanlayer.dev/) - Human-in-the-loop AI tools
- [Getting Started with Vel](getting-started.md)
- [Complete API Reference](api-reference.md)

---

## Attribution

The **12-Factor Agents** methodology referenced in this document was created by [Dex](https://github.com/humanlayer) and contributors from the SF MLOps community. The original content is available at [github.com/humanlayer/12-factor-agents](https://github.com/humanlayer/12-factor-agents).

**Licenses:**
- 12-Factor Agents Content: [Creative Commons BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/)
- 12-Factor Agents Code: [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0)
- This Document (Vel Implementation): MIT License

This document describes Vel's implementation of the 12-Factor Agent principles. It is not an official part of the 12-Factor Agents project, but rather our interpretation and implementation of those principles in the Vel runtime.
