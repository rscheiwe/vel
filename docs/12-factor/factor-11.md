---
layout: default
title: "Factor 11: Trigger from Anywhere"
parent: 12-Factor Alignment
nav_order: 11
---

# Factor 11: Trigger from Anywhere

**Principle:** Enable flexible initiation of AI tasks across platforms and interfaces.

## How Vel Implements This

Vel provides both SDK and REST API interfaces:

### Python SDK

```python
from vel import Agent

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

### REST API

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

## Benefits

- ✓ Python SDK for direct integration
- ✓ REST API for cross-platform access
- ✓ Streaming and non-streaming endpoints
- ✓ Easy integration with existing systems

**See:** [Getting Started - REST API](../getting-started#rest-api)
