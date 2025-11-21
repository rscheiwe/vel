---
layout: default
title: Guardrails
nav_order: 7
---

# Guardrails

Validation and safety layers for agent inputs, outputs, and tool calls.

## Overview

Guardrails are async functions that validate and optionally transform content at key points in agent execution:

- **Input Guardrails**: Validate user messages before LLM call
- **Output Guardrails**: Validate LLM responses before returning to user
- **Tool Guardrails**: Validate tool arguments before execution

## Quick Start

```python
from vel import Agent
from vel.core import GuardrailResult

# Define guardrail functions
async def no_pii(content, ctx):
    """Reject content containing PII patterns"""
    import re
    if re.search(r'\b\d{3}-\d{2}-\d{4}\b', content):  # SSN pattern
        return GuardrailResult(passed=False, message="Content contains SSN")
    return GuardrailResult(passed=True)

async def require_json(content, ctx):
    """Ensure output is valid JSON"""
    import json
    try:
        json.loads(content)
        return GuardrailResult(passed=True)
    except:
        return GuardrailResult(passed=False, message="Output must be valid JSON")

# Use guardrails
agent = Agent(
    id='safe-agent',
    model={'provider': 'openai', 'model': 'gpt-4o'},
    input_guardrails=[no_pii],
    output_guardrails=[require_json]
)
```

## Guardrail Function Signature

```python
async def my_guardrail(content: Any, ctx: dict) -> GuardrailResult | bool:
    """
    Args:
        content: The content to validate
            - For input: user message string or input dict
            - For output: LLM response string
            - For tool: tool arguments dict
        ctx: Context dict with run_id, session_id, etc.

    Returns:
        GuardrailResult or bool
    """
    pass
```

## GuardrailResult

```python
from vel.core import GuardrailResult

# Simple pass/fail
GuardrailResult(passed=True)
GuardrailResult(passed=False, message="Validation failed")

# With content modification
GuardrailResult(
    passed=True,
    modified_content="sanitized content"  # Replaces original
)
```

## Input Guardrails

Validate user messages before calling the LLM.

```python
async def validate_length(content, ctx):
    """Require minimum message length"""
    if len(content) < 10:
        return GuardrailResult(
            passed=False,
            message="Message too short (min 10 chars)"
        )
    return GuardrailResult(passed=True)

async def sanitize_input(content, ctx):
    """Remove potentially harmful content"""
    sanitized = content.replace('<script>', '').replace('</script>', '')
    return GuardrailResult(passed=True, modified_content=sanitized)

agent = Agent(
    id='my-agent',
    model={'provider': 'openai', 'model': 'gpt-4o'},
    input_guardrails=[validate_length, sanitize_input]
)
```

**Behavior:**
- Run sequentially before LLM call
- If any guardrail fails, agent raises `GuardrailError`
- Modified content is passed to subsequent guardrails and LLM

## Output Guardrails

Validate LLM responses before returning to user.

```python
async def no_harmful_content(content, ctx):
    """Block harmful or inappropriate responses"""
    harmful_patterns = ['confidential', 'password', 'secret key']
    for pattern in harmful_patterns:
        if pattern.lower() in content.lower():
            return GuardrailResult(
                passed=False,
                message=f"Response contains blocked term: {pattern}"
            )
    return GuardrailResult(passed=True)

async def format_response(content, ctx):
    """Ensure consistent formatting"""
    # Add disclaimer to all responses
    modified = f"{content}\n\n---\n*AI-generated response*"
    return GuardrailResult(passed=True, modified_content=modified)

agent = Agent(
    id='my-agent',
    model={'provider': 'openai', 'model': 'gpt-4o'},
    output_guardrails=[no_harmful_content, format_response]
)
```

## Tool Guardrails

Validate tool arguments before execution.

```python
async def validate_email_recipient(args, ctx):
    """Only allow emails to approved domains"""
    to = args.get('to', '')
    allowed_domains = ['company.com', 'partner.org']

    domain = to.split('@')[-1] if '@' in to else ''
    if domain not in allowed_domains:
        return GuardrailResult(
            passed=False,
            message=f"Email domain '{domain}' not in allowed list"
        )
    return GuardrailResult(passed=True)

async def validate_query_params(args, ctx):
    """Sanitize database query parameters"""
    # Prevent SQL injection
    query = args.get('query', '')
    if any(keyword in query.upper() for keyword in ['DROP', 'DELETE', 'TRUNCATE']):
        return GuardrailResult(
            passed=False,
            message="Dangerous SQL keywords detected"
        )
    return GuardrailResult(passed=True)

agent = Agent(
    id='my-agent',
    model={'provider': 'openai', 'model': 'gpt-4o'},
    tools=['send_email', 'query_database'],
    tool_guardrails={
        'send_email': [validate_email_recipient],
        'query_database': [validate_query_params]
    }
)
```

## Context Object

The `ctx` parameter provides runtime information:

```python
async def context_aware_guardrail(content, ctx):
    run_id = ctx.get('run_id')
    session_id = ctx.get('session_id')
    tool_name = ctx.get('tool_name')  # Only for tool guardrails

    # Use context for logging, conditional logic, etc.
    return GuardrailResult(passed=True)
```

## Simplified Return Types

For simple pass/fail, you can return a boolean:

```python
async def simple_check(content, ctx):
    return len(content) > 0  # True = pass, False = fail
```

Or a dict:

```python
async def dict_check(content, ctx):
    return {
        'passed': True,
        'message': 'Validation successful',
        'modified_content': content.strip()
    }
```

## Error Handling

When a guardrail fails:

### Non-Streaming (`run()`)

```python
from vel.core import GuardrailError

try:
    result = await agent.run({'message': 'test'})
except GuardrailError as e:
    print(f"Guardrail '{e.guardrail_name}' failed: {e.message}")
    print(f"Original content: {e.content}")
```

### Streaming (`run_stream()`)

```python
async for event in agent.run_stream({'message': 'test'}):
    if event['type'] == 'error':
        if 'guardrail' in event['error'].lower():
            print(f"Guardrail failed: {event['error']}")
```

## Common Patterns

### Rate Limiting

```python
from datetime import datetime, timedelta

user_requests = {}  # In production, use Redis or similar

async def rate_limit(content, ctx):
    user_id = ctx.get('user_id', 'anonymous')
    now = datetime.now()

    # Get user's request history
    history = user_requests.get(user_id, [])

    # Remove old requests (older than 1 minute)
    history = [t for t in history if now - t < timedelta(minutes=1)]

    # Check rate limit (10 requests per minute)
    if len(history) >= 10:
        return GuardrailResult(
            passed=False,
            message="Rate limit exceeded. Please wait."
        )

    # Record this request
    history.append(now)
    user_requests[user_id] = history

    return GuardrailResult(passed=True)
```

### Content Moderation

```python
async def moderate_content(content, ctx):
    """Use external moderation API"""
    import httpx

    async with httpx.AsyncClient() as client:
        response = await client.post(
            'https://api.moderationservice.com/check',
            json={'text': content}
        )
        result = response.json()

    if result.get('flagged'):
        return GuardrailResult(
            passed=False,
            message=f"Content flagged: {result.get('reason')}"
        )

    return GuardrailResult(passed=True)
```

### Schema Validation

```python
from pydantic import BaseModel, ValidationError

class ExpectedOutput(BaseModel):
    answer: str
    confidence: float

async def validate_schema(content, ctx):
    """Ensure output matches expected schema"""
    import json

    try:
        data = json.loads(content)
        ExpectedOutput(**data)
        return GuardrailResult(passed=True)
    except (json.JSONDecodeError, ValidationError) as e:
        return GuardrailResult(
            passed=False,
            message=f"Schema validation failed: {e}"
        )
```

### Logging / Audit

```python
import logging

logger = logging.getLogger('guardrails')

async def audit_log(content, ctx):
    """Log all inputs for audit purposes"""
    logger.info(
        f"Input received",
        extra={
            'run_id': ctx.get('run_id'),
            'session_id': ctx.get('session_id'),
            'content_length': len(content),
            'content_preview': content[:100]
        }
    )
    return GuardrailResult(passed=True)
```

## Best Practices

1. **Keep guardrails fast** - They run on every request
2. **Use async for I/O** - External API calls should be async
3. **Fail with clear messages** - Help users understand what went wrong
4. **Order matters** - Guardrails run sequentially; put fast checks first
5. **Don't over-validate** - Balance safety with user experience

## See Also

- [Tools](tools.md) - Tool guardrails
- [Structured Output](structured-output.md) - Output validation with Pydantic
- [API Reference](api-reference.md) - Complete API documentation
