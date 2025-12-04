---
layout: default
title: Structured Output
nav_order: 8
---

# Structured Output

Force agents to return validated, typed responses using Pydantic models.

## Overview

When you need structured data instead of free-form text, use `output_type` to:

- Force JSON mode on the LLM
- Validate output against a Pydantic schema
- Auto-retry on validation failure
- Return native Python objects

## Quick Start

```python
from pydantic import BaseModel
from vel import Agent

# Define your output schema
class WeatherResponse(BaseModel):
    city: str
    temperature: float
    conditions: str
    humidity: int

# Create agent with output_type
agent = Agent(
    id='weather-agent',
    model={'provider': 'openai', 'model': 'gpt-4o'},
    tools=['get_weather'],
    output_type=WeatherResponse  # Returns WeatherResponse, not str
)

# Result is a validated Pydantic model
result = await agent.run({'message': 'Weather in Tokyo?'})
print(result.city)        # "Tokyo"
print(result.temperature) # 72.5
print(type(result))       # <class 'WeatherResponse'>
```

## Pydantic Models

Use standard Pydantic models:

```python
from pydantic import BaseModel, Field
from typing import List, Optional
from enum import Enum

class Priority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

class Task(BaseModel):
    title: str = Field(description="Task title")
    priority: Priority
    due_date: Optional[str] = None
    tags: List[str] = []

class TaskList(BaseModel):
    tasks: List[Task]
    total: int

agent = Agent(
    id='task-agent',
    model={'provider': 'openai', 'model': 'gpt-4o'},
    output_type=TaskList
)

result = await agent.run({'message': 'Create 3 tasks for launching a product'})
for task in result.tasks:
    print(f"{task.priority.value}: {task.title}")
```

## Validation & Retry Policy

Configure how validation failures are handled:

```python
from vel.core import StructuredOutputPolicy

agent = Agent(
    id='my-agent',
    model={'provider': 'openai', 'model': 'gpt-4o'},
    output_type=MySchema,
    structured_output_policy=StructuredOutputPolicy(
        max_retries=2,           # Retry up to 2 times on validation failure
        on_failure="raise"       # What to do when all retries fail
    )
)
```

### Policy Options

**max_retries** (default: 1)
- Number of times to retry when validation fails
- Set to 0 for no retries (fail immediately)

**on_failure** (default: "raise")
- `"raise"`: Raise `StructuredOutputValidationError`
- `"return_raw"`: Return the raw string output
- `"return_last_valid"`: Return last valid output if any, else raise

### Example Policies

```python
# Strict: No retries, always raise on failure
StructuredOutputPolicy(max_retries=0, on_failure="raise")

# Lenient: Multiple retries, fall back to raw output
StructuredOutputPolicy(max_retries=3, on_failure="return_raw")

# Balanced (default): One retry, then raise
StructuredOutputPolicy(max_retries=1, on_failure="raise")
```

## How It Works

1. **Schema injection**: Vel adds a system message with your Pydantic schema
2. **LLM generates**: Model outputs JSON matching the schema
3. **Validation**: Vel parses and validates against Pydantic
4. **Retry on failure**: If invalid, error is added to prompt and LLM tries again
5. **Return typed object**: Success returns Pydantic model instance

```python
# What the LLM sees (automatically added):
"""
You must respond with valid JSON that matches this schema:
{
  "title": "WeatherResponse",
  "type": "object",
  "properties": {
    "city": {"type": "string"},
    "temperature": {"type": "number"},
    "conditions": {"type": "string"}
  },
  "required": ["city", "temperature", "conditions"]
}

Do not include any text before or after the JSON.
"""
```

## Error Handling

```python
from vel.core import StructuredOutputValidationError

try:
    result = await agent.run({'message': 'Get weather'})
except StructuredOutputValidationError as e:
    print(f"Validation failed: {e.validation_error}")
    print(f"Raw output: {e.raw_output}")
    print(f"Expected type: {e.output_type}")
```

## Streaming with Structured Output

Vel supports **progressive structured output streaming** - you get validated data as it streams, not just at the end.

### Array Streaming (List[X])

When `output_type` is a `List[Model]`, Vel emits `data-object-element` events as each array item completes:

```python
from typing import List
from pydantic import BaseModel

class AIAgent(BaseModel):
    name: str
    description: str
    use_case: str

agent = Agent(
    id='agent-generator',
    model={'provider': 'openai', 'model': 'gpt-4o'},
    output_type=List[AIAgent]  # Array mode - streams elements one-by-one
)

async for event in agent.run_stream({'message': 'Generate 5 AI agent ideas'}):
    if event['type'] == 'text-delta':
        # Raw JSON tokens (for debugging or custom parsing)
        print(event['delta'], end='')

    elif event['type'] == 'data-object-element':
        # Validated array element - update UI immediately!
        agent_data = event['data']['element']
        index = event['data']['index']
        print(f"Agent {index}: {agent_data['name']}")

    elif event['type'] == 'data-object-complete':
        # Final validated array
        all_agents = event['data']['object']
        print(f"Total: {len(all_agents)} agents")
```

### Object Streaming (Single Model)

When `output_type` is a single Pydantic model, Vel emits `data-object-partial` events as fields complete:

```python
class WeatherResponse(BaseModel):
    city: str
    temperature: float
    conditions: str
    humidity: int

agent = Agent(
    id='weather-agent',
    model={'provider': 'openai', 'model': 'gpt-4o'},
    output_type=WeatherResponse  # Object mode - streams partial updates
)

async for event in agent.run_stream({'message': 'Weather in Tokyo?'}):
    if event['type'] == 'data-object-partial':
        # Partial object with fields parsed so far
        partial = event['data']['partial']
        if 'city' in partial:
            print(f"City: {partial['city']}")
        if 'temperature' in partial:
            print(f"Temp: {partial['temperature']}")

    elif event['type'] == 'data-object-complete':
        # Final validated object
        weather = event['data']['object']
        print(f"Complete: {weather}")
```

### Event Types

| Event | When | Data |
|-------|------|------|
| `text-delta` | Every token | `{delta: "..."}` - raw JSON text |
| `data-object-element` | Array item complete | `{index: N, element: {...}}` - validated item |
| `data-object-partial` | Object field complete | `{partial: {...}}` - partial object (unvalidated) |
| `data-object-complete` | Stream finished | `{object: ..., mode: "array"\|"object"}` - final validated output |

### Frontend Integration (useChat)

These events work with Vercel AI SDK's `useChat` hook via the `onData` handler:

```javascript
const { messages, sendMessage } = useChat({
  api: '/api/chat',
  onData: (data) => {
    if (data.type === 'data-object-element') {
      // Progressive array element
      setItems(prev => [...prev, data.data.element]);
    }
    if (data.type === 'data-object-complete') {
      // Final validated data
      setFinalResult(data.data.object);
    }
  }
});
```

## Complex Schemas

### Nested Models

```python
class Address(BaseModel):
    street: str
    city: str
    country: str

class Person(BaseModel):
    name: str
    age: int
    address: Address

agent = Agent(
    output_type=Person,
    ...
)
```

### Optional Fields

```python
class SearchResult(BaseModel):
    query: str
    results: List[str]
    next_page: Optional[str] = None
    error: Optional[str] = None
```

### Constrained Types

```python
from pydantic import Field

class Review(BaseModel):
    rating: int = Field(ge=1, le=5, description="Rating from 1-5")
    text: str = Field(min_length=10, max_length=1000)
    verified: bool = False
```

## Combining with Tools

Structured output works alongside tools:

```python
class AnalysisResult(BaseModel):
    summary: str
    sentiment: str
    key_points: List[str]
    confidence: float

agent = Agent(
    id='analysis-agent',
    model={'provider': 'openai', 'model': 'gpt-4o'},
    tools=['search_docs', 'fetch_data'],
    output_type=AnalysisResult
)

# Agent uses tools, then formats final answer as AnalysisResult
result = await agent.run({'message': 'Analyze customer feedback'})
```

## Combining with Guardrails

Output guardrails run before structured output validation:

```python
async def check_confidence(content, ctx):
    """Ensure confidence is reasonable"""
    import json
    data = json.loads(content)
    if data.get('confidence', 0) < 0.5:
        return GuardrailResult(
            passed=False,
            message="Confidence too low"
        )
    return GuardrailResult(passed=True)

agent = Agent(
    output_type=AnalysisResult,
    output_guardrails=[check_confidence]
)
```

## Best Practices

1. **Use descriptive Field descriptions** - Helps LLM understand expected content
2. **Provide examples in system prompt** - Show expected output format
3. **Keep schemas focused** - Don't try to capture everything in one model
4. **Use Optional for uncertain fields** - Let LLM omit fields it's unsure about
5. **Test with edge cases** - Ensure schema handles unexpected inputs

## Troubleshooting

### LLM returns text instead of JSON

- Increase `max_retries`
- Use a more capable model
- Add explicit instructions in your prompt

### Validation always fails

- Check if schema is too strict
- Verify Pydantic constraints are achievable
- Look at `e.raw_output` to see what LLM generated

### Performance is slow

- Reduce `max_retries`
- Simplify schema
- Use `on_failure="return_raw"` for non-critical use cases

## See Also

- [Guardrails](guardrails.md) - Additional validation layers
- [Tools](tools.md) - Combine with tool calls
- [API Reference](api-reference.md) - Complete API documentation
