---
layout: default
title: "Factor 1: Natural Language to Tool Calls"
parent: 12-Factor Alignment
nav_order: 1
---

# Factor 1: Natural Language to Tool Calls

**Principle:** Translate natural language into structured tool calls for reliable execution.

## How Vel Implements This

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

## Benefits

- ✓ JSON schema validation ensures type safety
- ✓ Provider-agnostic tool format works across OpenAI, Gemini, and Claude
- ✓ Clear separation between LLM decision-making and code execution

**See:** [Tools Documentation](../tools)
