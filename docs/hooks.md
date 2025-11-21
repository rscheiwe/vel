---
layout: default
title: Lifecycle Hooks
nav_order: 10
---

# Lifecycle Hooks

Observe and trace agent execution with lifecycle event callbacks.

## Overview

Hooks are callback functions that fire at key points during agent execution:

- **Step events**: Start/end of each execution step
- **LLM events**: Before/after LLM calls
- **Tool events**: Before/after tool execution
- **Completion events**: Success or error

Use hooks for logging, tracing, metrics, debugging, and custom logic injection.

## Quick Start

```python
from vel import Agent

def log_tool_call(event):
    print(f"Tool called: {event.tool_name}")
    print(f"Args: {event.args}")

def log_error(event):
    print(f"Error in run {event.run_id}: {event.error_message}")

agent = Agent(
    id='my-agent',
    model={'provider': 'openai', 'model': 'gpt-4o'},
    tools=['get_weather'],
    hooks={
        'on_tool_call': log_tool_call,
        'on_error': log_error
    }
)
```

## Available Hooks

| Hook | Fires When | Event Type |
|------|------------|------------|
| `on_step_start` | Execution step begins | `StepStartHookEvent` |
| `on_step_end` | Execution step ends | `StepEndHookEvent` |
| `on_tool_call` | Before tool execution | `ToolCallHookEvent` |
| `on_tool_result` | After tool returns | `ToolResultHookEvent` |
| `on_llm_request` | Before LLM call | `LLMRequestHookEvent` |
| `on_llm_response` | After LLM responds | `LLMResponseHookEvent` |
| `on_finish` | Agent completes successfully | `FinishHookEvent` |
| `on_error` | Error occurs | `ErrorHookEvent` |

## Hook Events

All events share a base structure:

```python
@dataclass
class HookEvent:
    run_id: str
    session_id: Optional[str]
    trace_id: str
    metadata: Dict[str, Any]
```

### StepStartHookEvent

```python
@dataclass
class StepStartHookEvent(HookEvent):
    step: int  # Current step number (1-indexed)
```

### StepEndHookEvent

```python
@dataclass
class StepEndHookEvent(HookEvent):
    step: int
    duration_ms: float  # Step duration in milliseconds
```

### ToolCallHookEvent

```python
@dataclass
class ToolCallHookEvent(HookEvent):
    tool_name: str
    args: Dict[str, Any]
    step: int
```

### ToolResultHookEvent

```python
@dataclass
class ToolResultHookEvent(HookEvent):
    tool_name: str
    args: Dict[str, Any]
    result: Any
    duration_ms: float
    step: int
```

### LLMRequestHookEvent

```python
@dataclass
class LLMRequestHookEvent(HookEvent):
    messages: List[Dict[str, Any]]
    model: str
    step: int
```

### LLMResponseHookEvent

```python
@dataclass
class LLMResponseHookEvent(HookEvent):
    response: Any
    duration_ms: float
    step: int
    usage: Optional[Dict[str, int]]  # Token counts
```

### FinishHookEvent

```python
@dataclass
class FinishHookEvent(HookEvent):
    result: Any
    total_steps: int
    total_duration_ms: float
```

### ErrorHookEvent

```python
@dataclass
class ErrorHookEvent(HookEvent):
    error: Exception
    error_message: str
    step: int
```

## Async Hooks

Hooks can be async:

```python
async def async_log(event):
    await database.insert({
        'run_id': event.run_id,
        'tool': event.tool_name,
        'args': event.args,
        'timestamp': datetime.now()
    })

agent = Agent(
    hooks={'on_tool_call': async_log}
)
```

## Examples

### Logging

```python
import logging

logger = logging.getLogger('agent')

def setup_logging_hooks():
    return {
        'on_step_start': lambda e: logger.info(f"Step {e.step} started"),
        'on_step_end': lambda e: logger.info(f"Step {e.step} completed in {e.duration_ms}ms"),
        'on_tool_call': lambda e: logger.info(f"Calling {e.tool_name} with {e.args}"),
        'on_tool_result': lambda e: logger.info(f"{e.tool_name} returned in {e.duration_ms}ms"),
        'on_llm_request': lambda e: logger.debug(f"LLM request: {len(e.messages)} messages"),
        'on_llm_response': lambda e: logger.debug(f"LLM response: {e.usage}"),
        'on_finish': lambda e: logger.info(f"Completed in {e.total_steps} steps, {e.total_duration_ms}ms"),
        'on_error': lambda e: logger.error(f"Error: {e.error_message}")
    }

agent = Agent(
    hooks=setup_logging_hooks()
)
```

### Metrics Collection

```python
from prometheus_client import Counter, Histogram

tool_calls = Counter('agent_tool_calls_total', 'Tool calls', ['tool_name'])
tool_duration = Histogram('agent_tool_duration_seconds', 'Tool duration', ['tool_name'])
llm_tokens = Counter('agent_llm_tokens_total', 'LLM tokens', ['type'])

def metrics_hooks():
    return {
        'on_tool_result': lambda e: (
            tool_calls.labels(tool_name=e.tool_name).inc(),
            tool_duration.labels(tool_name=e.tool_name).observe(e.duration_ms / 1000)
        ),
        'on_llm_response': lambda e: (
            llm_tokens.labels(type='input').inc(e.usage.get('prompt_tokens', 0)),
            llm_tokens.labels(type='output').inc(e.usage.get('completion_tokens', 0))
        ) if e.usage else None
    }
```

### OpenTelemetry Tracing

```python
from opentelemetry import trace

tracer = trace.get_tracer(__name__)

def otel_hooks():
    spans = {}

    def on_step_start(event):
        span = tracer.start_span(f"step-{event.step}")
        span.set_attribute("run_id", event.run_id)
        spans[f"{event.run_id}-{event.step}"] = span

    def on_step_end(event):
        span = spans.pop(f"{event.run_id}-{event.step}", None)
        if span:
            span.set_attribute("duration_ms", event.duration_ms)
            span.end()

    def on_tool_call(event):
        span = tracer.start_span(f"tool-{event.tool_name}")
        span.set_attribute("args", str(event.args))
        spans[f"{event.run_id}-tool-{event.step}"] = span

    def on_tool_result(event):
        span = spans.pop(f"{event.run_id}-tool-{event.step}", None)
        if span:
            span.set_attribute("duration_ms", event.duration_ms)
            span.end()

    return {
        'on_step_start': on_step_start,
        'on_step_end': on_step_end,
        'on_tool_call': on_tool_call,
        'on_tool_result': on_tool_result
    }
```

### Cost Tracking

```python
# Token costs (example rates)
COSTS = {
    'gpt-4o': {'input': 0.005 / 1000, 'output': 0.015 / 1000},
    'gpt-4o-mini': {'input': 0.00015 / 1000, 'output': 0.0006 / 1000}
}

class CostTracker:
    def __init__(self):
        self.total_cost = 0

    def on_llm_response(self, event):
        if event.usage:
            model = 'gpt-4o'  # Get from event metadata
            rates = COSTS.get(model, {'input': 0, 'output': 0})
            cost = (
                event.usage.get('prompt_tokens', 0) * rates['input'] +
                event.usage.get('completion_tokens', 0) * rates['output']
            )
            self.total_cost += cost
            print(f"Run cost: ${cost:.4f} (Total: ${self.total_cost:.4f})")

tracker = CostTracker()
agent = Agent(
    hooks={'on_llm_response': tracker.on_llm_response}
)
```

### Debug Mode

```python
def debug_hooks():
    """Verbose hooks for debugging"""
    return {
        'on_step_start': lambda e: print(f"\n{'='*50}\nSTEP {e.step}\n{'='*50}"),

        'on_llm_request': lambda e: print(
            f"📤 LLM Request ({len(e.messages)} messages):\n" +
            '\n'.join(f"  [{m['role']}] {m['content'][:100]}..." for m in e.messages[-3:])
        ),

        'on_llm_response': lambda e: print(
            f"📥 LLM Response:\n  {str(e.response)[:200]}..."
        ),

        'on_tool_call': lambda e: print(
            f"🔧 Tool: {e.tool_name}\n  Args: {e.args}"
        ),

        'on_tool_result': lambda e: print(
            f"✅ Result: {str(e.result)[:200]}...\n  Duration: {e.duration_ms:.0f}ms"
        ),

        'on_error': lambda e: print(
            f"❌ ERROR: {e.error_message}"
        ),

        'on_finish': lambda e: print(
            f"\n{'='*50}\n✅ DONE ({e.total_steps} steps, {e.total_duration_ms:.0f}ms)\n{'='*50}"
        )
    }

# Use in development
agent = Agent(
    hooks=debug_hooks()
)
```

## Error Handling in Hooks

Hook errors are logged but don't fail agent execution:

```python
def buggy_hook(event):
    raise Exception("Hook error!")

agent = Agent(
    hooks={'on_tool_call': buggy_hook}
)

# Agent still runs; error is logged
result = await agent.run({'message': 'test'})
```

## Trace ID

All events in a run share a `trace_id` for correlation:

```python
def correlate_events(event):
    # Use trace_id to group all events from this run
    print(f"[{event.trace_id}] {event.run_id}")

agent = Agent(
    hooks={
        'on_step_start': correlate_events,
        'on_tool_call': correlate_events,
        'on_finish': correlate_events
    }
)
```

## Best Practices

1. **Keep hooks fast** - Slow hooks delay agent execution
2. **Use async for I/O** - Database writes, API calls should be async
3. **Handle errors gracefully** - Don't let hook errors crash your app
4. **Don't modify events** - Hooks are for observation, not mutation
5. **Use trace_id for correlation** - Link related events together

## See Also

- [API Reference](api-reference.md) - Complete event type documentation
- [Providers](providers.md) - Provider-specific usage tracking
