---
layout: default
title: Extended Thinking
nav_order: 11
---

# Extended Thinking

Extended Thinking enables standard LLM models to perform deep, multi-pass reasoning through a Reflection Controller pattern. Instead of generating a single response, the model analyzes the question, critiques its own reasoning, refines its analysis, and then produces a final answer.

## Overview

Extended Thinking implements the **Reflection Pattern**:

```
Question → Analyze → Critique → Refine (loop) → Conclude → Answer
```

This approach helps with:
- **Complex questions** requiring careful analysis
- **Nuanced topics** where initial responses may miss important considerations
- **Decision-making** that benefits from exploring multiple perspectives
- **Research tasks** where thoroughness matters

## Quick Start

```python
from vel import Agent
from vel.thinking import ThinkingConfig

# Create agent with extended thinking
agent = Agent(
    id='deep-thinker',
    model={'provider': 'openai', 'model': 'gpt-4o'},
    thinking=ThinkingConfig(mode='reflection')
)

# Stream with thinking
async for event in agent.run_stream({'message': 'What causes inflation?'}):
    if event['type'] == 'reasoning-delta':
        print(f"[Thinking] {event['delta']}", end='')
    elif event['type'] == 'text-delta':
        print(f"[Answer] {event['delta']}", end='')
```

## Configuration

### ThinkingConfig Options

```python
from vel.thinking import ThinkingConfig

config = ThinkingConfig(
    mode='reflection',              # 'reflection' or 'none'

    # Display controls
    show_analysis=True,             # Show analysis in reasoning events
    show_critiques=True,            # Show critiques in reasoning events
    show_refinements=True,          # Show refinements in reasoning events
    stream_thinking=True,           # Stream reasoning tokens live

    # Adaptive iteration
    max_refinements=3,              # Max refine iterations (1-5)
    confidence_threshold=0.8,       # Exit early if confidence >= this (0-1)
    thinking_timeout=120.0,         # Max seconds for thinking process

    # Tool support
    thinking_tools=True,            # Allow tools during thinking
    max_tool_rounds_per_phase=3,    # Max tool calls per phase

    # Model override
    thinking_model=None             # Use different model for thinking
)
```

### Instance-Level vs Runtime

You can configure thinking at the agent level or per-request:

```python
# Instance-level (all requests use thinking)
agent = Agent(
    id='always-thinks',
    model={'provider': 'openai', 'model': 'gpt-4o'},
    thinking=ThinkingConfig(mode='reflection')
)

# Runtime override (thinking for specific requests)
agent = Agent(id='flexible', model={'provider': 'openai', 'model': 'gpt-4o'})

# Simple question - no thinking
await agent.run_stream({'message': 'What is 2+2?'})

# Complex question - enable thinking
await agent.run_stream(
    {'message': 'Analyze the economic implications of AI automation'},
    thinking=ThinkingConfig(mode='reflection')
)
```

## The Reflection Flow

### Phase 1: Analyze

The model breaks down the question:
- Identifies what is being asked
- Notes key components and relationships
- Considers relevant context
- Forms an initial hypothesis

### Phase 2: Critique

The model reviews its analysis for weaknesses:
- Identifies logical flaws or gaps
- Notes unsupported assumptions
- Considers alternative interpretations
- Highlights potential errors

### Phase 3: Refine (Adaptive Loop)

The model addresses critiques:
- Systematically addresses each critique
- Fills gaps in reasoning
- Strengthens weak points
- Assesses confidence (0-100%)

If confidence is below the threshold, it loops back to critique and refine again.

### Phase 4: Conclude

The model synthesizes a final answer:
- Integrates refined reasoning
- Provides clear, direct response
- No tools used (pure synthesis)

## Stream Events

Extended Thinking emits these events:

| Event | Description | When |
|-------|-------------|------|
| `reasoning-start` | Reasoning block begins | Start of thinking |
| `data-thinking-stage` | Phase transition | Each phase change |
| `reasoning-delta` | Reasoning content | During each phase |
| `tool-input-available` | Tool call detected | If tools used |
| `tool-output-available` | Tool result | After tool execution |
| `reasoning-end` | Reasoning block ends | Before final answer |
| `text-start` | Answer begins | Start of final answer |
| `text-delta` | Answer content | Streaming answer |
| `text-end` | Answer ends | End of final answer |
| `data-thinking-complete` | Metadata | End of thinking |

### Event Sequence Example

```
start
reasoning-start        {id: "reasoning_abc"}
data-thinking-stage    {stage: "analyzing", step: 1}
reasoning-delta        {delta: "Breaking down the question..."}
reasoning-delta        {delta: "Key components are..."}
data-thinking-stage    {stage: "critiquing", step: 2}
reasoning-delta        {delta: "The analysis assumes..."}
data-thinking-stage    {stage: "refining", step: 3, iteration: 1, confidence: 0.7}
reasoning-delta        {delta: "Addressing the assumption..."}
data-thinking-stage    {stage: "refining", step: 4, iteration: 2, confidence: 0.85}
reasoning-delta        {delta: "Further strengthening..."}
data-thinking-stage    {stage: "concluding", step: 5}
reasoning-end          {id: "reasoning_abc"}
text-start             {id: "text_xyz"}
text-delta             {delta: "Based on my analysis..."}
text-end               {id: "text_xyz"}
data-thinking-complete {steps: 5, iterations: 2, final_confidence: 0.85}
finish
```

## Using Tools During Thinking

Enable tool usage during thinking phases:

```python
from vel import Agent, ToolSpec
from vel.thinking import ThinkingConfig

def search_web(query: str) -> dict:
    """Search the web for information."""
    # Implementation...
    return {'results': [...]}

agent = Agent(
    id='research-agent',
    model={'provider': 'openai', 'model': 'gpt-4o'},
    tools=[ToolSpec.from_function(search_web)],
    thinking=ThinkingConfig(
        mode='reflection',
        thinking_tools=True  # Enable tools during thinking
    )
)
```

Tools are available during Analyze, Critique, and Refine phases. The Conclude phase never uses tools to ensure pure synthesis.

## Cost Optimization

Use a cheaper model for thinking steps:

```python
agent = Agent(
    id='cost-optimized',
    model={'provider': 'openai', 'model': 'gpt-4o'},  # Final answer
    thinking=ThinkingConfig(
        mode='reflection',
        thinking_model={
            'provider': 'openai',
            'model': 'gpt-4o-mini'  # Thinking (cheaper)
        }
    )
)
```

This reduces costs since thinking phases typically generate more tokens than the final answer.

## Confidence and Iteration

The model assesses its confidence after each refinement:

```python
config = ThinkingConfig(
    mode='reflection',
    confidence_threshold=0.9,  # Require 90% confidence
    max_refinements=4          # Allow up to 4 refinement iterations
)
```

- **Early exit**: If confidence >= threshold after first refinement, skip remaining iterations
- **Max cap**: Never exceed max_refinements to control costs
- **Extraction**: Confidence parsed from LLM response (e.g., "Confidence: 85%")

## Message Storage

Thinking results are stored as multi-part messages:

```python
{
    'role': 'assistant',
    'content': [
        {'type': 'reasoning', 'text': '[Analysis]\n...\n[Critique]\n...'},
        {'type': 'text', 'text': 'The final answer is...'}
    ],
    'thinking_metadata': {
        'steps': 5,
        'iterations': 2,
        'final_confidence': 0.85
    }
}
```

This preserves the full reasoning trace for:
- Debugging and auditing
- Showing reasoning to users (expandable UI)
- Context for follow-up questions

## UI Integration

### Basic Event Handling

```javascript
// React/Next.js with Vercel AI SDK
const { messages, input, handleSubmit } = useChat({
  onData: (dataPart) => {
    if (dataPart.type === 'data-thinking-stage') {
      setThinkingStage(dataPart.data.stage);
      setConfidence(dataPart.data.confidence);
    }
  }
});
```

### Progress Indicator

```tsx
function ThinkingProgress({ stage, confidence }) {
  const icons = {
    analyzing: '🔍',
    critiquing: '🤔',
    refining: '✨',
    concluding: '📝'
  };

  return (
    <div className="thinking-progress">
      <span className="icon">{icons[stage]}</span>
      <span className="label">{stage}...</span>
      {confidence && (
        <span className="confidence">{Math.round(confidence * 100)}%</span>
      )}
    </div>
  );
}
```

### Collapsible Reasoning

```tsx
function ReasoningBlock({ reasoning, isExpanded, onToggle }) {
  return (
    <div className="reasoning-block">
      <button onClick={onToggle}>
        {isExpanded ? '▼' : '▶'} Show thinking process
      </button>
      {isExpanded && (
        <pre className="reasoning-content">{reasoning}</pre>
      )}
    </div>
  );
}
```

## Error Handling

Extended Thinking includes graceful degradation:

- **Timeout**: If thinking exceeds `thinking_timeout`, provides best available answer
- **Phase failure**: If any phase fails, uses best content from previous phases
- **Confidence fallback**: If confidence can't be extracted, defaults to 60%

```python
# Handle potential issues
async for event in agent.run_stream({'message': question}):
    if event['type'] == 'error':
        print(f"Error: {event['errorText']}")
    elif event['type'] == 'data-thinking-complete':
        if event['data'].get('timeout'):
            print("Thinking timed out, answer may be incomplete")
```

## Best Practices

### When to Use Extended Thinking

✅ **Good for:**
- Complex analytical questions
- Decision-making scenarios
- Research and investigation
- Multi-faceted problems
- Questions requiring nuance

❌ **Not ideal for:**
- Simple factual queries
- Quick lookups
- Real-time chat (latency sensitive)
- Cost-sensitive high-volume applications

### Configuration Tips

1. **Start with defaults**: `ThinkingConfig(mode='reflection')` works well for most cases
2. **Tune confidence**: Lower threshold (0.7) for faster responses, higher (0.9) for thoroughness
3. **Limit refinements**: 2-3 refinements usually sufficient; more rarely improves quality
4. **Use model override**: Use `gpt-4o-mini` for thinking, `gpt-4o` for final answer to save costs
5. **Disable tools**: Set `thinking_tools=False` if tools aren't needed during analysis

## Examples

See `examples/extended_thinking.py` for complete examples:

1. Basic Extended Thinking
2. Runtime Override
3. Cost-Optimized (Different Models)
4. Thinking With Tools
5. High Confidence Threshold
6. Silent Thinking
7. Progress Tracking UI

## API Reference

### ThinkingConfig

```python
class ThinkingConfig:
    mode: Literal['reflection', 'none'] = 'none'
    show_analysis: bool = True
    show_critiques: bool = True
    show_refinements: bool = True
    stream_thinking: bool = True
    max_refinements: int = 3
    confidence_threshold: float = 0.8
    thinking_timeout: float = 120.0
    thinking_tools: bool = True
    max_tool_rounds_per_phase: int = 3
    thinking_model: Optional[Dict[str, Any]] = None

    def to_dict() -> Dict[str, Any]: ...
```

### Agent Integration

```python
class Agent:
    def __init__(
        self,
        # ... existing params ...
        thinking: Optional[ThinkingConfig] = None
    ): ...

    async def run_stream(
        self,
        input: Dict[str, Any],
        session_id: Optional[str] = None,
        # ... existing params ...
        thinking: Optional[ThinkingConfig] = None  # Runtime override
    ) -> AsyncGenerator[Dict[str, Any], None]: ...
```

### Events

```python
# Stage progress (transient)
{
    'type': 'data-thinking-stage',
    'data': {
        'stage': 'refining',
        'step': 3,
        'iteration': 1,
        'confidence': 0.7
    },
    'transient': True
}

# Completion metadata (persistent)
{
    'type': 'data-thinking-complete',
    'data': {
        'steps': 5,
        'iterations': 2,
        'final_confidence': 0.85,
        'thinking_tokens': 2450,
        'thinking_model': 'gpt-4o-mini'
    },
    'transient': False
}
```
