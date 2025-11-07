# Multi-Step Tools

AI SDK compatible tools for building multi-step agents in Vel. Each tool returns output with a `state` field for frontend compatibility with `useChat` and other Vercel AI SDK hooks.

## Available Tools

### 1. Web Search (`websearch`)

Searches the web for current information and resources using **Perplexity Sonar API**.

**Setup:**
1. Get API key from: https://www.perplexity.ai/settings/api
2. Set environment variable: `export PERPLEXITY_API_KEY=pplx-...`

**Input:**
```python
{
    'query': str,      # Search query
    'limit': int       # Max results (1-20, default: 5)
}
```

**Output:**
```python
{
    'state': 'ready',
    'query': str,
    'results': [
        {
            'title': str,
            'url': str,
            'snippet': str,
            'source': str
        }
    ]
}
```

**Example:**
```python
# In agent with tools=['websearch']
# Agent will call: websearch(query="AI trends 2024", limit=5)
```

**See also:** `examples/perplexity_web_search_example.py` for complete usage example.

---

### 2. News Search (`news`)

Searches Hacker News for recent headlines and developments.

**Input:**
```python
{
    'topic': str,      # Topic to search
    'limit': int       # Max items (1-20, default: 5)
}
```

**Output:**
```python
{
    'state': 'ready',
    'topic': str,
    'items': [
        {
            'id': str,
            'title': str,
            'url': str,
            'publishedAt': str  # ISO 8601 format
        }
    ]
}
```

**Example:**
```python
# In agent with tools=['news']
# Agent will call: news(topic="machine learning", limit=5)
```

---

### 3. Analyze (`analyze`)

Breaks down complex problems into manageable components using different analytical approaches.

**Input:**
```python
{
    'problem': str,           # Problem to analyze
    'approach': str           # 'systematic', 'creative', or 'technical'
}
```

**Output:**
```python
{
    'state': 'ready',
    'problem': str,
    'approach': str,
    'breakdown': str,         # Detailed breakdown
    'components': [str]       # List of components
}
```

**Approaches:**
- **systematic**: Step-by-step logical breakdown (problem identification → root cause → solutions → implementation)
- **creative**: Unconventional and innovative approaches
- **technical**: Architecture and implementation focus

**Example:**
```python
# In agent with tools=['analyze']
# Agent will call: analyze(
#     problem="build scalable microservices",
#     approach="technical"
# )
```

---

### 4. Decide (`decide`)

Makes decisions between options based on evaluation criteria with scoring.

**Input:**
```python
{
    'options': [str],         # List of options (min 2)
    'criteria': [str],        # Evaluation criteria (min 1)
    'context': str            # Context for decision
}
```

**Output:**
```python
{
    'state': 'ready',
    'context': str,
    'options': [str],
    'criteria': [str],
    'evaluation': [
        {
            'option': str,
            'score': int,         # 1-10
            'reasoning': str
        }
    ],
    'decision': str,              # Best option
    'reasoning': str              # Why it was chosen
}
```

**Example:**
```python
# In agent with tools=['decide']
# Agent will call: decide(
#     options=["React", "Vue", "Svelte"],
#     criteria=["performance", "ecosystem", "learning curve"],
#     context="Building a dashboard app"
# )
```

---

### 5. Provide Answer (`provideAnswer`)

**Terminating tool** that provides the final structured answer with citations and reasoning steps.

**Input:**
```python
{
    'answer': str,            # Final answer with citations [1], [2]
    'steps': [
        {
            'step': str,      # What was done
            'reasoning': str, # Why it was done
            'result': str     # Result of this step
        }
    ],
    'confidence': float,      # 0.0-1.0
    'sources': [str],         # Optional: Source URLs
    'citations': [            # Optional: Detailed citations
        {
            'number': str,    # Citation number
            'title': str,
            'url': str,
            'description': str,
            'snippet': str
        }
    ]
}
```

**Output:**
```python
{
    'state': 'ready',
    'answer': str,
    'steps': [...],
    'confidence': float,
    'sources': [str],
    'citations': [...],
    'summary': str            # Generated summary
}
```

**Example:**
```python
# In agent with tools=['provideAnswer']
# Agent will call: provideAnswer(
#     answer="The latest AI trends include...",
#     steps=[...],
#     confidence=0.9,
#     citations=[...]
# )
```

---

## Usage

### Register All Tools

```python
from multi_step_tools import (
    web_search_tool,
    news_search_tool,
    analyze_tool,
    decide_tool,
    provide_answer_tool
)

# Tools are automatically registered on import
```

### Create Multi-Step Agent

```python
from vel import Agent

agent = Agent(
    id='multi-step-agent',
    model={'provider': 'openai', 'model': 'gpt-4o'},
    tools=[
        'websearch',
        'news',
        'analyze',
        'decide',
        'provideAnswer'
    ],
    policies={'max_steps': 8},
    generation_config={
        'temperature': 0.7,
        'max_tokens': 2000
    }
)

# Run agent
async for event in agent.run_stream({'message': 'Your question'}):
    print(event)
```

### Tool Workflow Example

**Simple Question:**
```
1. websearch(query="AI trends")
2. provideAnswer(answer="...", citations=[...])
```

**Complex Question:**
```
1. analyze(problem="build ML pipeline", approach="technical")
2. websearch(query="ML pipeline best practices")
3. decide(options=[...], criteria=[...])
4. provideAnswer(answer="...", citations=[...])
```

---

## Frontend Integration

These tools work seamlessly with Vercel AI SDK's `useChat` hook:

```typescript
const { messages } = useChat({
  api: '/api/vel-agent',
});

// Messages will have parts array with tool invocations
messages[0].parts?.map((part) => {
  if (part.type === 'tool-websearch') {
    return <WebSearchCard {...part} />;
  }
  if (part.type === 'tool-provideAnswer') {
    return <FinalAnswer {...part} />;
  }
});
```

---

## Implementation Notes

### State Field

All tools return a `state` field in their output:
- `"loading"`: Tool is processing (not used in Vel's synchronous pattern)
- `"ready"`: Tool has completed and output is available

### Tool Naming Convention

Tool names match the `type` field in `useChat` parts:
- Tool name: `websearch` → Part type: `tool-websearch`
- Tool name: `provideAnswer` → Part type: `tool-provideAnswer`

### Error Handling

Tools handle errors gracefully and return valid output with `state: "ready"` even on failure:

```python
try:
    # Perform operation
    result = await fetch_data()
except Exception as e:
    # Return error result
    return {
        'state': 'ready',
        'error': str(e),
        # ... minimal valid output
    }
```

---

## Testing

Run the comprehensive example:

```bash
cd examples
python comprehensive_multi_step_agent.py
```

This demonstrates all tools working together with formatted output showing:
- Step tracking
- Tool calls and inputs
- Tool outputs and state
- Final answers with citations

---

## See Also

- [Multi-Step Agent Pattern Documentation](../../features/MULTI_STEP_AGENT_PATTERN.md)
- [Vel Agent Documentation](../../docs/api-reference.md)
- [Vercel AI SDK Multi-Step Agents](https://sdk.vercel.ai/docs/ai-sdk-core/agents)
