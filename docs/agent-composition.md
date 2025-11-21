---
layout: default
title: Agent Composition
nav_order: 9
---

# Agent Composition

Build multi-agent systems using agent-as-tool and handoff patterns.

## Overview

Vel supports composing multiple agents together:

- **Agent-as-Tool**: Expose an agent as a tool for another agent to call
- **Handoffs**: Transfer control from one agent to another (via custom handler)

## Agent-as-Tool

Convert any agent into a tool that other agents can call.

### Basic Usage

```python
from vel import Agent, register_tool

# Create specialist agent
math_agent = Agent(
    id='math-solver',
    model={'provider': 'openai', 'model': 'gpt-4o'},
    tools=['calculator']
)

# Expose as tool
math_tool = math_agent.as_tool(
    name='solve_math',
    description='Solve complex math problems step by step'
)

# Register and use in orchestrator
register_tool(math_tool)

orchestrator = Agent(
    id='orchestrator',
    model={'provider': 'openai', 'model': 'gpt-4o'},
    tools=['solve_math', 'search_docs', 'send_email']
)

# Orchestrator can now delegate math to specialist
result = await orchestrator.run({
    'message': 'Calculate compound interest for $1000 at 5% for 3 years'
})
```

### as_tool() Parameters

```python
agent.as_tool(
    name='tool_name',           # Optional, defaults to agent.id
    description='Tool desc',     # Optional, defaults to "Run the {agent.id} agent"
    system_prompt_override=None  # Optional, override agent's default prompt
)
```

### Example: Research Team

```python
# Specialist agents
researcher = Agent(
    id='researcher',
    model={'provider': 'openai', 'model': 'gpt-4o'},
    tools=['web_search', 'read_paper']
)

writer = Agent(
    id='writer',
    model={'provider': 'openai', 'model': 'gpt-4o'},
)

fact_checker = Agent(
    id='fact-checker',
    model={'provider': 'openai', 'model': 'gpt-4o'},
    tools=['verify_claim']
)

# Expose as tools
register_tool(researcher.as_tool(
    name='research',
    description='Research a topic using web search and academic papers'
))
register_tool(writer.as_tool(
    name='write_content',
    description='Write polished content based on research'
))
register_tool(fact_checker.as_tool(
    name='fact_check',
    description='Verify facts and claims for accuracy'
))

# Lead agent orchestrates the team
lead = Agent(
    id='lead',
    model={'provider': 'openai', 'model': 'gpt-4o'},
    tools=['research', 'write_content', 'fact_check']
)

result = await lead.run({
    'message': 'Write a fact-checked article about quantum computing'
})
```

## Handoffs via Custom Handler

For more control over agent-to-agent transfers, use custom tool handlers:

```python
from vel.core import ToolUseBehavior, ToolUseDecision, ToolUseDirective

# Specialist agents
refund_agent = Agent(id='refund-agent', ...)
support_agent = Agent(id='support-agent', ...)
sales_agent = Agent(id='sales-agent', ...)

def routing_handler(event):
    """Route to specialist based on intent classification"""

    if event.tool_name == 'classify_intent':
        intent = event.output.get('intent')

        if intent == 'refund':
            return ToolUseDirective(
                decision=ToolUseDecision.CONTINUE,
                handoff_agent=refund_agent
            )
        elif intent == 'sales':
            return ToolUseDirective(
                decision=ToolUseDecision.CONTINUE,
                handoff_agent=sales_agent
            )

    return ToolUseDecision.CONTINUE

router = Agent(
    id='router',
    model={'provider': 'openai', 'model': 'gpt-4o'},
    tools=['classify_intent'],
    policies={
        'tool_use_behavior': ToolUseBehavior.CUSTOM_HANDLER,
        'custom_tool_handler': routing_handler
    }
)
```

### HandoffConfig

Configure how context is shared during handoffs:

```python
from vel.core import HandoffConfig

# Share full context (default)
handoff = HandoffConfig(
    target_agent=refund_agent,
    share_context=True
)

# Fresh context (no history)
handoff = HandoffConfig(
    target_agent=refund_agent,
    share_context=False
)

# Filtered context (last 5 messages only)
handoff = HandoffConfig(
    target_agent=refund_agent,
    share_context=True,
    input_filter=lambda msgs: msgs[-5:]
)

# Custom filter (remove system messages)
handoff = HandoffConfig(
    target_agent=secure_agent,
    share_context=True,
    input_filter=lambda msgs: [m for m in msgs if m['role'] != 'system']
)
```

## Patterns

### 1. Hub and Spoke

Central orchestrator delegates to specialists:

```python
# Specialists
code_agent = Agent(id='coder', tools=['write_code', 'run_tests'])
review_agent = Agent(id='reviewer', tools=['analyze_code'])
docs_agent = Agent(id='documenter', tools=['generate_docs'])

# Hub
orchestrator = Agent(
    id='orchestrator',
    tools=[
        code_agent.as_tool('write_code'),
        review_agent.as_tool('review_code'),
        docs_agent.as_tool('write_docs')
    ]
)
```

### 2. Pipeline

Sequential processing through agents:

```python
def pipeline_handler(event):
    if event.tool_name == 'extract_data':
        # Pass to transformer
        return ToolUseDirective(
            decision=ToolUseDecision.CONTINUE,
            handoff_agent=transformer_agent
        )
    elif event.tool_name == 'transform_data':
        # Pass to loader
        return ToolUseDirective(
            decision=ToolUseDecision.CONTINUE,
            handoff_agent=loader_agent
        )
    return ToolUseDecision.CONTINUE
```

### 3. Hierarchical

Multi-level delegation:

```python
# Level 3: Specialists
junior_coder = Agent(id='junior', tools=['write_simple_code'])
senior_coder = Agent(id='senior', tools=['write_complex_code'])

# Level 2: Team leads
code_lead = Agent(
    id='code-lead',
    tools=[
        junior_coder.as_tool('assign_junior'),
        senior_coder.as_tool('assign_senior')
    ]
)

# Level 1: Director
director = Agent(
    id='director',
    tools=[code_lead.as_tool('delegate_coding')]
)
```

## Best Practices

1. **Clear agent responsibilities** - Each agent should have a focused purpose
2. **Consistent interfaces** - Use similar input/output schemas across agents
3. **Avoid deep nesting** - Keep agent hierarchies shallow (2-3 levels max)
4. **Handle failures gracefully** - Sub-agents can fail; parent should handle
5. **Monitor costs** - Each agent call uses tokens; composition multiplies usage

## Limitations

- **No automatic state sharing** - You must explicitly configure context sharing
- **Sequential execution** - Sub-agents run one at a time (no parallel)
- **Token accumulation** - Nested agents accumulate context, watch for limits

## See Also

- [Tools](tools.md) - Tool system basics
- [Tool Use Behavior](tools.md#tool-use-behavior) - Custom handlers
- [Sessions](sessions.md) - Session management
