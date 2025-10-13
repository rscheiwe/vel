---
layout: default
title: "Factor 10: Small, Focused Agents"
parent: 12-Factor Alignment
nav_order: 10
---

# Factor 10: Small, Focused Agents

**Principle:** Create specialized agents rather than monolithic ones for better reliability.

## How Vel Implements This

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

## Benefits

- ✓ Each agent has focused responsibility
- ✓ Easier to test and maintain
- ✓ Agents can be developed independently
- ✓ Composition enables complex workflows

**See:** [Getting Started - Basic Usage](../getting-started#basic-usage)
