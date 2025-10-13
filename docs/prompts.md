---
layout: default
title: Prompt Templates
nav_order: 4
---

# Prompt Templates

Flexible prompt management system with Jinja2 templating, XML formatting, and environment-based configuration. Follows 12-Factor Agent principles and Anthropic's context engineering best practices.

## Overview

The prompt module provides:
- **Jinja2 templating** with variable interpolation
- **XML-first formatting** for clear context boundaries
- **Environment-based prompts** (dev/staging/prod)
- **Version control** for prompt evolution
- **Context formatting utilities** following Anthropic best practices
- **Seamless Agent integration** (backwards compatible, opt-in)

## Quick Start

### Basic Template

```python
from agents import Agent, PromptTemplate, register_prompt

# Define a template
template = PromptTemplate(
    id="chat-assistant:v1",
    system="""
    <system_instructions>
      <role>You are {{role_name}}, a helpful AI assistant.</role>
      <guidelines>
        - Be concise and clear
        - Provide accurate information
        - Admit when you don't know something
      </guidelines>
    </system_instructions>
    """,
    variables={"role_name": "Alex"}
)

# Register globally
register_prompt(template)

# Use with Agent
agent = Agent(
    id='chat-assistant:v1',
    model={'provider': 'anthropic', 'model': 'claude-sonnet-4'},
    prompt_id='chat-assistant:v1',
    prompt_vars={'role_name': 'Sarah'},
    prompt_env='prod'
)
```

## Core Concepts

### 1. PromptTemplate

Templates use Jinja2 for variable interpolation and support environment-based variations.

```python
from agents import PromptTemplate

# Single template for all environments
template = PromptTemplate(
    id="agent:v1",
    system="You are a {{role}} assistant with {{expertise}}.",
    variables={"role": "helpful", "expertise": "general knowledge"}
)

# Environment-specific templates
template = PromptTemplate(
    id="agent:v1",
    environments={
        "dev": """
        <system>Debug mode enabled. Verbose output.</system>
        """,
        "prod": """
        <system_instructions>
          <role>Professional AI assistant</role>
          <guidelines>Concise, accurate responses</guidelines>
        </system_instructions>
        """
    }
)
```

### 2. XML Structure

Following Anthropic's recommendations, use XML tags for clear context boundaries:

```python
template = PromptTemplate(
    id="rag-assistant:v1",
    system="""
    <system_instructions>
      <role>{{role_description}}</role>
      <capabilities>
        - Answer questions from knowledge base
        - Cite sources accurately
        - Admit uncertainty when needed
      </capabilities>
    </system_instructions>

    {% if context %}
    <context>
      <retrieved_documents>
        {{context}}
      </retrieved_documents>
    </context>
    {% endif %}

    <guidelines>
      - Prioritize information from context
      - Be precise and factual
      - Provide citations when possible
    </guidelines>
    """,
    variables={
        "role_description": "You are a knowledge assistant",
        "context": None
    }
)
```

### 3. Variable Interpolation

Use Jinja2 syntax for dynamic content:

```python
# Basic variables
{{ variable_name }}

# Conditionals
{% if condition %}
  Content when true
{% endif %}

# Loops
{% for item in items %}
  - {{ item }}
{% endfor %}

# Filters
{{ text|upper }}
{{ text|truncate(100) }}
```

### 4. Environment-Based Prompts

Different prompts for different environments:

```python
template = PromptTemplate(
    id="deployment-agent:v1",
    environments={
        "dev": """
        <system>
          DEV MODE - All safety checks disabled
          {{instructions}}
        </system>
        """,
        "staging": """
        <system_instructions>
          <environment>Staging</environment>
          <safety_level>Medium</safety_level>
          {{instructions}}
        </system_instructions>
        """,
        "prod": """
        <system_instructions>
          <environment>Production</environment>
          <safety_level>High</safety_level>
          <approval_required>true</approval_required>
          {{instructions}}
        </system_instructions>
        """
    },
    variables={"instructions": "Deploy with caution"}
)

# Render for specific environment
prod_prompt = template.render(environment='prod')
```

### 5. SystemPromptBuilder

Helper for building structured XML prompts:

```python
from agents import SystemPromptBuilder

builder = SystemPromptBuilder()
builder.add_role("You are a deployment automation assistant")
builder.add_capabilities([
    "Deploy applications to cloud",
    "Rollback failed deployments",
    "Monitor deployment status"
])
builder.add_guidelines([
    "Verify environment before deploying",
    "Request approval for production",
    "Log all actions"
])
builder.add_context("company", "Acme Corp - requires security scanning")

prompt = builder.build()
# Creates well-structured XML prompt
```

## Integration with Agent

### Option 1: Using prompt_id (Recommended)

```python
from agents import Agent, PromptTemplate, register_prompt

# 1. Create and register template
template = PromptTemplate(
    id="customer-support:v1",
    system="""
    <system_instructions>
      <role>You are {{agent_name}}, a customer support specialist.</role>
      <company>{{company_name}}</company>
      <guidelines>
        - Be empathetic and professional
        - Resolve issues efficiently
        - Escalate when necessary
      </guidelines>
    </system_instructions>
    """,
    variables={
        "agent_name": "Support Assistant",
        "company_name": "Acme Corp"
    }
)
register_prompt(template)

# 2. Create agent with prompt
agent = Agent(
    id='customer-support:v1',
    model={'provider': 'anthropic', 'model': 'claude-sonnet-4'},
    prompt_id='customer-support:v1',
    prompt_vars={
        'agent_name': 'Emily',
        'company_name': 'Acme Corporation'
    },
    prompt_env='prod'
)

# 3. Use agent normally
answer = await agent.run({'message': 'I need help with my order'})
```

### Option 2: Using PromptContextManager

```python
from agents import Agent, PromptContextManager, register_prompt

# Register template
register_prompt(template)

# Create custom context manager with prompt support
ctx_mgr = PromptContextManager(
    prompt_id='customer-support:v1',
    prompt_vars={'agent_name': 'Emily'},
    prompt_env='prod',
    max_history=20  # Limit context window
)

# Use with Agent
agent = Agent(
    id='customer-support:v1',
    model={'provider': 'anthropic', 'model': 'claude-sonnet-4'},
    context_manager=ctx_mgr
)
```

### Option 3: Backwards Compatible (No Prompts)

```python
# Existing code continues to work without changes
agent = Agent(
    id='my-agent',
    model={'provider': 'openai', 'model': 'gpt-4o'}
)
# No prompt template - works as before
```

## Prompt Registry

Centralized management of prompts:

```python
from agents import PromptRegistry, register_prompt, get_prompt

# Register prompts
register_prompt(template1)
register_prompt(template2)

# Retrieve prompts
template = get_prompt("agent:v1")

# Check existence
if has_prompt("agent:v1"):
    template = get_prompt("agent:v1")

# List all prompts
all_prompts = list_prompts()

# Advanced registry operations
registry = PromptRegistry.default()

# List by prefix (useful for versions)
chat_prompts = registry.list_by_prefix("chat")
# Returns: [chat:v1, chat:v2, chat:v3]

# List by version
versions = registry.list_by_version("agent")
# Returns: {'v1': template1, 'v2': template2}

# Update existing prompt
registry.update(updated_template)

# Remove prompt
registry.remove("agent:v1")
```

## Versioning

Manage prompt evolution with versions:

```python
# Version 1 - Basic
v1 = PromptTemplate(
    id="assistant:v1",
    system="You are a helpful assistant."
)

# Version 2 - Enhanced with capabilities
v2 = PromptTemplate(
    id="assistant:v2",
    system="""
    <system_instructions>
      <role>You are an expert assistant.</role>
      <capabilities>
        - Answer complex questions
        - Provide detailed explanations
        - Cite sources
      </capabilities>
    </system_instructions>
    """
)

# Version 3 - With environment support
v3 = PromptTemplate(
    id="assistant:v3",
    environments={
        "dev": "Debug assistant - verbose mode",
        "prod": "<system_instructions>...</system_instructions>"
    }
)

# Register all versions
for version in [v1, v2, v3]:
    register_prompt(version)

# Use specific version
agent = Agent(
    id='assistant:v3',
    model={'provider': 'anthropic', 'model': 'claude-sonnet-4'},
    prompt_id='assistant:v3'
)
```

## Context Formatters

Utilities for formatting context following Anthropic best practices:

### XMLFormatter

```python
from agents import XMLFormatter

# Format conversation history
messages = [
    {'role': 'user', 'content': 'Hello'},
    {'role': 'assistant', 'content': 'Hi there!'}
]
formatted = XMLFormatter.format_conversation_history(messages)
# <conversation_history>
#   <user>Hello</user>
#   <assistant>Hi there!</assistant>
# </conversation_history>

# Format context sections
formatted = XMLFormatter.format_context_section(
    section_name="background",
    content="Main information",
    subsections={"detail": "Additional details"}
)

# Format lists
items = ["Item 1", "Item 2", "Item 3"]
formatted = XMLFormatter.format_list(items, tag_name="items")

# Format key-value data
data = {"name": "Alice", "age": 30}
formatted = XMLFormatter.format_key_value(data)
```

### ContextCompactor

Strategies for compacting context to fit token budgets:

```python
from agents import ContextCompactor

messages = [
    # ... many messages
]

# Sliding window - keep most recent N messages
compacted = ContextCompactor.sliding_window(
    messages,
    max_messages=10,
    preserve_system=True  # Always keep system messages
)

# Summarize old messages
compacted = ContextCompactor.summarize_old_messages(
    messages,
    threshold=10,  # Keep 10 recent, summarize rest
    summary_placeholder="[Earlier conversation about X, Y, Z]"
)

# Truncate long messages
compacted = ContextCompactor.truncate_long_messages(
    messages,
    max_length=500,
    truncation_indicator="... [truncated]"
)

# Compact tool results
compacted = ContextCompactor.compact_tool_results(
    messages,
    max_result_length=200
)
```

## Advanced Usage

### Dynamic Context Injection

```python
template = PromptTemplate(
    id="rag-agent:v1",
    system="""
    <system_instructions>
      <role>Knowledge base assistant</role>
    </system_instructions>

    {% if retrieved_docs %}
    <context>
      <retrieved_knowledge>
        {% for doc in retrieved_docs %}
        <document>
          <title>{{doc.title}}</title>
          <content>{{doc.content}}</content>
        </document>
        {% endfor %}
      </retrieved_knowledge>
    </context>
    {% endif %}
    """,
    variables={"retrieved_docs": []}
)

register_prompt(template)

# At runtime, inject retrieved documents
agent = Agent(
    id='rag-agent:v1',
    model={'provider': 'anthropic', 'model': 'claude-sonnet-4'},
    prompt_id='rag-agent:v1',
    prompt_vars={
        'retrieved_docs': [
            {'title': 'Doc 1', 'content': 'Content 1'},
            {'title': 'Doc 2', 'content': 'Content 2'}
        ]
    }
)
```

### Updating Variables at Runtime

```python
# Create agent with initial vars
agent = Agent(
    id='agent:v1',
    model={'provider': 'anthropic', 'model': 'claude-sonnet-4'},
    prompt_id='agent:v1',
    prompt_vars={'context': 'Initial context'}
)

# Update variables (if using PromptContextManager)
if hasattr(agent.ctxmgr, 'update_prompt_vars'):
    agent.ctxmgr.update_prompt_vars(
        context='Updated context with new information'
    )
```

### Custom Context Manager

```python
from agents import PromptContextManager

class CustomPromptContextManager(PromptContextManager):
    def messages_for_llm(self, run_id, session_id=None, additional_prompt_vars=None):
        # Add custom logic (e.g., RAG retrieval)
        retrieved_docs = self.retrieve_relevant_docs(session_id)

        # Inject into prompt vars
        additional_vars = additional_prompt_vars or {}
        additional_vars['retrieved_docs'] = retrieved_docs

        # Call parent with enhanced vars
        return super().messages_for_llm(
            run_id,
            session_id,
            additional_vars
        )

    def retrieve_relevant_docs(self, session_id):
        # Your RAG logic here
        return ["doc1", "doc2"]

# Use custom context manager
agent = Agent(
    id='agent:v1',
    model={'provider': 'anthropic', 'model': 'claude-sonnet-4'},
    context_manager=CustomPromptContextManager(
        prompt_id='agent:v1',
        prompt_env='prod'
    )
)
```

## Best Practices

### 1. Use XML Structure

XML provides clear boundaries for context:

```python
# Good - Clear structure
<system_instructions>
  <role>You are an assistant</role>
  <guidelines>Be helpful</guidelines>
</system_instructions>

# Avoid - Ambiguous boundaries
You are an assistant.
Be helpful.
```

### 2. Keep Prompts Focused

Follow 12-Factor Agent principle: Small, focused agents.

```python
# Good - Specific agent
deployment_agent = Agent(
    prompt_id='deployment:v1',  # Focused on deployments
    ...
)

# Avoid - Monolithic agent
everything_agent = Agent(
    prompt_id='do-everything:v1',  # Too broad
    ...
)
```

### 3. Version Your Prompts

Track prompt evolution:

```python
# Good - Versioned
register_prompt(PromptTemplate(id="agent:v1", ...))
register_prompt(PromptTemplate(id="agent:v2", ...))

# Avoid - Overwriting
register_prompt(PromptTemplate(id="agent", ...))  # No version
```

### 4. Use Environments

Different prompts for different contexts:

```python
PromptTemplate(
    id="agent:v1",
    environments={
        "dev": "Debug mode - verbose",
        "staging": "Test mode - validation enabled",
        "prod": "Production mode - concise"
    }
)
```

### 5. Validate Templates

Test rendering before deployment:

```python
template = PromptTemplate(
    id="agent:v1",
    system="Hello {{name}}",
    variables={"name": "World"}
)

# Validate
is_valid, error = template.validate()
if not is_valid:
    print(f"Template error: {error}")
```

### 6. Manage Context Window

Use compaction strategies for long conversations:

```python
ctx_mgr = PromptContextManager(
    prompt_id='agent:v1',
    max_history=20  # Sliding window
)

# Or use custom compaction
messages = ContextCompactor.sliding_window(messages, max_messages=10)
```

## Examples

See `examples/prompt_templates.py` for comprehensive examples including:
- Basic templates
- Environment-based prompts
- SystemPromptBuilder usage
- Agent integration
- Versioned prompts
- Dynamic context injection
- Custom context managers

## 12-Factor Alignment

The prompt module follows 12-Factor Agent principles:

### Factor 2: Own Your Prompts
- Full visibility into prompts (templates as code)
- No hidden abstractions
- Direct control over system messages

### Factor 3: Own Your Context Window
- Custom context managers
- Compaction strategies
- XML-structured context

### Factor 10: Small, Focused Agents
- Prompts tied to specific agent IDs
- Composable design
- Single-purpose templates

### Factor 12: Stateless Reducer
- Templates are pure functions
- Reproducible rendering
- No hidden state

## Anthropic Best Practices

Following [Anthropic's context engineering guide](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents):

1. **XML Structure**: Clear boundaries with XML tags
2. **Minimal Formatting**: Clean, readable structure
3. **Context as Resource**: Compaction strategies for token budgets
4. **Progressive Disclosure**: Conditional content with Jinja2
5. **Structured Memory**: External note-taking via context managers

## Migration Guide

### From Raw Prompts

**Before:**
```python
agent = Agent(
    id='my-agent',
    model={'provider': 'anthropic', 'model': 'claude-sonnet-4'}
)
# System prompt hardcoded or managed manually
```

**After:**
```python
template = PromptTemplate(
    id='my-agent:v1',
    system="<system_instructions>...</system_instructions>"
)
register_prompt(template)

agent = Agent(
    id='my-agent:v1',
    model={'provider': 'anthropic', 'model': 'claude-sonnet-4'},
    prompt_id='my-agent:v1'
)
```

### From Custom Context Manager

**Before:**
```python
class CustomContextManager(ContextManager):
    def messages_for_llm(self, run_id, session_id=None):
        messages = super().messages_for_llm(run_id, session_id)
        # Manually inject system message
        messages.insert(0, {
            'role': 'system',
            'content': 'Hardcoded prompt'
        })
        return messages
```

**After:**
```python
# Define template once
template = PromptTemplate(
    id='agent:v1',
    system='<system_instructions>...</system_instructions>'
)
register_prompt(template)

# Use PromptContextManager
ctx_mgr = PromptContextManager(prompt_id='agent:v1')
agent = Agent(..., context_manager=ctx_mgr)
```

## API Reference

See module docstrings for detailed API documentation:
- `agents.prompts.template.PromptTemplate`
- `agents.prompts.registry.PromptRegistry`
- `agents.prompts.manager.PromptManager`
- `agents.prompts.context_manager.PromptContextManager`
- `agents.prompts.formatters.*`
