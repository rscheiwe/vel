---
layout: default
title: "Factor 2: Own Your Prompts"
parent: 12-Factor Alignment
nav_order: 2
---

# Factor 2: Own Your Prompts

**Principle:** Take direct control of prompts instead of outsourcing to framework abstractions.

## How Vel Implements This

Vel provides the primitives but doesn't hide prompts behind abstractions. You have full control:

```python
# Direct access to messages sent to LLM
messages = agent.ctxmgr.messages_for_llm(run_id, session_id)

# Custom context manager for full prompt control
class CustomContextManager(ContextManager):
    def messages_for_llm(self, run_id: str, session_id: Optional[str] = None):
        messages = super().messages_for_llm(run_id, session_id)

        # Add custom system message
        messages.insert(0, {
            'role': 'system',
            'content': 'You are a helpful deployment assistant. Always confirm before deploying.'
        })

        # Add retrieved context (RAG)
        retrieved_docs = self.retrieve_docs(session_id)
        messages.insert(1, {
            'role': 'system',
            'content': f"Relevant context: {retrieved_docs}"
        })

        return messages

agent = Agent(
    id='my-agent',
    model={'provider': 'openai', 'model': 'gpt-4o'},
    context_manager=CustomContextManager()
)
```

## Prompt Templates

Vel also provides a flexible prompt template system:

```python
from vel import PromptTemplate, register_prompt

template = PromptTemplate(
    id="assistant:v1",
    system="""
    <system_instructions>
      <role>You are {{role_name}}, a helpful assistant.</role>
      <guidelines>Be concise and accurate</guidelines>
    </system_instructions>
    """,
    variables={"role_name": "Alex"}
)

register_prompt(template)

agent = Agent(
    id='assistant:v1',
    model={'provider': 'anthropic', 'model': 'claude-sonnet-4'},
    prompt_id='assistant:v1',
    prompt_vars={'role_name': 'Sarah'}
)
```

## Benefits

- ✓ Full transparency: see exactly what's sent to the LLM
- ✓ Easy iteration: modify prompts based on performance
- ✓ Testable: create evaluations like regular code
- ✓ No hidden abstractions: prompts are first-class code

**See:**
- [Session Management](../sessions#advanced-usage)
- [Prompt Templates](../prompts)
