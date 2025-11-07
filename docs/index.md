---
layout: default
title: Home
nav_order: 1
description: "Vel is a 12-Factor Agent Runtime with streaming responses, multiple LLM providers, and flexible prompt management."
permalink: /
---

# Vel Agent Runtime
{: .fs-9 }

A production-ready AI agent runtime aligned with 12-Factor Agent principles. Built for reliability, scalability, and maintainability.
{: .fs-6 .fw-300 }

[Get started now](getting-started){: .btn .btn-primary .fs-5 .mb-4 .mb-md-0 .mr-2 }
[View on GitHub](https://github.com/rscheiwe/vel){: .btn .fs-5 .mb-4 .mb-md-0 }

---

## Features

**Dual Execution Modes**
: Streaming (SSE) and non-streaming (JSON) responses

**Multiple LLM Providers**
: OpenAI, Google Gemini, and Anthropic Claude with plug-and-play architecture

**Vercel AI SDK Compatible**
: Seamless integration with React frontends using `useChat` hook and `convertToModelMessages()`

**Message Translation**
: Automatic conversion from ModelMessage to provider-specific formats (OpenAI/Anthropic/Gemini)

**Flexible Prompts**
: Jinja2 templating with XML formatting, environment-based configuration, and version control

**Stream Protocol**
: Vercel AI SDK-compatible event system for provider-agnostic streaming

**Tool System**
: JSON schema-validated tools with async support

**Production-Ready**
: Pure Python library with minimal dependencies and 12-Factor alignment

---

## Quick Start

### Installation

```bash
# Clone and install
git clone https://github.com/rscheiwe/vel
cd vel
pip install -e .

# Set up environment
cp .env.example .env
# Edit .env with your API keys
```

### Basic Usage

```python
import asyncio
from vel import Agent

async def main():
    agent = Agent(
        id='chat-general:v1',
        model={'provider': 'openai', 'model': 'gpt-4o'},
        tools=['get_weather']
    )

    answer = await agent.run({'message': 'What is the weather?'})
    print(answer)

if __name__ == '__main__':
    asyncio.run(main())
```

---

## Documentation

<div class="grid">
  <div class="grid-item">
    <h3><a href="getting-started">Getting Started</a></h3>
    <p>Installation and quick start guide</p>
  </div>

  <div class="grid-item">
    <h3><a href="sessions">Session Management</a></h3>
    <p>Multi-turn conversations with memory</p>
  </div>

  <div class="grid-item">
    <h3><a href="prompts">Prompt Templates</a></h3>
    <p>Flexible prompt management with Jinja2</p>
  </div>

  <div class="grid-item">
    <h3><a href="message-formats">Message Formats</a></h3>
    <p>UIMessage, ModelMessage, and provider translation</p>
  </div>

  <div class="grid-item">
    <h3><a href="providers">Providers</a></h3>
    <p>OpenAI, Gemini, and Claude configuration</p>
  </div>

  <div class="grid-item">
    <h3><a href="tools">Tools</a></h3>
    <p>Custom tool creation and execution</p>
  </div>

  <div class="grid-item">
    <h3><a href="stream-protocol">Stream Protocol</a></h3>
    <p>Event streaming reference</p>
  </div>

  <div class="grid-item">
    <h3><a href="api-reference">API Reference</a></h3>
    <p>Complete API documentation</p>
  </div>

  <div class="grid-item">
    <h3><a href="12-factor-alignment">12-Factor Alignment</a></h3>
    <p>Production-ready agent principles</p>
  </div>
</div>

---

## 12-Factor Agent Principles

Vel is designed following the [12-Factor Agents](https://github.com/humanlayer/12-factor-agents) methodology:

- ✓ **Own Your Prompts** - Direct control, no abstractions
- ✓ **Own Your Context Window** - Custom context managers
- ✓ **Stateless Reducer** - Predictable, reproducible behavior
- ✓ **Small, Focused Agents** - Composable design

[Learn more about 12-Factor alignment →](12-factor-alignment)

---

## License

Vel is distributed under the [MIT License](https://github.com/rscheiwe/vel/blob/main/LICENSE).
