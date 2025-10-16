"""
Basic RLM Example

Demonstrates how to use RLM (Recursive Language Model) to handle large contexts.
"""
import asyncio
import os
from vel import Agent, RlmConfig


async def main():
    # Create large context (simulated document)
    large_doc = """
# Product Documentation

## Overview
Our platform provides a comprehensive suite of tools for data analysis and visualization.
Founded in 2020, we have grown to serve over 10,000 customers worldwide.

## Features

### Data Import
- Support for CSV, JSON, XML, and Parquet formats
- Direct database connections (PostgreSQL, MySQL, MongoDB)
- Real-time streaming from Kafka and RabbitMQ
- Batch import with scheduling

### Analytics Engine
- SQL and NoSQL query support
- Built-in statistical functions
- Machine learning model integration
- Custom Python/R script execution

### Visualization
- Interactive dashboards
- 20+ chart types
- Real-time updates
- Export to PNG, PDF, SVG

### Collaboration
- Team workspaces
- Role-based access control
- Comments and annotations
- Version history

## Pricing

### Free Tier
- Up to 1GB storage
- 100 queries/month
- Basic visualizations
- Community support

### Professional ($49/month)
- 100GB storage
- Unlimited queries
- All visualization types
- Email support
- Custom branding

### Enterprise (Contact Sales)
- Unlimited storage
- Dedicated infrastructure
- Priority support
- SSO and advanced security
- Custom integrations

## Technical Specifications

### System Requirements
- Browser: Chrome 90+, Firefox 88+, Safari 14+
- Internet: 5 Mbps minimum
- Screen: 1024x768 minimum resolution

### API
- RESTful API with JSON
- GraphQL endpoint available
- WebSocket for real-time data
- SDKs for Python, JavaScript, Java

### Security
- SOC 2 Type II certified
- GDPR compliant
- Data encryption at rest and in transit
- Regular security audits

## Support

### Documentation
- Getting started guide
- API reference
- Video tutorials
- Best practices

### Community
- Forum (10,000+ members)
- Discord server
- Monthly webinars
- Open-source examples on GitHub

### Contact
- Email: support@platform.com
- Phone: 1-800-DATA-VIZ
- Live chat: Available 24/7
- Office hours: Mon-Fri 9am-5pm PST

## Getting Started

1. Sign up at platform.com/signup
2. Create your first project
3. Import your data
4. Build a dashboard
5. Share with your team

For more help, see our quickstart guide or watch the intro video.
""" * 50  # Repeat to make it large (simulate 5MB+ document)

    # Configure agent with RLM
    agent = Agent(
        id='rlm-agent:v1',
        model={
            'provider': 'openai',
            'model': 'gpt-4o-mini'  # Control model (cheap, fast)
        },
        rlm={
            'enabled': True,
            'depth': 1,  # Allow one level of recursion
            'control_model': {
                'provider': 'openai',
                'model': 'gpt-4o-mini'
            },
            'writer_model': {
                'provider': 'openai',
                'model': 'gpt-4o'  # Stronger model for final synthesis
            },
            'notes_cap': 200,
            'notes_window': 40,
            'budgets': {
                'max_steps_root': 12,
                'max_steps_child': 8,
                'max_tokens_total': 120000,
                'max_cost_usd': 0.50
            },
            'tools': {
                'allow_exec': False,  # Disable python_exec for safety
                'probe_max_bytes': 4096
            },
            'stream_events': True
        }
    )

    print("=" * 60)
    print("RLM Basic Example")
    print("=" * 60)
    print(f"Document size: {len(large_doc):,} bytes")
    print(f"Provider: {agent.model_cfg['provider']} / {agent.model_cfg['model']}")
    print(f"RLM enabled: {agent.rlm_config.enabled}")
    print("=" * 60)
    print()

    # Example 1: Non-streaming
    print("Example 1: Non-streaming RLM")
    print("-" * 60)

    answer = await agent.run(
        input={'message': 'What are the pricing tiers and what does each include?'},
        context_refs=large_doc  # Pass large context
    )

    print(f"Answer: {answer}")
    print()

    # Example 2: Streaming
    print("\nExample 2: Streaming RLM with events")
    print("-" * 60)

    async for event in agent.run_stream(
        input={'message': 'What integrations and APIs are available?'},
        context_refs=large_doc
    ):
        event_type = event.get('type', 'unknown')
        
        if event_type == 'data-rlm-start':
            print(f"[RLM] Starting with depth={event['data']['depth']}")

        elif event_type == 'data-rlm-step-start':
            step = event['data']['step']
            budget = event['data']['budget']
            print(f"[RLM] Step {step} - Budget: {budget['steps']}/{budget['max_steps']} steps, ${budget['cost']:.4f}/${budget['max_cost']:.2f}")

        elif event_type == 'data-rlm-probe':
            tool = event['data']['tool']
            args = event['data']['args']
            print(f"[RLM] Probing: {tool}({list(args.keys())})")

        elif event_type == 'data-rlm-note':
            text = event['data']['text'][:80]
            print(f"[RLM] Note: {text}...")

        elif event_type == 'data-rlm-final':
            answer = event['data']['answer']
            print(f"[RLM] Final answer:\n{answer}")

        elif event_type == 'data-rlm-complete':
            meta = event['data']['meta']
            budget = meta['budget']
            print(f"\n[RLM] Complete!")
            print(f"  Total steps: {budget['steps']}")
            print(f"  Total tokens: {budget['tokens']}")
            print(f"  Total cost: ${budget['cost']:.4f}")

    print()
    print("=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == '__main__':
    # Set OpenAI API key
    if not os.getenv('OPENAI_API_KEY'):
        print("Error: OPENAI_API_KEY environment variable not set")
        exit(1)

    asyncio.run(main())
