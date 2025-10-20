"""
API Key Configuration Example

Demonstrates two ways to provide API keys to Vel agents:
1. Environment variables (recommended for applications)
2. Explicit API keys (recommended for libraries/multi-tenant)
"""
import asyncio
import os
from vel import Agent

async def main():
    print("=== Vel API Key Configuration Examples ===\n")

    # ============================================================================
    # Method 1: Environment Variables (recommended for applications)
    # ============================================================================
    print("Method 1: Environment Variables")
    print("-" * 50)

    # Set environment variables in your shell or .env file:
    # export OPENAI_API_KEY='sk-...'
    # export ANTHROPIC_API_KEY='sk-ant-...'
    # export GOOGLE_API_KEY='...'

    # Then create agents without specifying api_key
    agent_env = Agent(
        id='env-agent',
        model={'provider': 'openai', 'model': 'gpt-4o'}
        # Uses OPENAI_API_KEY from environment
    )
    print("✓ Agent created using OPENAI_API_KEY from environment")
    print(f"  API key: {agent_env._get_provider().api_key[:10]}...")
    print()

    # ============================================================================
    # Method 2: Explicit API Keys (recommended for libraries/multi-tenant)
    # ============================================================================
    print("Method 2: Explicit API Keys")
    print("-" * 50)

    # Pass API key directly in model config
    agent_explicit = Agent(
        id='explicit-agent',
        model={
            'provider': 'openai',
            'model': 'gpt-4o',
            'api_key': 'sk-explicit-key-123'  # Override environment variable
        }
    )
    print("✓ Agent created with explicit API key")
    print(f"  API key: {agent_explicit._get_provider().api_key}")
    print()

    # ============================================================================
    # Use Case: Multi-Tenant Application
    # ============================================================================
    print("Use Case: Multi-Tenant Application")
    print("-" * 50)

    # Different agents for different tenants/users
    tenant_agents = {
        'tenant-1': Agent(
            id='tenant-1-agent',
            model={'provider': 'openai', 'model': 'gpt-4o', 'api_key': 'sk-tenant1-key'}
        ),
        'tenant-2': Agent(
            id='tenant-2-agent',
            model={'provider': 'openai', 'model': 'gpt-4o', 'api_key': 'sk-tenant2-key'}
        ),
        'tenant-3': Agent(
            id='tenant-3-agent',
            model={'provider': 'anthropic', 'model': 'claude-3-5-sonnet-20241022', 'api_key': 'sk-ant-tenant3'}
        ),
    }

    for tenant_id, agent in tenant_agents.items():
        print(f"✓ {tenant_id}: {agent._get_provider().name} with key {agent._get_provider().api_key[:15]}...")

    print()

    # ============================================================================
    # Use Case: Library/Package Development
    # ============================================================================
    print("Use Case: Library/Package Development")
    print("-" * 50)
    print("""
If you're building a library that uses Vel internally, you should:

1. Accept API keys as parameters in your library's functions
2. Pass them to Vel agents via model config
3. Don't rely on environment variables (your users control those)

Example:

    class MyLibrary:
        def __init__(self, openai_api_key: str):
            self.agent = Agent(
                id='my-library-agent',
                model={
                    'provider': 'openai',
                    'model': 'gpt-4o',
                    'api_key': openai_api_key  # User-provided
                }
            )

        async def process(self, text: str):
            return await self.agent.run({'message': text})

    # Your library users do this:
    lib = MyLibrary(openai_api_key='sk-...')
    result = await lib.process('hello')
    """)

    # ============================================================================
    # Error Handling: No API Key Available
    # ============================================================================
    print("Error Handling: No API Key Available")
    print("-" * 50)

    # Temporarily remove env var to demonstrate error
    original_key = os.environ.pop('OPENAI_API_KEY', None)

    try:
        agent_no_key = Agent(
            id='no-key-agent',
            model={'provider': 'openai', 'model': 'gpt-4o'}
        )
        # This will fail when trying to access the provider
        provider = agent_no_key._get_provider()
        print("✗ Should have raised an error")
    except ValueError as e:
        print(f"✓ Correctly raised error: {e}")
    finally:
        # Restore env var
        if original_key:
            os.environ['OPENAI_API_KEY'] = original_key

    print()
    print("=== Examples Complete ===")

if __name__ == '__main__':
    asyncio.run(main())
