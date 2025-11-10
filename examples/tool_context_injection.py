"""
Tool Context Injection Example

Demonstrates how to use tool_context to inject shared resources into tool handlers.
Shows three practical use cases:
1. Database connections
2. Storage backends
3. Configuration and user context

The tool_context parameter enables dependency injection, making tools flexible,
testable, and independent of global state.
"""
import asyncio
import json
from typing import Dict, Any, List
from vel import Agent, ToolSpec, register_tool

# ====================================================================
# Example 1: Mock Database Connection
# ====================================================================

class MockDatabase:
    """Simulated database for demonstration"""
    def __init__(self):
        self.data = {
            'users': [
                {'id': 1, 'name': 'Alice', 'email': 'alice@example.com'},
                {'id': 2, 'name': 'Bob', 'email': 'bob@example.com'}
            ],
            'products': [
                {'id': 1, 'name': 'Laptop', 'price': 999.99},
                {'id': 2, 'name': 'Mouse', 'price': 29.99}
            ]
        }

    async def query(self, table: str, filter_key: str = None, filter_value: Any = None) -> List[Dict]:
        """Query database table with optional filtering"""
        await asyncio.sleep(0.1)  # Simulate DB latency

        results = self.data.get(table, [])

        if filter_key and filter_value:
            results = [r for r in results if r.get(filter_key) == filter_value]

        return results

# ====================================================================
# Example 2: Mock Storage Backend
# ====================================================================

class MockStorage:
    """Simulated storage backend for artifacts/files"""
    def __init__(self):
        self.artifacts = {}

    async def get_artifact(self, artifact_id: str) -> Dict[str, Any]:
        """Retrieve artifact by ID"""
        await asyncio.sleep(0.05)  # Simulate storage latency
        return self.artifacts.get(artifact_id)

    async def save_artifact(self, artifact_id: str, data: Dict[str, Any]) -> None:
        """Save artifact"""
        await asyncio.sleep(0.05)
        self.artifacts[artifact_id] = data

# ====================================================================
# Tool 1: Query Database (uses injected DB connection)
# ====================================================================

async def query_database_handler(input: Dict, ctx: Dict) -> Dict:
    """
    Tool that queries database using injected connection.

    Accesses 'db' from ctx - no global variables needed.
    """
    # Get database from context (injected by agent)
    db = ctx.get('db')
    if not db:
        return {'error': 'Database connection not available'}

    table = input['table']
    filter_key = input.get('filter_key')
    filter_value = input.get('filter_value')

    # Use injected database
    results = await db.query(table, filter_key, filter_value)

    return {
        'table': table,
        'count': len(results),
        'results': results
    }

register_tool(ToolSpec(
    name='query_database',
    description='Query the database for users, products, or other tables',
    input_schema={
        'type': 'object',
        'properties': {
            'table': {'type': 'string', 'description': 'Table name (users, products)'},
            'filter_key': {'type': 'string', 'description': 'Optional: field to filter by'},
            'filter_value': {'description': 'Optional: value to match'}
        },
        'required': ['table']
    },
    output_schema={
        'type': 'object',
        'properties': {
            'table': {'type': 'string'},
            'count': {'type': 'integer'},
            'results': {'type': 'array'}
        }
    },
    handler=query_database_handler
))

# ====================================================================
# Tool 2: Manage Artifact (uses injected storage)
# ====================================================================

async def manage_artifact_handler(input: Dict, ctx: Dict) -> Dict:
    """
    Tool that manages artifacts using injected storage backend.

    Can get or save artifacts depending on operation.
    """
    storage = ctx.get('storage')
    if not storage:
        return {'error': 'Storage backend not available'}

    operation = input['operation']
    artifact_id = input['artifact_id']

    if operation == 'get':
        artifact = await storage.get_artifact(artifact_id)
        return {
            'operation': 'get',
            'artifact_id': artifact_id,
            'found': artifact is not None,
            'data': artifact
        }

    elif operation == 'save':
        data = input.get('data', {})
        await storage.save_artifact(artifact_id, data)
        return {
            'operation': 'save',
            'artifact_id': artifact_id,
            'status': 'saved'
        }

    return {'error': f'Unknown operation: {operation}'}

register_tool(ToolSpec(
    name='manage_artifact',
    description='Get or save artifacts in storage',
    input_schema={
        'type': 'object',
        'properties': {
            'operation': {'type': 'string', 'enum': ['get', 'save']},
            'artifact_id': {'type': 'string'},
            'data': {'type': 'object', 'description': 'Data to save (for save operation)'}
        },
        'required': ['operation', 'artifact_id']
    },
    output_schema={
        'type': 'object',
        'properties': {
            'operation': {'type': 'string'},
            'artifact_id': {'type': 'string'},
            'found': {'type': 'boolean'},
            'status': {'type': 'string'},
            'data': {'type': 'object'}
        }
    },
    handler=manage_artifact_handler
))

# ====================================================================
# Tool 3: User-Aware Action (uses injected user context)
# ====================================================================

def user_action_handler(input: Dict, ctx: Dict) -> Dict:
    """
    Tool that accesses user context for personalized behavior.

    Gets user_id and permissions from context.
    """
    user_id = ctx.get('user_id', 'unknown')
    permissions = ctx.get('permissions', [])
    action = input['action']

    # Check permissions
    if action == 'delete' and 'delete' not in permissions:
        return {
            'action': action,
            'status': 'denied',
            'reason': 'User lacks delete permission'
        }

    return {
        'action': action,
        'status': 'allowed',
        'user_id': user_id,
        'permissions': permissions
    }

register_tool(ToolSpec(
    name='user_action',
    description='Perform user action with permission checking',
    input_schema={
        'type': 'object',
        'properties': {
            'action': {'type': 'string', 'enum': ['read', 'write', 'delete']}
        },
        'required': ['action']
    },
    output_schema={
        'type': 'object',
        'properties': {
            'action': {'type': 'string'},
            'status': {'type': 'string'},
            'reason': {'type': 'string'},
            'user_id': {'type': 'string'},
            'permissions': {'type': 'array'}
        }
    },
    handler=user_action_handler
))

# ====================================================================
# Main Example
# ====================================================================

async def main():
    print("=" * 70)
    print("Tool Context Injection Example")
    print("=" * 70)
    print()

    # ================================================================
    # Example 1: Database Tool
    # ================================================================
    print("Example 1: Database Connection Injection")
    print("-" * 70)

    # Create mock database
    db = MockDatabase()

    # Create agent with database in tool_context
    db_agent = Agent(
        id='db-agent:v1',
        model={'provider': 'openai', 'model': 'gpt-4o-mini'},
        tools=['query_database'],
        tool_context={'db': db}  # ← Inject database connection
    )

    query = "Show me all users in the database"
    print(f"Query: {query}")
    print()

    result = await db_agent.run({'message': query})
    print(f"Result: {result}")
    print()
    print()

    # ================================================================
    # Example 2: Storage Backend Tool
    # ================================================================
    print("Example 2: Storage Backend Injection")
    print("-" * 70)

    # Create mock storage
    storage = MockStorage()

    # Pre-populate with an artifact
    await storage.save_artifact('artifact_123', {
        'title': 'Sales Report',
        'type': 'table',
        'data': 'revenue,expenses\n1000,500'
    })

    # Create agent with storage in tool_context
    storage_agent = Agent(
        id='storage-agent:v1',
        model={'provider': 'openai', 'model': 'gpt-4o-mini'},
        tools=['manage_artifact'],
        tool_context={'storage': storage}  # ← Inject storage backend
    )

    query = "Get the artifact with ID artifact_123"
    print(f"Query: {query}")
    print()

    result = await storage_agent.run({'message': query})
    print(f"Result: {result}")
    print()
    print()

    # ================================================================
    # Example 3: User Context Tool
    # ================================================================
    print("Example 3: User Context Injection")
    print("-" * 70)

    # Scenario A: User with limited permissions
    user_agent_limited = Agent(
        id='user-agent:v1',
        model={'provider': 'openai', 'model': 'gpt-4o-mini'},
        tools=['user_action'],
        tool_context={
            'user_id': 'user_456',
            'permissions': ['read', 'write']  # No delete permission
        }
    )

    query = "Delete the file"
    print(f"User with limited permissions")
    print(f"Query: {query}")
    print()

    result = await user_agent_limited.run({'message': query})
    print(f"Result: {result}")
    print()

    # Scenario B: Admin user with full permissions
    user_agent_admin = Agent(
        id='user-agent:v1',
        model={'provider': 'openai', 'model': 'gpt-4o-mini'},
        tools=['user_action'],
        tool_context={
            'user_id': 'admin_001',
            'permissions': ['read', 'write', 'delete']  # Full permissions
        }
    )

    print(f"Admin user with full permissions")
    print(f"Query: {query}")
    print()

    result = await user_agent_admin.run({'message': query})
    print(f"Result: {result}")
    print()
    print()

    # ================================================================
    # Example 4: Multiple Resources
    # ================================================================
    print("Example 4: Multiple Resources in Context")
    print("-" * 70)

    # Create agent with ALL resources
    multi_agent = Agent(
        id='multi-agent:v1',
        model={'provider': 'openai', 'model': 'gpt-4o-mini'},
        tools=['query_database', 'manage_artifact', 'user_action'],
        tool_context={
            'db': db,                    # Database connection
            'storage': storage,          # Storage backend
            'user_id': 'user_789',       # User context
            'permissions': ['read'],     # Permissions
            'config': {                  # Additional config
                'env': 'production',
                'region': 'us-west-2'
            }
        }
    )

    print("Agent has access to:")
    print("  - Database connection")
    print("  - Storage backend")
    print("  - User context (ID + permissions)")
    print("  - Configuration")
    print()
    print("All tools can access their required resources from tool_context")
    print()

    # ================================================================
    # Key Takeaways
    # ================================================================
    print("=" * 70)
    print("Key Takeaways")
    print("=" * 70)
    print()
    print("1. tool_context enables dependency injection for tools")
    print("2. Pass shared resources (DB, storage, config) via tool_context")
    print("3. Tools access resources via ctx.get('resource_name')")
    print("4. Different agent instances can have different contexts")
    print("5. No global variables needed - better for testing and flexibility")
    print()
    print("=" * 70)
    print("Use Cases:")
    print("=" * 70)
    print("✓ Database connections (per-tenant, per-user)")
    print("✓ Storage backends (S3, local files, message-based)")
    print("✓ API clients (external services)")
    print("✓ User context (ID, permissions, preferences)")
    print("✓ Configuration (environment, feature flags)")
    print("✓ Caching layers (Redis, in-memory)")
    print()

if __name__ == '__main__':
    asyncio.run(main())
