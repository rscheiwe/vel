"""
Web Search Tool - AI SDK Multi-Step Pattern Compatible

Example tool that matches Vercel AI SDK's multi-step agent pattern.
Includes 'state' field in output for frontend compatibility.
"""
from vel import ToolSpec, register_tool


async def web_search_handler(input: dict, ctx: dict) -> dict:
    """
    Web search tool handler.

    Returns output with 'state' field for AI SDK compatibility.
    The 'state' field can be:
    - "loading": Tool is processing
    - "ready": Tool has completed
    """
    query = input['query']
    limit = input.get('limit', 5)

    # Simulate web search (in production, call actual search API)
    results = [
        {
            'title': f'Result {i+1} for {query}',
            'url': f'https://example.com/result-{i+1}',
            'snippet': f'This is a sample snippet for result {i+1}',
            'source': 'example.com'
        }
        for i in range(min(limit, 5))
    ]

    # Return with 'state' field for AI SDK compatibility
    return {
        'state': 'ready',  # AI SDK expects this field
        'query': query,
        'results': results
    }


# Tool specification
web_search_tool = ToolSpec(
    name='websearch',
    input_schema={
        'type': 'object',
        'properties': {
            'query': {
                'type': 'string',
                'description': 'Search query'
            },
            'limit': {
                'type': 'number',
                'description': 'Maximum number of results',
                'minimum': 1,
                'maximum': 20,
                'default': 5
            }
        },
        'required': ['query']
    },
    output_schema={
        'type': 'object',
        'properties': {
            'state': {
                'type': 'string',
                'enum': ['loading', 'ready']
            },
            'query': {'type': 'string'},
            'results': {
                'type': 'array',
                'items': {
                    'type': 'object',
                    'properties': {
                        'title': {'type': 'string'},
                        'url': {'type': 'string'},
                        'snippet': {'type': 'string'},
                        'source': {'type': 'string'}
                    },
                    'required': ['title', 'url']
                }
            }
        },
        'required': ['state', 'query', 'results']
    },
    handler=web_search_handler
)

# Register tool
register_tool(web_search_tool)
