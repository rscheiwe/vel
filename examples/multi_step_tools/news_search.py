"""
News Search Tool - AI SDK Multi-Step Pattern Compatible

Searches Hacker News for recent headlines related to a topic.
"""
import httpx
from datetime import datetime
from vel import ToolSpec, register_tool


async def news_search_handler(input: dict, ctx: dict) -> dict:
    """
    News search tool handler using Hacker News Algolia API.

    Returns output with 'state' field for AI SDK compatibility.
    """
    topic = input['topic']
    limit = input.get('limit', 5)

    try:
        # Use Hacker News Algolia API
        params = {
            'query': topic,
            'tags': 'story',
            'hitsPerPage': str(limit)
        }

        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(
                'https://hn.algolia.com/api/v1/search',
                params=params
            )
            response.raise_for_status()
            data = response.json()

        # Parse results
        items = []
        for hit in data.get('hits', [])[:limit]:
            items.append({
                'id': str(hit.get('objectID', '')),
                'title': hit.get('title') or hit.get('story_title') or '(untitled)',
                'url': hit.get('url') or hit.get('story_url'),
                'publishedAt': hit.get('created_at')
            })

        # Return with 'state' field for AI SDK compatibility
        return {
            'state': 'ready',
            'topic': topic,
            'items': items
        }

    except Exception as e:
        print(f"News search error: {e}")
        # Return error result
        return {
            'state': 'ready',
            'topic': topic,
            'items': [{
                'id': 'error',
                'title': 'News search failed',
                'url': 'https://news.ycombinator.com',
                'publishedAt': datetime.utcnow().isoformat() + 'Z'
            }]
        }


# Tool specification
news_search_tool = ToolSpec(
    name='news',
    input_schema={
        'type': 'object',
        'properties': {
            'topic': {
                'type': 'string',
                'description': 'Topic to search for in news',
                'minLength': 1
            },
            'limit': {
                'type': 'number',
                'description': 'Maximum number of news items',
                'minimum': 1,
                'maximum': 20,
                'default': 5
            }
        },
        'required': ['topic']
    },
    output_schema={
        'type': 'object',
        'properties': {
            'state': {
                'type': 'string',
                'enum': ['loading', 'ready']
            },
            'topic': {'type': 'string'},
            'items': {
                'type': 'array',
                'items': {
                    'type': 'object',
                    'properties': {
                        'id': {'type': 'string'},
                        'title': {'type': 'string'},
                        'url': {'type': 'string'},
                        'publishedAt': {'type': 'string'}
                    },
                    'required': ['id', 'title']
                }
            }
        },
        'required': ['state', 'topic', 'items']
    },
    handler=news_search_handler
)

# Register tool
register_tool(news_search_tool)
