"""
Web Search Tool - Perplexity API Integration

Production-ready web search using Perplexity Sonar API.
Matches Vercel AI SDK's multi-step agent pattern with 'state' field.

Environment Variables:
    PERPLEXITY_API_KEY: Perplexity API key (required)

Example:
    export PERPLEXITY_API_KEY=pplx-...
"""
import os
import json
from typing import Optional
from urllib.parse import urlparse
from openai import AsyncOpenAI
from vel import ToolSpec, register_tool


async def web_search_handler(input: dict, ctx: dict) -> dict:
    """
    Web search tool handler using Perplexity Sonar API.

    Args:
        input: Dict with 'query' (str) and optional 'limit' (int, default 5)
        ctx: Context dict with 'run_id' and 'session_id'

    Returns:
        Dict with 'state', 'query', and 'results' array

    The 'state' field can be:
    - "loading": Tool is processing (not used in async handler)
    - "ready": Tool has completed
    """
    query = input['query']
    limit = input.get('limit', 5)

    # Get API key from environment
    api_key = os.getenv('PERPLEXITY_API_KEY')
    if not api_key:
        return {
            'state': 'ready',
            'query': query,
            'results': [{
                'title': 'Configuration Error',
                'url': 'https://docs.perplexity.ai',
                'snippet': 'PERPLEXITY_API_KEY environment variable not set',
                'source': 'Error'
            }]
        }

    try:
        # Initialize Perplexity client (OpenAI-compatible API)
        client = AsyncOpenAI(
            api_key=api_key,
            base_url='https://api.perplexity.ai'
        )

        # System prompt guides structured JSON output
        system_prompt = (
            "You are a search assistant. Return strictly valid JSON matching this schema:\n"
            '{"query": "string", "results": [{"title": "string", "url": "string", '
            '"snippet": "string", "source": "string"}]}\n'
            "For each result include title, url, a short snippet, and a source hostname."
        )

        # User prompt with query and limit
        user_prompt = (
            f"Search the web for: {query}. "
            f"Return up to {limit} high-quality, diverse results with proper URLs."
        )

        # Call Perplexity Sonar model
        response = await client.chat.completions.create(
            model='sonar',
            messages=[
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_prompt}
            ],
            temperature=0.2,
            max_tokens=2000
        )

        # Parse JSON response
        content = response.choices[0].message.content
        if not content:
            raise ValueError("Empty response from Perplexity API")

        data = json.loads(content)

        # Normalize results
        results = []
        for r in (data.get('results', []))[:limit]:
            # Extract source hostname if not provided
            source = r.get('source')
            if not source and r.get('url'):
                try:
                    source = urlparse(r['url']).hostname
                except Exception:
                    source = None

            results.append({
                'title': r.get('title', ''),
                'url': r.get('url', ''),
                'snippet': r.get('snippet', ''),
                'source': source
            })

        return {
            'state': 'ready',
            'query': query,
            'results': results
        }

    except json.JSONDecodeError as e:
        return {
            'state': 'ready',
            'query': query,
            'results': [{
                'title': 'Parse Error',
                'url': 'https://perplexity.ai',
                'snippet': f'Failed to parse Perplexity response: {str(e)}',
                'source': 'Error'
            }]
        }
    except Exception as e:
        return {
            'state': 'ready',
            'query': query,
            'results': [{
                'title': 'Search Failed',
                'url': 'https://perplexity.ai',
                'snippet': f'Unable to perform web search: {str(e)}',
                'source': 'Error'
            }]
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
