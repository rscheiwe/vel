"""
Provide Answer Tool - AI SDK Multi-Step Pattern Compatible

Terminating tool that provides final structured answer with citations.
Matches Vercel AI SDK's provideAnswer pattern.
"""
from vel import ToolSpec, register_tool


async def provide_answer_handler(input: dict, ctx: dict) -> dict:
    """
    Provide final answer tool handler.

    This tool terminates the agent loop and provides structured output.
    Returns output with 'state' field for AI SDK compatibility.
    """
    answer = input['answer']
    steps = input.get('steps', [])
    confidence = input.get('confidence', 0.8)
    sources = input.get('sources', [])
    citations = input.get('citations', [])

    # Return with 'state' field for AI SDK compatibility
    return {
        'state': 'ready',  # AI SDK expects this field
        'answer': answer,
        'steps': steps,
        'confidence': confidence,
        'sources': sources,
        'citations': citations,
        'summary': f"Based on my analysis using {len(steps)} steps, here's what I found:"
    }


# Tool specification
provide_answer_tool = ToolSpec(
    name='provideAnswer',
    input_schema={
        'type': 'object',
        'properties': {
            'answer': {
                'type': 'string',
                'description': 'The final answer with inline citations [1], [2], etc.'
            },
            'steps': {
                'type': 'array',
                'description': 'All steps taken to reach the answer',
                'items': {
                    'type': 'object',
                    'properties': {
                        'step': {'type': 'string', 'description': 'What was done'},
                        'reasoning': {'type': 'string', 'description': 'Why it was done'},
                        'result': {'type': 'string', 'description': 'Result of this step'}
                    },
                    'required': ['step', 'reasoning', 'result']
                }
            },
            'confidence': {
                'type': 'number',
                'description': 'Confidence level (0-1)',
                'minimum': 0,
                'maximum': 1
            },
            'sources': {
                'type': 'array',
                'description': 'Source URLs used',
                'items': {'type': 'string'}
            },
            'citations': {
                'type': 'array',
                'description': 'Detailed citation information',
                'items': {
                    'type': 'object',
                    'properties': {
                        'number': {'type': 'string', 'description': 'Citation number (1, 2, etc.)'},
                        'title': {'type': 'string'},
                        'url': {'type': 'string'},
                        'description': {'type': 'string'},
                        'snippet': {'type': 'string'}
                    },
                    'required': ['number', 'title', 'url']
                }
            }
        },
        'required': ['answer', 'steps', 'confidence']
    },
    output_schema={
        'type': 'object',
        'properties': {
            'state': {
                'type': 'string',
                'enum': ['loading', 'ready']
            },
            'answer': {'type': 'string'},
            'steps': {'type': 'array'},
            'confidence': {'type': 'number'},
            'sources': {'type': 'array'},
            'citations': {'type': 'array'},
            'summary': {'type': 'string'}
        },
        'required': ['state', 'answer', 'steps', 'confidence']
    },
    handler=provide_answer_handler
)

# Register tool
register_tool(provide_answer_tool)
