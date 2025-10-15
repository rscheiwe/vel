"""
Analyze Tool - AI SDK Multi-Step Pattern Compatible

Breaks down complex problems into smaller, manageable components using
different analytical approaches (systematic, creative, technical).
"""
import asyncio
from vel import ToolSpec, register_tool


async def analyze_handler(input: dict, ctx: dict) -> dict:
    """
    Analysis tool handler.

    Breaks down problems using different approaches:
    - systematic: Step-by-step logical breakdown
    - creative: Unconventional and innovative approaches
    - technical: Architecture and implementation focus

    Returns output with 'state' field for AI SDK compatibility.
    """
    problem = input['problem']
    approach = input['approach']

    # Simulate analysis processing time
    await asyncio.sleep(1.0)

    # Generate analysis based on approach
    analysis_templates = {
        'systematic': (
            "Systematic breakdown: "
            "1) Problem identification, "
            "2) Root cause analysis, "
            "3) Solution alternatives, "
            "4) Implementation plan"
        ),
        'creative': (
            f"Creative analysis: Exploring unconventional approaches and "
            f"innovative solutions for {problem}"
        ),
        'technical': (
            f"Technical analysis: Examining {problem} from a technical "
            f"perspective including architecture, implementation, and "
            f"optimization considerations"
        )
    }

    breakdown = analysis_templates.get(approach, analysis_templates['systematic'])

    # Generate components
    components = [
        f"Component 1: Analysis of {problem}",
        "Component 2: Solution design",
        "Component 3: Implementation strategy"
    ]

    # Return with 'state' field for AI SDK compatibility
    return {
        'state': 'ready',
        'problem': problem,
        'approach': approach,
        'breakdown': breakdown,
        'components': components
    }


# Tool specification
analyze_tool = ToolSpec(
    name='analyze',
    input_schema={
        'type': 'object',
        'properties': {
            'problem': {
                'type': 'string',
                'description': 'The problem to analyze'
            },
            'approach': {
                'type': 'string',
                'description': 'The analysis approach',
                'enum': ['systematic', 'creative', 'technical']
            }
        },
        'required': ['problem', 'approach']
    },
    output_schema={
        'type': 'object',
        'properties': {
            'state': {
                'type': 'string',
                'enum': ['loading', 'ready']
            },
            'problem': {'type': 'string'},
            'approach': {
                'type': 'string',
                'enum': ['systematic', 'creative', 'technical']
            },
            'breakdown': {'type': 'string'},
            'components': {
                'type': 'array',
                'items': {'type': 'string'}
            }
        },
        'required': ['state', 'problem', 'approach', 'breakdown', 'components']
    },
    handler=analyze_handler
)

# Register tool
register_tool(analyze_tool)
