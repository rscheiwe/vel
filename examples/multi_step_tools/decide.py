"""
Decide Tool - AI SDK Multi-Step Pattern Compatible

Makes decisions between different options based on evaluation criteria.
Uses scoring heuristics to recommend the best option.
"""
import asyncio
from vel import ToolSpec, register_tool


async def decide_handler(input: dict, ctx: dict) -> dict:
    """
    Decision-making tool handler.

    Evaluates multiple options against criteria and recommends the best choice
    with scoring and reasoning.

    Returns output with 'state' field for AI SDK compatibility.
    """
    options = input['options']
    criteria = input['criteria']
    context = input['context']

    # Simulate decision processing time
    await asyncio.sleep(0.8)

    # Evaluate each option
    evaluation = []

    for idx, option in enumerate(options):
        # Base score
        score = 5

        # Extract keywords from criteria
        relevant_keywords = []
        for criterion in criteria:
            relevant_keywords.extend(criterion.lower().split())

        # Check keyword matches in option
        option_words = option.lower().split()
        keyword_matches = sum(
            1 for keyword in relevant_keywords
            if any(keyword in word or word in keyword for word in option_words)
        )

        # Increase score based on keyword relevance
        score += keyword_matches * 2

        # Adjust score based on option detail level
        if len(option) > 50:
            score += 1
        if len(option) > 100:
            score += 1

        # Cap score between 1-10
        score = max(1, min(10, score))

        # Generate reasoning
        reasoning = (
            f"Option {idx + 1} scored {score}/10 based on relevance to criteria: "
            f"{', '.join(criteria)}. "
        )

        if keyword_matches > 0:
            reasoning += f"Found {keyword_matches} keyword matches."
        else:
            reasoning += "No direct keyword matches found."

        evaluation.append({
            'option': option,
            'score': score,
            'reasoning': reasoning
        })

    # Find best option
    best = max(evaluation, key=lambda e: e['score'])

    # Return with 'state' field for AI SDK compatibility
    return {
        'state': 'ready',
        'context': context,
        'options': options,
        'criteria': criteria,
        'evaluation': evaluation,
        'decision': best['option'],
        'reasoning': (
            f"Selected: {best['option']} (Score: {best['score']}/10) - "
            f"{best['reasoning']}"
        )
    }


# Tool specification
decide_tool = ToolSpec(
    name='decide',
    input_schema={
        'type': 'object',
        'properties': {
            'options': {
                'type': 'array',
                'description': 'List of options to choose from',
                'items': {'type': 'string'},
                'minItems': 2
            },
            'criteria': {
                'type': 'array',
                'description': 'Criteria for evaluation',
                'items': {'type': 'string'},
                'minItems': 1
            },
            'context': {
                'type': 'string',
                'description': 'Context for the decision'
            }
        },
        'required': ['options', 'criteria', 'context']
    },
    output_schema={
        'type': 'object',
        'properties': {
            'state': {
                'type': 'string',
                'enum': ['loading', 'ready']
            },
            'context': {'type': 'string'},
            'options': {
                'type': 'array',
                'items': {'type': 'string'}
            },
            'criteria': {
                'type': 'array',
                'items': {'type': 'string'}
            },
            'evaluation': {
                'type': 'array',
                'items': {
                    'type': 'object',
                    'properties': {
                        'option': {'type': 'string'},
                        'score': {'type': 'number'},
                        'reasoning': {'type': 'string'}
                    },
                    'required': ['option', 'score', 'reasoning']
                }
            },
            'decision': {'type': 'string'},
            'reasoning': {'type': 'string'}
        },
        'required': ['state', 'context', 'options', 'criteria', 'evaluation', 'decision', 'reasoning']
    },
    handler=decide_handler
)

# Register tool
register_tool(decide_tool)
