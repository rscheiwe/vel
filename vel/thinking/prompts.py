"""Prompt templates for Extended Thinking phases."""


ANALYZE_PROMPT = """Analyze the following question carefully and thoroughly.

**Your task:**
1. Identify what is being asked
2. Break down the problem into key components
3. Note any ambiguities or assumptions
4. Consider relevant knowledge and context
5. Form an initial approach or hypothesis

**Question:**
{question}

**Provide your analysis:**"""


CRITIQUE_PROMPT = """Review the following reasoning and identify potential issues.

**Your task:**
1. Identify logical flaws or gaps in reasoning
2. Note unsupported assumptions
3. Consider alternative interpretations
4. Point out missing considerations
5. Highlight anything that could be wrong

**Content to critique:**
{content}

**Provide your critiques:**"""


REFINE_PROMPT = """Address the critiques and strengthen your reasoning.

**Original question:**
{question}

**Current reasoning:**
{content}

**Critiques to address:**
{critiques}

**Your task:**
1. Address each critique systematically
2. Fill in gaps in reasoning
3. Reconsider assumptions
4. Strengthen weak points
5. Assess your confidence in the refined reasoning

**Provide your refined reasoning, then on a separate line state your confidence level as: Confidence: X%**"""


CONCLUDE_PROMPT = """Based on your reasoning, provide a clear final answer.

**Question:**
{question}

**Your reasoning:**
{reasoning}

**Reasoning confidence:** {confidence:.0%}

**Provide your final answer. Be clear, direct, and well-structured:**"""


# Phase labels for UI display
PHASE_LABELS = {
    'analyzing': 'Analyzing the question...',
    'critiquing': 'Reviewing reasoning...',
    'refining': 'Refining analysis...',
    'concluding': 'Synthesizing answer...'
}
