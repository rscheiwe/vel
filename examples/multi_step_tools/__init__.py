"""
Multi-Step Tools Module

AI SDK compatible tools for building multi-step agents.
Each tool returns output with 'state' field for frontend compatibility.
"""

# Import all tools to register them
from .web_search import web_search_tool
from .news_search import news_search_tool
from .analyze import analyze_tool
from .decide import decide_tool
from .provide_answer import provide_answer_tool

__all__ = [
    'web_search_tool',
    'news_search_tool',
    'analyze_tool',
    'decide_tool',
    'provide_answer_tool'
]
