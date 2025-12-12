"""
Extended Thinking module for Vel.

Provides multi-pass reasoning (Analyze -> Critique -> Refine -> Conclude)
that emits Vercel AI SDK V5 stream protocol events.
"""

from .config import ThinkingConfig
from .controller import ReflectionController

__all__ = ['ThinkingConfig', 'ReflectionController']
