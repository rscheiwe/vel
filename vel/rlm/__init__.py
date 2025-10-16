"""
RLM (Recursive Language Model) Module

Provides middleware for handling very long contexts through recursive reasoning,
bounded iteration, scratchpad notes, and specialized context probing.

Based on Alex Zhang's RLM approach: https://alexzhang13.github.io/blog/2025/rlm/
"""
from __future__ import annotations

from .config import RlmConfig
from .scratchpad import Scratchpad, Note
from .budget import Budget
from .context_store import ContextStore
from .controller import RlmController

__all__ = [
    'RlmConfig',
    'Scratchpad',
    'Note',
    'Budget',
    'ContextStore',
    'RlmController',
]
