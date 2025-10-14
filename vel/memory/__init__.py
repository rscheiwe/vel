"""
Memory system for Vel agents.

Provides optional runtime-owned memory without LLM tool calls:
- FactStore: Namespaced key-value store for long-term facts
- ReasoningBank: Strategic memory with embeddings
"""
from __future__ import annotations
import warnings

__all__ = [
    'FactStore',
    'ReasoningBank',
    'ReasoningBankStore',
    'Embeddings',
    'StrategyItem',
    # Deprecated (backwards compatibility)
    'EpisodicStore'
]

try:
    from .fact_store import FactStore
    # Backwards compatibility - will be removed in v2.0
    EpisodicStore = FactStore
except ImportError:
    FactStore = None
    EpisodicStore = None

try:
    from .strategy_reasoningbank import (
        ReasoningBank,
        ReasoningBankStore,
        Embeddings,
        StrategyItem,
    )
except ImportError:
    ReasoningBank = None
    ReasoningBankStore = None
    Embeddings = None
    StrategyItem = None
