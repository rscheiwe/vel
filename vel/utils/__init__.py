"""
Utility modules for Vel agents.
"""
from __future__ import annotations

__all__ = ['WorkQueue', 'MessageReducer']

try:
    from .async_queue import WorkQueue
except ImportError:
    WorkQueue = None

from .message_reducer import MessageReducer
