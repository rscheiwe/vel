"""
Utility modules for Vel agents.
"""
from __future__ import annotations

__all__ = ['WorkQueue', 'MessageReducer', 'convert_to_model_messages', 'convert_from_legacy_format']

try:
    from .async_queue import WorkQueue
except ImportError:
    WorkQueue = None

from .message_reducer import MessageReducer
from .message_converter import convert_to_model_messages, convert_from_legacy_format
