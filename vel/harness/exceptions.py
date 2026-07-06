"""Harness Mode domain exceptions.

``BudgetExhausted`` is defined in :mod:`vel.harness.budget` (it is raised by the
budget itself); it is re-exported here so callers have a single import surface
for harness control-flow exceptions.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, List

from .budget import BudgetExhausted

if TYPE_CHECKING:  # avoid import cycles at runtime
    from .approvals import ApprovalRequest
    from .checkpoint import RunCheckpoint


class HarnessError(Exception):
    """Base class for Harness Mode errors."""


class SuspendRun(HarnessError):
    """Raised to suspend a durable run (e.g. awaiting human approval).

    Carries the checkpoint to persist and the approval requests that triggered
    the suspension so the controller can persist state and emit the right
    ``data-harness-*`` events before returning control to the caller.
    """

    def __init__(
        self,
        checkpoint: 'RunCheckpoint',
        approval_requests: 'List[ApprovalRequest]',
    ) -> None:
        super().__init__('run suspended awaiting approval')
        self.checkpoint = checkpoint
        self.approval_requests = approval_requests


class CompactionError(HarnessError):
    """Raised when context compaction fails or would corrupt message format."""


class SandboxError(HarnessError):
    """Raised when a sandbox operation fails or the provider is unavailable."""


__all__ = [
    'HarnessError',
    'SuspendRun',
    'BudgetExhausted',
    'CompactionError',
    'SandboxError',
]
