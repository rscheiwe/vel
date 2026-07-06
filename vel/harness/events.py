from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional

from vel.events import DataEvent


@dataclass
class HarnessRunStartedEvent(DataEvent):
    type: str = 'data-harness-run-started'
    run_id: str = ''
    agent_id: Optional[str] = None
    durable: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            'type': self.type,
            'data': {
                'run_id': self.run_id,
                'agent_id': self.agent_id,
                'durable': self.durable,
            },
            'transient': True,
        }


@dataclass
class HarnessStepEvent(DataEvent):
    type: str = 'data-harness-step'
    step: int = 0
    budget: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            'type': self.type,
            'data': {
                'step': self.step,
                'budget': self.budget,
            },
            'transient': True,
        }


@dataclass
class HarnessCompactionEvent(DataEvent):
    type: str = 'data-harness-compaction'
    before_tokens: int = 0
    after_tokens: int = 0
    strategy: str = ''
    removed: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            'type': self.type,
            'data': {
                'before_tokens': self.before_tokens,
                'after_tokens': self.after_tokens,
                'strategy': self.strategy,
                'removed': self.removed,
            },
            'transient': False,
        }


@dataclass
class HarnessApprovalRequiredEvent(DataEvent):
    type: str = 'data-harness-approval-required'
    approval_id: str = ''
    run_id: str = ''
    tool_call_id: str = ''
    tool_name: str = ''
    reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            'type': self.type,
            'data': {
                'approval_id': self.approval_id,
                'run_id': self.run_id,
                'tool_call_id': self.tool_call_id,
                'tool_name': self.tool_name,
                'reason': self.reason,
            },
            'transient': False,
        }


@dataclass
class HarnessSuspendedEvent(DataEvent):
    type: str = 'data-harness-suspended'
    run_id: str = ''
    reason: Literal['approval'] = 'approval'

    def to_dict(self) -> Dict[str, Any]:
        return {
            'type': self.type,
            'data': {
                'run_id': self.run_id,
                'reason': self.reason,
            },
            'transient': False,
        }


@dataclass
class HarnessResumedEvent(DataEvent):
    type: str = 'data-harness-resumed'
    run_id: str = ''

    def to_dict(self) -> Dict[str, Any]:
        return {
            'type': self.type,
            'data': {'run_id': self.run_id},
            'transient': True,
        }


@dataclass
class HarnessRecoveredEvent(DataEvent):
    type: str = 'data-harness-recovered'
    run_id: str = ''
    # Number of already-completed tool calls skipped on recovery.
    skipped_tools: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            'type': self.type,
            'data': {
                'run_id': self.run_id,
                'skipped_tools': self.skipped_tools,
            },
            'transient': True,
        }


@dataclass
class HarnessBudgetExhaustedEvent(DataEvent):
    type: str = 'data-harness-budget-exhausted'
    reason: str = ''

    def to_dict(self) -> Dict[str, Any]:
        return {
            'type': self.type,
            'data': {'reason': self.reason},
            'transient': True,
        }


@dataclass
class HarnessSandboxEvent(DataEvent):
    type: str = 'data-harness-sandbox'
    event: Literal['created', 'connected', 'closed'] = 'created'
    sandbox_ref: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            'type': self.type,
            'data': {
                'event': self.event,
                'sandbox_ref': self.sandbox_ref,
            },
            'transient': True,
        }


@dataclass
class HarnessRunFinishedEvent(DataEvent):
    type: str = 'data-harness-run-finished'
    run_id: str = ''
    status: str = ''
    usage: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            'type': self.type,
            'data': {
                'run_id': self.run_id,
                'status': self.status,
                'usage': self.usage,
            },
            'transient': False,
        }


__all__ = [
    'HarnessApprovalRequiredEvent',
    'HarnessBudgetExhaustedEvent',
    'HarnessCompactionEvent',
    'HarnessRecoveredEvent',
    'HarnessResumedEvent',
    'HarnessRunFinishedEvent',
    'HarnessRunStartedEvent',
    'HarnessSandboxEvent',
    'HarnessStepEvent',
    'HarnessSuspendedEvent',
]
