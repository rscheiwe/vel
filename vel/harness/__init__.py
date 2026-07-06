"""Vel Harness Mode — opt-in durable long-horizon execution.

Default-off bolt-on (mirrors the RLM/Thinking subsystems): with no
``HarnessConfig`` supplied, nothing here activates and every existing Vel code
path runs unchanged.

Only pure-Python modules are imported eagerly here. Optional, heavyweight
adapters (sandbox providers such as e2b) are imported lazily at use-site so
``import vel`` never requires harness extras to be installed.
"""
from __future__ import annotations

from .config import (
    ApprovalConfig,
    CompactionConfig,
    HarnessBudgetConfig,
    HarnessConfig,
    SandboxConfig,
    SkillRef,
)
from .budget import BudgetExhausted, HarnessBudget
from .exceptions import (
    CompactionError,
    HarnessError,
    SandboxError,
    SuspendRun,
)
from .events import (
    HarnessApprovalRequiredEvent,
    HarnessBudgetExhaustedEvent,
    HarnessCompactionEvent,
    HarnessRecoveredEvent,
    HarnessResumedEvent,
    HarnessRunFinishedEvent,
    HarnessRunStartedEvent,
    HarnessSandboxEvent,
    HarnessStepEvent,
    HarnessSuspendedEvent,
)
from .approvals import (
    ApprovalContext,
    ApprovalDecision,
    ApprovalGate,
    ApprovalRequest,
    ApprovalStatus,
    normalize_approval_status,
)
from .checkpoint import ApprovalMemoryStore, CheckpointStore, RunCheckpoint
from .pubsub import InProcessPubSub, PubSub, RedisPubSub
from .runner import EventLogStore, PostgresEventLogStore, RunManager, SQLiteEventLogStore
from .sandbox import ExecResult, SandboxProvider, SandboxSession, build_sandbox_tools
from .skills import Skill, SkillRegistry, default_registry, resolve_skills
from .controller import HarnessController
from .channels import Channel, CLIChannel, SlackChannel, format_event

__all__ = [
    # config
    'HarnessConfig',
    'CompactionConfig',
    'ApprovalConfig',
    'SandboxConfig',
    'HarnessBudgetConfig',
    'SkillRef',
    # budget
    'HarnessBudget',
    'BudgetExhausted',
    # exceptions
    'HarnessError',
    'SuspendRun',
    'CompactionError',
    'SandboxError',
    # events
    'HarnessRunStartedEvent',
    'HarnessStepEvent',
    'HarnessCompactionEvent',
    'HarnessApprovalRequiredEvent',
    'HarnessSuspendedEvent',
    'HarnessResumedEvent',
    'HarnessRecoveredEvent',
    'HarnessBudgetExhaustedEvent',
    'HarnessSandboxEvent',
    'HarnessRunFinishedEvent',
    # approvals
    'ApprovalGate',
    'ApprovalRequest',
    'ApprovalDecision',
    'ApprovalContext',
    'ApprovalStatus',
    'normalize_approval_status',
    # checkpoint
    'CheckpointStore',
    'RunCheckpoint',
    'ApprovalMemoryStore',
    # runner
    'EventLogStore',
    'InProcessPubSub',
    'PostgresEventLogStore',
    'PubSub',
    'RedisPubSub',
    'RunManager',
    'SQLiteEventLogStore',
    # sandbox
    'ExecResult',
    'SandboxProvider',
    'SandboxSession',
    'build_sandbox_tools',
    # skills
    'Skill',
    'SkillRegistry',
    'resolve_skills',
    'default_registry',
    # controller
    'HarnessController',
    # channels
    'Channel',
    'CLIChannel',
    'SlackChannel',
    'format_event',
]
