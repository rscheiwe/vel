from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Literal, Optional

if TYPE_CHECKING:
    # Imported lazily for typing only — approvals.py imports this module, so a
    # runtime import here would be circular.
    from .approvals import ApprovalContext, ApprovalStatus


@dataclass
class CompactionConfig:
    enabled: bool = True
    trigger_token_ratio: float = 0.75
    keep_last_messages: int = 6
    strategy: Literal['summarize', 'reduce', 'memory_offload'] = 'summarize'
    summarizer_model: Optional[Dict[str, Any]] = None
    summary_max_tokens: int = 1024


@dataclass
class ApprovalConfig:
    enabled: bool = True
    require_for_tools: List[str] = field(default_factory=list)
    require_for_confirmation_flag: bool = True
    mode: Literal['durable', 'inline'] = 'durable'
    timeout_seconds: Optional[int] = None
    on_timeout: Literal['deny', 'approve', 'fail'] = 'deny'
    # Optional input-aware predicate ``(ApprovalContext) -> ApprovalStatus``.
    # Returning None abstains (fall through to the static gate); a concrete
    # status ("approved"/"denied"/"user-approval"/"not-applicable"/bool) is
    # decisive. See vel/harness/approvals.py.
    policy: Optional[Callable[['ApprovalContext'], 'ApprovalStatus']] = None
    # Approve-once-per-session: once a tool is approved (by a human or an
    # "approved" policy result) it is not re-prompted for the rest of the
    # session. Eve-like default; harness-only, so it never affects the
    # non-harness path.
    remember_approvals: bool = True


@dataclass
class SandboxConfig:
    enabled: bool = False
    provider: Literal['e2b', 'local_subprocess', 'none'] = 'none'
    lifecycle: Literal['per_run', 'per_session', 'persistent'] = 'per_session'
    image: Optional[str] = None
    timeout_seconds: int = 300
    workdir: str = '/workspace'
    tools: List[Literal['read', 'write', 'edit', 'list', 'bash', 'python']] = field(default_factory=list)
    env: Dict[str, str] = field(default_factory=dict)
    provider_options: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HarnessBudgetConfig:
    max_steps: Optional[int] = 100
    max_tokens: Optional[int] = None
    max_cost_usd: Optional[float] = None
    max_wallclock_seconds: Optional[int] = None


@dataclass
class SkillRef:
    name: str
    skill: Optional['Skill'] = None


@dataclass
class HarnessConfig:
    enabled: bool = False
    durable: bool = True
    compaction: CompactionConfig = field(default_factory=CompactionConfig)
    approval: ApprovalConfig = field(default_factory=ApprovalConfig)
    sandbox: SandboxConfig = field(default_factory=SandboxConfig)
    budget: HarnessBudgetConfig = field(default_factory=HarnessBudgetConfig)
    skills: List[SkillRef] = field(default_factory=list)
    store_backend: Optional[Literal['sqlite', 'postgres']] = None
    db_path: str = ".vel/vel.db"
    # Crash recovery: when True, persist a running checkpoint after *each* tool
    # result (not only once per step) so a mid-step crash can resume without
    # re-running already-completed tools. Default False = one checkpoint per step
    # (unchanged behavior). See Agent.recover / RunManager.recover.
    checkpoint_each_tool: bool = False

    def __post_init__(self):
        if isinstance(self.compaction, dict):
            self.compaction = CompactionConfig(**self.compaction)
        if isinstance(self.approval, dict):
            self.approval = ApprovalConfig(**self.approval)
        if isinstance(self.sandbox, dict):
            self.sandbox = SandboxConfig(**self.sandbox)
        if isinstance(self.budget, dict):
            self.budget = HarnessBudgetConfig(**self.budget)
        self.skills = [
            SkillRef(**skill) if isinstance(skill, dict) else skill
            for skill in self.skills
        ]


__all__ = [
    'ApprovalConfig',
    'CompactionConfig',
    'HarnessBudgetConfig',
    'HarnessConfig',
    'SandboxConfig',
    'SkillRef',
]
