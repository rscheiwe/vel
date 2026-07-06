from __future__ import annotations

import json
import sqlite3
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, FrozenSet, Iterable, List, Literal, Optional, Protocol, Union

from .config import ApprovalConfig

# An approval policy returns one of these. Modelled on eve's ApprovalStatus
# (packages/eve/src/public/definitions/approval.ts):
#   - "approved"        -> auto-approve, no human prompt
#   - "denied"          -> auto-deny, no human prompt
#   - "user-approval"   -> require a human decision (durable suspend)
#   - "not-applicable"  -> no approval needed (decisive)
#   - True              -> "user-approval"
#   - False             -> "not-applicable"
#   - None              -> the policy abstains; fall through to the static gate
ApprovalStatus = Union[
    Literal['approved', 'denied', 'user-approval', 'not-applicable'], bool, None
]

# The four normalized outcomes evaluate() resolves to.
NormalizedStatus = Literal['approved', 'denied', 'user-approval', 'not-applicable']


@dataclass
class ApprovalRequest:
    approval_id: str
    run_id: str
    tool_call_id: str
    tool_name: str
    args: Dict[str, Any]
    reason: Optional[str] = None
    created_at: float = field(default_factory=time.time)


@dataclass
class ApprovalDecision:
    approval_id: str
    decision: Literal['approve', 'reject']
    note: Optional[str] = None
    decided_by: Optional[str] = None
    decided_at: float = field(default_factory=time.time)


@dataclass
class ApprovalContext:
    """Context passed to an ``ApprovalConfig.policy`` predicate.

    Mirrors eve's ``ApprovalContext``: the policy decides from the tool name,
    the actual input, and the set of tools already approved this session
    (``approved_tools``) so it can implement input-aware and
    approve-once-per-session logic.
    """

    tool_name: str
    tool_input: Dict[str, Any]
    tool_call_id: str
    run_id: str
    session_id: Optional[str]
    step: int
    approved_tools: FrozenSet[str]
    requires_confirmation: bool


def normalize_approval_status(status: ApprovalStatus) -> NormalizedStatus:
    """Coerce a policy return value to one of the four normalized outcomes.

    ``None`` is treated as ``not-applicable`` here; callers that need to
    distinguish "policy abstained" from "policy said not-applicable" must check
    for ``None`` *before* calling this.
    """
    if status is True:
        return 'user-approval'
    if status is False or status is None:
        return 'not-applicable'
    if status in ('approved', 'denied', 'user-approval', 'not-applicable'):
        return status  # type: ignore[return-value]
    raise ValueError(f"invalid approval status: {status!r}")


class ApprovalStore(Protocol):
    async def open(self, requests: List[ApprovalRequest]) -> None: ...
    async def record(self, decision: ApprovalDecision) -> None: ...
    async def get_pending(self, run_id: str) -> List[ApprovalRequest]: ...
    async def get_decision(self, tool_call_id: str) -> Optional[ApprovalDecision]: ...


class InMemoryApprovalStore:
    def __init__(self):
        self.requests: Dict[str, ApprovalRequest] = {}
        self.decisions: Dict[str, ApprovalDecision] = {}

    async def open(self, requests: List[ApprovalRequest]) -> None:
        for request in requests:
            self.requests[request.approval_id] = request

    async def record(self, decision: ApprovalDecision) -> None:
        self.decisions[decision.approval_id] = decision

    async def get_pending(self, run_id: str) -> List[ApprovalRequest]:
        return [
            request
            for request in self.requests.values()
            if request.run_id == run_id and request.approval_id not in self.decisions
        ]

    async def get_decision(self, tool_call_id: str) -> Optional[ApprovalDecision]:
        for approval_id, decision in self.decisions.items():
            request = self.requests.get(approval_id)
            if request and request.tool_call_id == tool_call_id:
                return decision
        return None


class SQLiteApprovalStore:
    def __init__(self, db_path: str = ".vel/vel.db"):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.db_path = db_path
        self.db = sqlite3.connect(db_path)
        self.db.execute("PRAGMA journal_mode=WAL;")
        self.db.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        self.db.executescript(
            """
            CREATE TABLE IF NOT EXISTS vel_approvals (
                approval_id TEXT PRIMARY KEY,
                run_id TEXT NOT NULL,
                tool_call_id TEXT NOT NULL,
                tool_name TEXT NOT NULL,
                args TEXT NOT NULL,
                reason TEXT,
                status TEXT NOT NULL DEFAULT 'pending',
                decision TEXT,
                created_at REAL DEFAULT (strftime('%s','now')),
                decided_at REAL
            );
            CREATE INDEX IF NOT EXISTS idx_appr_run ON vel_approvals(run_id);
            CREATE INDEX IF NOT EXISTS idx_appr_tool_call ON vel_approvals(tool_call_id);
            """
        )
        self.db.commit()

    async def open(self, requests: List[ApprovalRequest]) -> None:
        for request in requests:
            self.db.execute(
                """
                INSERT INTO vel_approvals(
                    approval_id, run_id, tool_call_id, tool_name, args, reason,
                    status, created_at)
                VALUES(?,?,?,?,?,?,?,?)
                ON CONFLICT(approval_id) DO UPDATE SET
                    run_id=excluded.run_id,
                    tool_call_id=excluded.tool_call_id,
                    tool_name=excluded.tool_name,
                    args=excluded.args,
                    reason=excluded.reason,
                    status='pending',
                    decision=NULL,
                    created_at=excluded.created_at,
                    decided_at=NULL
                """,
                (
                    request.approval_id,
                    request.run_id,
                    request.tool_call_id,
                    request.tool_name,
                    json.dumps(request.args),
                    request.reason,
                    'pending',
                    request.created_at,
                ),
            )
        self.db.commit()

    async def record(self, decision: ApprovalDecision) -> None:
        status = 'approved' if decision.decision == 'approve' else 'rejected'
        self.db.execute(
            """
            UPDATE vel_approvals
            SET status=?, decision=?, decided_at=?
            WHERE approval_id=?
            """,
            (
                status,
                json.dumps(_decision_to_dict(decision)),
                decision.decided_at,
                decision.approval_id,
            ),
        )
        self.db.commit()

    async def get_pending(self, run_id: str) -> List[ApprovalRequest]:
        rows = self.db.execute(
            """
            SELECT approval_id, run_id, tool_call_id, tool_name, args, reason, created_at
            FROM vel_approvals
            WHERE run_id=? AND status='pending'
            ORDER BY created_at, approval_id
            """,
            (run_id,),
        ).fetchall()
        return [_request_from_row(row) for row in rows]

    async def get_decision(self, tool_call_id: str) -> Optional[ApprovalDecision]:
        row = self.db.execute(
            """
            SELECT decision
            FROM vel_approvals
            WHERE tool_call_id=? AND status IN ('approved', 'rejected')
            ORDER BY decided_at DESC
            LIMIT 1
            """,
            (tool_call_id,),
        ).fetchone()
        if not row or not row['decision']:
            return None
        return _decision_from_dict(json.loads(row['decision']))

    def close(self) -> None:
        self.db.close()


class ApprovalGate:
    def __init__(
        self,
        config: Optional[ApprovalConfig] = None,
        store: Optional[ApprovalStore] = None,
    ):
        self.config = config or ApprovalConfig()
        self.store = store or InMemoryApprovalStore()

    def requires_approval(self, tool: Any, name: str) -> bool:
        """Static gate: does this tool require approval by name/flag alone?

        Retained for callers that only need a bool. The richer, context-aware
        decision (policy predicate + session memory) is :meth:`evaluate`.
        """
        if not self.config.enabled:
            return False
        if name in self.config.require_for_tools:
            return True
        return bool(
            self.config.require_for_confirmation_flag
            and getattr(tool, 'requires_confirmation', False)
        )

    def evaluate(self, ctx: ApprovalContext) -> NormalizedStatus:
        """Resolve the approval outcome for a single tool call.

        Precedence:
          1. session memory — if ``remember_approvals`` and the tool was already
             approved this session, it is ``not-applicable`` (no re-prompt);
          2. the ``policy`` predicate — if set and it returns a non-``None``
             (decisive) status, that wins;
          3. the static gate — ``require_for_tools`` / the confirmation flag →
             ``user-approval`` else ``not-applicable``.
        """
        if not self.config.enabled:
            return 'not-applicable'

        if self.config.remember_approvals and ctx.tool_name in ctx.approved_tools:
            return 'not-applicable'

        if self.config.policy is not None:
            status = self.config.policy(ctx)
            if status is not None:  # None == abstain -> fall through
                return normalize_approval_status(status)

        static_required = ctx.tool_name in self.config.require_for_tools or (
            self.config.require_for_confirmation_flag and ctx.requires_confirmation
        )
        return 'user-approval' if static_required else 'not-applicable'

    def build_request(
        self,
        *,
        run_id: str,
        tool_call_id: str,
        tool_name: str,
        args: Dict[str, Any],
        reason: Optional[str] = None,
        approval_id: Optional[str] = None,
    ) -> ApprovalRequest:
        return ApprovalRequest(
            approval_id=approval_id or str(uuid.uuid4()),
            run_id=run_id,
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            args=args,
            reason=reason,
        )

    async def open(self, requests: Iterable[ApprovalRequest]) -> List[ApprovalRequest]:
        opened = list(requests)
        await self.store.open(opened)
        return opened

    async def record(self, decision: ApprovalDecision) -> None:
        await self.store.record(decision)

    async def get_pending(self, run_id: str) -> List[ApprovalRequest]:
        pending = await self.store.get_pending(run_id)
        if self.config.timeout_seconds is None:
            return pending

        now = time.time()
        active: List[ApprovalRequest] = []
        for request in pending:
            timed_out = now - request.created_at >= self.config.timeout_seconds
            if not timed_out:
                active.append(request)
                continue
            if self.config.on_timeout == 'approve':
                await self.record(ApprovalDecision(request.approval_id, 'approve'))
            elif self.config.on_timeout == 'deny':
                await self.record(ApprovalDecision(request.approval_id, 'reject'))
            else:
                active.append(request)
        return active

    async def get_decision(self, tool_call_id: str) -> Optional[ApprovalDecision]:
        return await self.store.get_decision(tool_call_id)


def _request_from_row(row: sqlite3.Row) -> ApprovalRequest:
    return ApprovalRequest(
        approval_id=row['approval_id'],
        run_id=row['run_id'],
        tool_call_id=row['tool_call_id'],
        tool_name=row['tool_name'],
        args=json.loads(row['args']),
        reason=row['reason'],
        created_at=row['created_at'],
    )


def _decision_to_dict(decision: ApprovalDecision) -> Dict[str, Any]:
    return {
        'approval_id': decision.approval_id,
        'decision': decision.decision,
        'note': decision.note,
        'decided_by': decision.decided_by,
        'decided_at': decision.decided_at,
    }


def _decision_from_dict(data: Dict[str, Any]) -> ApprovalDecision:
    return ApprovalDecision(
        approval_id=data['approval_id'],
        decision=data['decision'],
        note=data.get('note'),
        decided_by=data.get('decided_by'),
        decided_at=data.get('decided_at', time.time()),
    )


__all__ = [
    'ApprovalContext',
    'ApprovalDecision',
    'ApprovalGate',
    'ApprovalRequest',
    'ApprovalStatus',
    'ApprovalStore',
    'InMemoryApprovalStore',
    'SQLiteApprovalStore',
    'normalize_approval_status',
]
