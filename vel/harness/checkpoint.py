"""Durable run checkpoints for Harness Mode (M3).

A :class:`RunCheckpoint` is the serializable snapshot of everything needed to
suspend a run and resume it later — possibly in a fresh process / ``Agent``
instance. :class:`CheckpointStore` persists it.

The store is SQLite-backed by default (mirroring ``vel/memory/fact_store.py``:
WAL mode, self-initializing schema, epoch-real timestamps) so tests and the
``transient`` session backend work with zero external services. The matching
Postgres DDL ships in ``alembic/versions/0002_harness_mode.py`` for the
``persistent`` backend; both use ``CREATE TABLE IF NOT EXISTS`` and are additive
per the backwards-compatibility contract (§8).
"""
from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from .approvals import ApprovalRequest

Status = str  # 'running' | 'suspended' | 'completed' | 'failed' | 'cancelled'


@dataclass
class RunCheckpoint:
    """Serializable snapshot of an in-flight (or suspended) harness run."""

    run_id: str
    agent_id: str
    session_id: Optional[str]
    status: Status
    step: int
    messages: List[Dict[str, Any]]
    pending_approvals: List[ApprovalRequest] = field(default_factory=list)
    # Full tool_calls of the suspended step (gated + non-gated); re-executed on
    # resume. Suspension happens before any tool runs, so none have results yet.
    pending_tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    # Tool-call ids of the in-flight step whose results are already committed to
    # ``messages`` (per-tool crash-recovery checkpointing). On recover, these are
    # skipped instead of re-executed. Empty for step-boundary checkpoints.
    completed_tool_calls: List[str] = field(default_factory=list)
    budget_state: Dict[str, Any] = field(default_factory=dict)
    sandbox_ref: Optional[str] = None
    config_hash: str = ""
    # Extended-thinking durability: when set, this is a suspended REFLECTION run
    # ({scratch_id, cursor, state, approved_tools}); resume continues the phase
    # state machine instead of the step loop. None for ordinary step-loop runs.
    reflection: Optional[Dict[str, Any]] = None
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    # ---------------------------------------------------------- serialization
    def snapshot(self) -> Dict[str, Any]:
        """The JSON-serializable payload stored in the ``snapshot`` column."""
        return {
            'messages': self.messages,
            'pending_approvals': [_approval_to_dict(a) for a in self.pending_approvals],
            'pending_tool_calls': self.pending_tool_calls,
            'completed_tool_calls': self.completed_tool_calls,
            'budget_state': self.budget_state,
            'sandbox_ref': self.sandbox_ref,
            'reflection': self.reflection,
        }

    @classmethod
    def from_row(
        cls,
        *,
        run_id: str,
        agent_id: str,
        session_id: Optional[str],
        status: Status,
        step: int,
        snapshot: Dict[str, Any],
        config_hash: str,
        created_at: float,
        updated_at: float,
    ) -> 'RunCheckpoint':
        return cls(
            run_id=run_id,
            agent_id=agent_id,
            session_id=session_id,
            status=status,
            step=step,
            messages=snapshot.get('messages', []),
            pending_approvals=[
                _approval_from_dict(a) for a in snapshot.get('pending_approvals', [])
            ],
            pending_tool_calls=snapshot.get('pending_tool_calls', []),
            completed_tool_calls=snapshot.get('completed_tool_calls', []),
            budget_state=snapshot.get('budget_state', {}),
            sandbox_ref=snapshot.get('sandbox_ref'),
            reflection=snapshot.get('reflection'),
            config_hash=config_hash,
            created_at=created_at,
            updated_at=updated_at,
        )


def _approval_to_dict(a: ApprovalRequest) -> Dict[str, Any]:
    return {
        'approval_id': a.approval_id,
        'run_id': a.run_id,
        'tool_call_id': a.tool_call_id,
        'tool_name': a.tool_name,
        'args': a.args,
        'reason': a.reason,
        'created_at': a.created_at,
    }


def _approval_from_dict(d: Dict[str, Any]) -> ApprovalRequest:
    return ApprovalRequest(
        approval_id=d['approval_id'],
        run_id=d['run_id'],
        tool_call_id=d['tool_call_id'],
        tool_name=d['tool_name'],
        args=d.get('args', {}),
        reason=d.get('reason'),
        created_at=d.get('created_at', time.time()),
    )


class CheckpointStore:
    """SQLite-backed persistence for :class:`RunCheckpoint`.

    Args:
        db_path: Path to the SQLite database file (created if absent).
    """

    def __init__(self, db_path: str = ".vel/vel.db") -> None:
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.db_path = db_path
        self.db = sqlite3.connect(db_path)
        self.db.execute("PRAGMA journal_mode=WAL;")
        self.db.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        self.db.executescript(
            """
            CREATE TABLE IF NOT EXISTS vel_checkpoints(
              run_id TEXT PRIMARY KEY,
              agent_id TEXT NOT NULL,
              session_id TEXT,
              status TEXT NOT NULL DEFAULT 'running',
              step INTEGER NOT NULL DEFAULT 0,
              snapshot TEXT NOT NULL,
              config_hash TEXT NOT NULL,
              created_at REAL DEFAULT (strftime('%s','now')),
              updated_at REAL DEFAULT (strftime('%s','now'))
            );
            CREATE INDEX IF NOT EXISTS idx_ckpt_session ON vel_checkpoints(session_id);
            CREATE INDEX IF NOT EXISTS idx_ckpt_status ON vel_checkpoints(status);
            """
        )
        self.db.commit()

    def save(self, ckpt: RunCheckpoint) -> None:
        """Insert or update a checkpoint (keyed by ``run_id``)."""
        ckpt.updated_at = time.time()
        self.db.execute(
            """
            INSERT INTO vel_checkpoints(
                run_id, agent_id, session_id, status, step, snapshot, config_hash,
                created_at, updated_at)
            VALUES(?,?,?,?,?,?,?,?,?)
            ON CONFLICT(run_id) DO UPDATE SET
                agent_id=excluded.agent_id,
                session_id=excluded.session_id,
                status=excluded.status,
                step=excluded.step,
                snapshot=excluded.snapshot,
                config_hash=excluded.config_hash,
                updated_at=excluded.updated_at
            """,
            (
                ckpt.run_id,
                ckpt.agent_id,
                ckpt.session_id,
                ckpt.status,
                ckpt.step,
                json.dumps(ckpt.snapshot()),
                ckpt.config_hash,
                ckpt.created_at,
                ckpt.updated_at,
            ),
        )
        self.db.commit()

    def load(self, run_id: str) -> Optional[RunCheckpoint]:
        row = self.db.execute(
            "SELECT * FROM vel_checkpoints WHERE run_id=?", (run_id,)
        ).fetchone()
        if not row:
            return None
        return RunCheckpoint.from_row(
            run_id=row['run_id'],
            agent_id=row['agent_id'],
            session_id=row['session_id'],
            status=row['status'],
            step=row['step'],
            snapshot=json.loads(row['snapshot']),
            config_hash=row['config_hash'],
            created_at=row['created_at'],
            updated_at=row['updated_at'],
        )

    def set_status(self, run_id: str, status: Status) -> None:
        self.db.execute(
            "UPDATE vel_checkpoints SET status=?, updated_at=? WHERE run_id=?",
            (status, time.time(), run_id),
        )
        self.db.commit()

    def list_suspended(self, session_id: Optional[str] = None) -> List[RunCheckpoint]:
        if session_id is None:
            rows = self.db.execute(
                "SELECT run_id FROM vel_checkpoints WHERE status='suspended'"
            ).fetchall()
        else:
            rows = self.db.execute(
                "SELECT run_id FROM vel_checkpoints WHERE status='suspended' AND session_id=?",
                (session_id,),
            ).fetchall()
        out: List[RunCheckpoint] = []
        for r in rows:
            ckpt = self.load(r['run_id'])
            if ckpt is not None:
                out.append(ckpt)
        return out

    def delete(self, run_id: str) -> None:
        self.db.execute("DELETE FROM vel_checkpoints WHERE run_id=?", (run_id,))
        self.db.commit()

    def close(self) -> None:
        self.db.close()


class SandboxSessionStore:
    """Persists ``session_id -> sandbox_ref`` so ``per_session``/``persistent``
    sandboxes are reused (reconnected) across runs and process restarts.

    Reuse only works if the provider can reconnect by ref across processes (E2B
    can; the local-subprocess provider only within a live process / explicit
    root). An optional idle TTL drops stale refs so a dead remote sandbox is
    recreated rather than reconnected.
    """

    def __init__(self, db_path: str = ".vel/vel.db") -> None:
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(db_path)
        self.db.execute("PRAGMA journal_mode=WAL;")
        self.db.row_factory = sqlite3.Row
        self.db.executescript(
            """
            CREATE TABLE IF NOT EXISTS vel_sandbox_sessions(
              session_id TEXT PRIMARY KEY,
              sandbox_ref TEXT NOT NULL,
              provider TEXT,
              created_at REAL DEFAULT (strftime('%s','now')),
              updated_at REAL DEFAULT (strftime('%s','now'))
            );
            """
        )
        self.db.commit()

    def get(self, session_id: str, *, idle_ttl_seconds: Optional[int] = None) -> Optional[str]:
        row = self.db.execute(
            "SELECT sandbox_ref, updated_at FROM vel_sandbox_sessions WHERE session_id=?",
            (session_id,),
        ).fetchone()
        if not row:
            return None
        if idle_ttl_seconds is not None and (time.time() - row['updated_at']) > idle_ttl_seconds:
            self.delete(session_id)
            return None
        return row['sandbox_ref']

    def put(self, session_id: str, sandbox_ref: str, provider: Optional[str] = None) -> None:
        now = time.time()
        self.db.execute(
            """
            INSERT INTO vel_sandbox_sessions(session_id, sandbox_ref, provider, created_at, updated_at)
            VALUES(?,?,?,?,?)
            ON CONFLICT(session_id) DO UPDATE SET
                sandbox_ref=excluded.sandbox_ref,
                provider=excluded.provider,
                updated_at=excluded.updated_at
            """,
            (session_id, sandbox_ref, provider, now, now),
        )
        self.db.commit()

    def delete(self, session_id: str) -> None:
        self.db.execute("DELETE FROM vel_sandbox_sessions WHERE session_id=?", (session_id,))
        self.db.commit()

    def close(self) -> None:
        self.db.close()


class ApprovalMemoryStore:
    """Persists ``(session_id, tool_name)`` pairs for approve-once-per-session.

    When ``ApprovalConfig.remember_approvals`` is on, a tool approved once (by a
    human decision or an ``"approved"`` policy result) is recorded here so later
    calls to the same tool in the same session are auto-approved without a
    re-prompt — eve's ``approvedTools`` behavior. Session-scoped and durable, so
    it survives suspend/resume and process restarts.
    """

    def __init__(self, db_path: str = ".vel/vel.db") -> None:
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(db_path)
        self.db.execute("PRAGMA journal_mode=WAL;")
        self.db.row_factory = sqlite3.Row
        self.db.executescript(
            """
            CREATE TABLE IF NOT EXISTS vel_approved_tools(
              session_id TEXT NOT NULL,
              tool_name TEXT NOT NULL,
              created_at REAL DEFAULT (strftime('%s','now')),
              PRIMARY KEY(session_id, tool_name)
            );
            """
        )
        self.db.commit()

    def add(self, session_id: str, tool_name: str) -> None:
        self.db.execute(
            "INSERT OR IGNORE INTO vel_approved_tools(session_id, tool_name, created_at) "
            "VALUES(?,?,?)",
            (session_id, tool_name, time.time()),
        )
        self.db.commit()

    def get(self, session_id: str) -> set:
        rows = self.db.execute(
            "SELECT tool_name FROM vel_approved_tools WHERE session_id=?",
            (session_id,),
        ).fetchall()
        return {row['tool_name'] for row in rows}

    def close(self) -> None:
        self.db.close()


__all__ = [
    'RunCheckpoint',
    'CheckpointStore',
    'SandboxSessionStore',
    'ApprovalMemoryStore',
]
