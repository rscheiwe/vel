# vel/memory/trajectory_store.py
"""
TrajectoryStore: Records agent execution traces for ReasoningBank Phase 2.

Enables automatic learning by storing:
- Full conversation history (messages)
- Tool calls with inputs/outputs
- Timing information
- Links to strategies used
- Evaluation results from LLM-as-Judge
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from pathlib import Path
import sqlite3
import json
from time import time


@dataclass
class ToolCallRecord:
    """Records a single tool invocation within a trajectory."""
    step_index: int
    tool_name: str
    tool_call_id: Optional[str]
    input: Dict[str, Any]
    output: Optional[Any]
    error: Optional[str]
    duration_ms: Optional[int]


@dataclass
class Trajectory:
    """Represents a complete agent execution trace."""
    id: Optional[int]
    run_id: str
    session_id: Optional[str]
    agent_id: Optional[str]
    signature: Dict[str, Any]

    # Trajectory data
    input_message: str
    messages: List[Dict[str, Any]]
    tool_calls: List[ToolCallRecord]
    final_answer: Optional[str]
    error: Optional[str]

    # Timing
    started_at: float
    finished_at: Optional[float]
    duration_ms: Optional[int]
    step_count: int

    # Strategy linkage
    strategies_used: List[int]

    # Evaluation state (Phase 2.2)
    evaluated: bool = False
    success: Optional[bool] = None
    evaluation_confidence: Optional[float] = None
    evaluation_notes: Optional[str] = None

    # Extraction state (Phase 2.3)
    strategies_extracted: bool = False

    @classmethod
    def from_row(cls, row: sqlite3.Row, tool_calls: Optional[List[ToolCallRecord]] = None) -> 'Trajectory':
        """Create Trajectory from database row."""
        return cls(
            id=row["id"],
            run_id=row["run_id"],
            session_id=row["session_id"],
            agent_id=row["agent_id"],
            signature=json.loads(row["signature_json"] or "{}"),
            input_message=row["input_message"],
            messages=json.loads(row["messages_json"] or "[]"),
            tool_calls=tool_calls or [],
            final_answer=row["final_answer"],
            error=row["error"],
            started_at=row["started_at"],
            finished_at=row["finished_at"],
            duration_ms=row["duration_ms"],
            step_count=row["step_count"],
            strategies_used=json.loads(row["strategies_used_json"] or "[]"),
            evaluated=bool(row["evaluated"]),
            success=row["success"] if row["evaluated"] else None,
            evaluation_confidence=row["evaluation_confidence"],
            evaluation_notes=row["evaluation_notes"],
            strategies_extracted=bool(row["strategies_extracted"]),
        )


class TrajectoryStore:
    """
    Stores and retrieves agent execution trajectories.

    Part of ReasoningBank Phase 2 - enables automatic learning from
    agent execution history.

    Usage:
        ```python
        store = TrajectoryStore(".vel/vel.db")

        # Start recording
        store.start_trajectory(
            run_id="run-123",
            signature={"intent": "planning", "domain": "api"},
            input_message="Help me design an API",
            agent_id="my-agent"
        )

        # Record tool calls during execution
        store.record_tool_call(
            run_id="run-123",
            step_index=0,
            tool_name="search_docs",
            input={"query": "REST API best practices"},
            output={"results": [...]}
        )

        # Finish recording
        store.finish_trajectory(
            run_id="run-123",
            messages=[...],
            final_answer="Here's the API design..."
        )
        ```
    """

    def __init__(self, db_path: str):
        """Initialize TrajectoryStore with SQLite database."""
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(db_path, check_same_thread=False)
        self.db.execute("PRAGMA journal_mode=WAL;")
        self.db.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self):
        """Initialize database schema."""
        self.db.executescript("""
        CREATE TABLE IF NOT EXISTS rb_trajectories (
            id INTEGER PRIMARY KEY,
            run_id TEXT NOT NULL UNIQUE,
            session_id TEXT,
            agent_id TEXT,
            signature_json TEXT NOT NULL,
            input_message TEXT NOT NULL,
            messages_json TEXT NOT NULL DEFAULT '[]',
            tool_calls_json TEXT DEFAULT '[]',
            final_answer TEXT,
            error TEXT,
            started_at REAL NOT NULL,
            finished_at REAL,
            duration_ms INTEGER,
            step_count INTEGER DEFAULT 0,
            strategies_used_json TEXT DEFAULT '[]',
            evaluated BOOLEAN DEFAULT 0,
            success BOOLEAN,
            evaluation_confidence REAL,
            evaluation_notes TEXT,
            strategies_extracted BOOLEAN DEFAULT 0,
            created_at REAL DEFAULT (strftime('%s','now'))
        );

        CREATE INDEX IF NOT EXISTS idx_traj_run_id ON rb_trajectories(run_id);
        CREATE INDEX IF NOT EXISTS idx_traj_session ON rb_trajectories(session_id);
        CREATE INDEX IF NOT EXISTS idx_traj_agent ON rb_trajectories(agent_id);
        CREATE INDEX IF NOT EXISTS idx_traj_eval ON rb_trajectories(evaluated, success);
        CREATE INDEX IF NOT EXISTS idx_traj_extracted ON rb_trajectories(strategies_extracted);
        CREATE INDEX IF NOT EXISTS idx_traj_created ON rb_trajectories(created_at);

        CREATE TABLE IF NOT EXISTS rb_trajectory_tool_calls (
            id INTEGER PRIMARY KEY,
            trajectory_id INTEGER NOT NULL,
            step_index INTEGER NOT NULL,
            tool_name TEXT NOT NULL,
            tool_call_id TEXT,
            input_json TEXT NOT NULL,
            output_json TEXT,
            error TEXT,
            duration_ms INTEGER,
            created_at REAL DEFAULT (strftime('%s','now')),
            FOREIGN KEY(trajectory_id) REFERENCES rb_trajectories(id) ON DELETE CASCADE
        );

        CREATE INDEX IF NOT EXISTS idx_tool_calls_traj ON rb_trajectory_tool_calls(trajectory_id);
        CREATE INDEX IF NOT EXISTS idx_tool_calls_name ON rb_trajectory_tool_calls(tool_name);
        """)
        self.db.commit()

    # ----- Recording API -----

    def start_trajectory(
        self,
        run_id: str,
        signature: Dict[str, Any],
        input_message: str,
        agent_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> int:
        """
        Begin recording a new trajectory.

        Args:
            run_id: Unique run identifier
            signature: Task signature for retrieval matching
            input_message: Original user input
            agent_id: Optional agent identifier
            session_id: Optional session identifier

        Returns:
            Trajectory ID
        """
        cur = self.db.cursor()
        cur.execute("""
            INSERT INTO rb_trajectories(
                run_id, session_id, agent_id, signature_json,
                input_message, started_at
            ) VALUES (?, ?, ?, ?, ?, ?)
        """, (
            run_id,
            session_id,
            agent_id,
            json.dumps(signature, sort_keys=True),
            input_message,
            time()
        ))
        self.db.commit()
        return cur.lastrowid  # type: ignore

    def record_tool_call(
        self,
        run_id: str,
        step_index: int,
        tool_name: str,
        input: Dict[str, Any],
        output: Optional[Any] = None,
        error: Optional[str] = None,
        tool_call_id: Optional[str] = None,
        duration_ms: Optional[int] = None,
    ) -> None:
        """
        Record a tool call within the trajectory.

        Args:
            run_id: Run identifier
            step_index: Order in execution sequence
            tool_name: Name of the tool called
            input: Tool input arguments
            output: Tool output (if successful)
            error: Error message (if failed)
            tool_call_id: Provider-assigned tool call ID
            duration_ms: Execution time in milliseconds
        """
        # Get trajectory ID
        row = self.db.execute(
            "SELECT id FROM rb_trajectories WHERE run_id = ?",
            (run_id,)
        ).fetchone()

        if not row:
            return  # Trajectory not found, silent fail

        trajectory_id = row["id"]

        self.db.execute("""
            INSERT INTO rb_trajectory_tool_calls(
                trajectory_id, step_index, tool_name, tool_call_id,
                input_json, output_json, error, duration_ms
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            trajectory_id,
            step_index,
            tool_name,
            tool_call_id,
            json.dumps(input),
            json.dumps(output) if output is not None else None,
            error,
            duration_ms
        ))
        self.db.commit()

    def finish_trajectory(
        self,
        run_id: str,
        messages: List[Dict[str, Any]],
        final_answer: Optional[str] = None,
        error: Optional[str] = None,
        strategies_used: Optional[List[int]] = None,
        step_count: int = 0,
    ) -> None:
        """
        Complete trajectory recording.

        Args:
            run_id: Run identifier
            messages: Full conversation history
            final_answer: Final LLM response
            error: Error message if failed
            strategies_used: List of strategy IDs used
            step_count: Number of execution steps
        """
        finished_at = time()

        # Get started_at to calculate duration
        row = self.db.execute(
            "SELECT started_at FROM rb_trajectories WHERE run_id = ?",
            (run_id,)
        ).fetchone()

        if not row:
            return  # Trajectory not found

        started_at = row["started_at"]
        duration_ms = int((finished_at - started_at) * 1000)

        self.db.execute("""
            UPDATE rb_trajectories SET
                messages_json = ?,
                final_answer = ?,
                error = ?,
                strategies_used_json = ?,
                step_count = ?,
                finished_at = ?,
                duration_ms = ?
            WHERE run_id = ?
        """, (
            json.dumps(messages),
            final_answer,
            error,
            json.dumps(strategies_used or []),
            step_count,
            finished_at,
            duration_ms,
            run_id
        ))
        self.db.commit()

    # ----- Retrieval API -----

    def get_trajectory(self, run_id: str) -> Optional[Trajectory]:
        """
        Get a single trajectory by run_id.

        Args:
            run_id: Run identifier

        Returns:
            Trajectory or None if not found
        """
        row = self.db.execute(
            "SELECT * FROM rb_trajectories WHERE run_id = ?",
            (run_id,)
        ).fetchone()

        if not row:
            return None

        tool_calls = self.get_tool_calls(row["id"])
        return Trajectory.from_row(row, tool_calls)

    def get_trajectories(
        self,
        session_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        signature: Optional[Dict[str, Any]] = None,
        success: Optional[bool] = None,
        evaluated: Optional[bool] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Trajectory]:
        """
        Query trajectories with filters.

        Args:
            session_id: Filter by session
            agent_id: Filter by agent
            signature: Filter by signature (exact match)
            success: Filter by evaluation result
            evaluated: Filter by evaluation status
            limit: Maximum results
            offset: Skip first N results

        Returns:
            List of matching trajectories
        """
        conditions = []
        params: List[Any] = []

        if session_id is not None:
            conditions.append("session_id = ?")
            params.append(session_id)

        if agent_id is not None:
            conditions.append("agent_id = ?")
            params.append(agent_id)

        if signature is not None:
            conditions.append("signature_json = ?")
            params.append(json.dumps(signature, sort_keys=True))

        if success is not None:
            conditions.append("success = ?")
            params.append(int(success))

        if evaluated is not None:
            conditions.append("evaluated = ?")
            params.append(int(evaluated))

        where_clause = " AND ".join(conditions) if conditions else "1=1"

        query = f"""
            SELECT * FROM rb_trajectories
            WHERE {where_clause}
            ORDER BY created_at DESC
            LIMIT ? OFFSET ?
        """
        params.extend([limit, offset])

        rows = self.db.execute(query, params).fetchall()

        trajectories = []
        for row in rows:
            tool_calls = self.get_tool_calls(row["id"])
            trajectories.append(Trajectory.from_row(row, tool_calls))

        return trajectories

    def get_tool_calls(self, trajectory_id: int) -> List[ToolCallRecord]:
        """
        Get all tool calls for a trajectory.

        Args:
            trajectory_id: Trajectory database ID

        Returns:
            List of tool call records
        """
        rows = self.db.execute("""
            SELECT * FROM rb_trajectory_tool_calls
            WHERE trajectory_id = ?
            ORDER BY step_index ASC
        """, (trajectory_id,)).fetchall()

        return [
            ToolCallRecord(
                step_index=row["step_index"],
                tool_name=row["tool_name"],
                tool_call_id=row["tool_call_id"],
                input=json.loads(row["input_json"] or "{}"),
                output=json.loads(row["output_json"]) if row["output_json"] else None,
                error=row["error"],
                duration_ms=row["duration_ms"]
            )
            for row in rows
        ]

    # ----- Evaluation API (for Phase 2.2) -----

    def get_unevaluated_trajectories(self, limit: int = 100) -> List[Trajectory]:
        """
        Get trajectories that haven't been evaluated.

        Only returns finished trajectories (finished_at IS NOT NULL).

        Args:
            limit: Maximum results

        Returns:
            List of unevaluated trajectories
        """
        rows = self.db.execute("""
            SELECT * FROM rb_trajectories
            WHERE evaluated = 0 AND finished_at IS NOT NULL
            ORDER BY created_at ASC
            LIMIT ?
        """, (limit,)).fetchall()

        trajectories = []
        for row in rows:
            tool_calls = self.get_tool_calls(row["id"])
            trajectories.append(Trajectory.from_row(row, tool_calls))

        return trajectories

    def mark_evaluated(
        self,
        trajectory_id: int,
        success: bool,
        confidence: float = 0.5,
        notes: Optional[str] = None,
    ) -> None:
        """
        Mark a trajectory as evaluated with verdict.

        Args:
            trajectory_id: Trajectory database ID
            success: Whether the trajectory was successful
            confidence: Judge confidence (0.0-1.0)
            notes: Optional evaluation notes/reasoning
        """
        self.db.execute("""
            UPDATE rb_trajectories SET
                evaluated = 1,
                success = ?,
                evaluation_confidence = ?,
                evaluation_notes = ?
            WHERE id = ?
        """, (int(success), confidence, notes, trajectory_id))
        self.db.commit()

    # ----- Extraction API (for Phase 2.3) -----

    def get_successful_unextracted(self, limit: int = 100) -> List[Trajectory]:
        """
        Get successful trajectories without extracted strategies.

        Args:
            limit: Maximum results

        Returns:
            List of successful, unextracted trajectories
        """
        rows = self.db.execute("""
            SELECT * FROM rb_trajectories
            WHERE evaluated = 1
              AND success = 1
              AND strategies_extracted = 0
            ORDER BY created_at ASC
            LIMIT ?
        """, (limit,)).fetchall()

        trajectories = []
        for row in rows:
            tool_calls = self.get_tool_calls(row["id"])
            trajectories.append(Trajectory.from_row(row, tool_calls))

        return trajectories

    def mark_strategies_extracted(self, trajectory_id: int) -> None:
        """
        Mark that strategies have been extracted from this trajectory.

        Args:
            trajectory_id: Trajectory database ID
        """
        self.db.execute("""
            UPDATE rb_trajectories SET strategies_extracted = 1 WHERE id = ?
        """, (trajectory_id,))
        self.db.commit()

    # ----- Maintenance API -----

    def delete_trajectory(self, run_id: str) -> bool:
        """
        Delete a trajectory and all related records.

        Args:
            run_id: Run identifier

        Returns:
            True if deleted, False if not found
        """
        row = self.db.execute(
            "SELECT id FROM rb_trajectories WHERE run_id = ?",
            (run_id,)
        ).fetchone()

        if not row:
            return False

        trajectory_id = row["id"]

        # Delete tool calls first (foreign key)
        self.db.execute(
            "DELETE FROM rb_trajectory_tool_calls WHERE trajectory_id = ?",
            (trajectory_id,)
        )

        # Delete trajectory
        self.db.execute(
            "DELETE FROM rb_trajectories WHERE id = ?",
            (trajectory_id,)
        )

        self.db.commit()
        return True

    def prune_old_trajectories(
        self,
        older_than_days: int = 30,
        keep_successful: bool = True,
    ) -> int:
        """
        Delete old trajectories to manage database size.

        Args:
            older_than_days: Delete trajectories older than this
            keep_successful: If True, never delete successful trajectories

        Returns:
            Number of trajectories deleted
        """
        cutoff = time() - (older_than_days * 24 * 60 * 60)

        if keep_successful:
            # Delete old non-successful trajectories
            cur = self.db.execute("""
                DELETE FROM rb_trajectories
                WHERE created_at < ? AND (success = 0 OR success IS NULL)
            """, (cutoff,))
        else:
            # Delete all old trajectories
            cur = self.db.execute(
                "DELETE FROM rb_trajectories WHERE created_at < ?",
                (cutoff,)
            )

        deleted = cur.rowcount
        self.db.commit()
        return deleted

    def get_statistics(
        self,
        agent_id: Optional[str] = None,
        time_range_days: int = 7,
    ) -> Dict[str, Any]:
        """
        Get aggregate statistics for trajectories.

        Args:
            agent_id: Filter by agent (None = all agents)
            time_range_days: Look back this many days

        Returns:
            Dict with total, evaluated, successful counts and rates
        """
        cutoff = time() - (time_range_days * 24 * 60 * 60)

        if agent_id:
            base_query = "FROM rb_trajectories WHERE agent_id = ? AND created_at > ?"
            params: tuple = (agent_id, cutoff)
        else:
            base_query = "FROM rb_trajectories WHERE created_at > ?"
            params = (cutoff,)

        total = self.db.execute(
            f"SELECT COUNT(*) {base_query}", params
        ).fetchone()[0]

        evaluated = self.db.execute(
            f"SELECT COUNT(*) {base_query} AND evaluated = 1", params
        ).fetchone()[0]

        successful = self.db.execute(
            f"SELECT COUNT(*) {base_query} AND success = 1", params
        ).fetchone()[0]

        avg_duration = self.db.execute(
            f"SELECT AVG(duration_ms) {base_query} AND duration_ms IS NOT NULL", params
        ).fetchone()[0]

        return {
            "total": total,
            "evaluated": evaluated,
            "successful": successful,
            "failed": evaluated - successful,
            "pending_evaluation": total - evaluated,
            "success_rate": successful / evaluated if evaluated > 0 else 0.0,
            "avg_duration_ms": avg_duration or 0.0,
            "time_range_days": time_range_days,
        }

    def close(self):
        """Close database connection."""
        self.db.close()
