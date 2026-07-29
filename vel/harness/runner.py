from __future__ import annotations

import asyncio
import json
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, Iterable, List, Optional, Protocol, Tuple

from vel.events import ErrorEvent

from .pubsub import InProcessPubSub, PubSub


TERMINAL_STATUSES = {'completed', 'failed', 'cancelled'}
STREAM_END_STATUSES = TERMINAL_STATUSES | {'suspended'}


class EventLogStore(Protocol):
    def ensure_run(self, run_id: str, agent_id: Optional[str], status: str = 'running') -> None: ...
    def set_run_status(self, run_id: str, status: str) -> None: ...
    def get_run_status(self, run_id: str) -> Optional[str]: ...
    def append_event(self, run_id: str, event: Dict[str, Any]) -> int: ...
    def events_after(self, run_id: str, cursor: int = 0) -> List[Tuple[int, Dict[str, Any]]]: ...
    def close(self) -> None: ...


class SQLiteEventLogStore:
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
            CREATE TABLE IF NOT EXISTS vel_runs (
                id TEXT PRIMARY KEY,
                agent_id TEXT,
                status TEXT DEFAULT 'running',
                created_at REAL DEFAULT (strftime('%s','now')),
                updated_at REAL DEFAULT (strftime('%s','now'))
            );
            CREATE TABLE IF NOT EXISTS vel_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT,
                ts REAL DEFAULT (strftime('%s','now')),
                kind TEXT NOT NULL,
                payload TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_vel_events_run_id ON vel_events(run_id, id);
            """
        )
        self.db.commit()

    def ensure_run(self, run_id: str, agent_id: Optional[str], status: str = 'running') -> None:
        now = time.time()
        self.db.execute(
            """
            INSERT INTO vel_runs(id, agent_id, status, created_at, updated_at)
            VALUES(?,?,?,?,?)
            ON CONFLICT(id) DO UPDATE SET
                agent_id=COALESCE(excluded.agent_id, vel_runs.agent_id),
                status=excluded.status,
                updated_at=excluded.updated_at
            """,
            (run_id, agent_id, status, now, now),
        )
        self.db.commit()

    def set_run_status(self, run_id: str, status: str) -> None:
        self.db.execute(
            "UPDATE vel_runs SET status=?, updated_at=? WHERE id=?",
            (status, time.time(), run_id),
        )
        self.db.commit()

    def get_run_status(self, run_id: str) -> Optional[str]:
        row = self.db.execute("SELECT status FROM vel_runs WHERE id=?", (run_id,)).fetchone()
        return row['status'] if row else None

    def append_event(self, run_id: str, event: Dict[str, Any]) -> int:
        cur = self.db.execute(
            "INSERT INTO vel_events(run_id, ts, kind, payload) VALUES(?,?,?,?)",
            (run_id, time.time(), event.get('type', 'unknown'), json.dumps(event)),
        )
        self.db.commit()
        return int(cur.lastrowid)

    def events_after(self, run_id: str, cursor: int = 0) -> List[Tuple[int, Dict[str, Any]]]:
        rows = self.db.execute(
            """
            SELECT id, payload
            FROM vel_events
            WHERE run_id=? AND id > ?
            ORDER BY id
            """,
            (run_id, cursor),
        ).fetchall()
        return [(int(row['id']), json.loads(row['payload'])) for row in rows]

    def close(self) -> None:
        self.db.close()


class PostgresEventLogStore:
    def __init__(self, dsn: str):
        self.dsn = dsn
        self._psycopg = _load_psycopg()
        self.conn = self._psycopg.connect(dsn)
        self._init_schema()

    def _init_schema(self) -> None:
        with self.conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS vel_runs(
                    id TEXT PRIMARY KEY,
                    agent_id TEXT,
                    status TEXT DEFAULT 'running',
                    created_at TIMESTAMPTZ DEFAULT now(),
                    updated_at TIMESTAMPTZ DEFAULT now()
                )
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS vel_events(
                    id BIGSERIAL PRIMARY KEY,
                    run_id TEXT REFERENCES vel_runs(id),
                    ts TIMESTAMPTZ DEFAULT now(),
                    kind TEXT NOT NULL,
                    payload JSONB NOT NULL
                )
                """
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_vel_events_run ON vel_events(run_id, id)"
            )
        self.conn.commit()

    def ensure_run(self, run_id: str, agent_id: Optional[str], status: str = 'running') -> None:
        with self.conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO vel_runs(id, agent_id, status, created_at, updated_at)
                VALUES(%s, %s, %s, now(), now())
                ON CONFLICT(id) DO UPDATE SET
                    agent_id=COALESCE(EXCLUDED.agent_id, vel_runs.agent_id),
                    status=EXCLUDED.status,
                    updated_at=now()
                """,
                (run_id, agent_id, status),
            )
        self.conn.commit()

    def set_run_status(self, run_id: str, status: str) -> None:
        with self.conn.cursor() as cur:
            cur.execute(
                "UPDATE vel_runs SET status=%s, updated_at=now() WHERE id=%s",
                (status, run_id),
            )
        self.conn.commit()

    def get_run_status(self, run_id: str) -> Optional[str]:
        with self.conn.cursor() as cur:
            cur.execute("SELECT status FROM vel_runs WHERE id=%s", (run_id,))
            row = cur.fetchone()
        return row[0] if row else None

    def append_event(self, run_id: str, event: Dict[str, Any]) -> int:
        with self.conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO vel_events(run_id, kind, payload)
                VALUES(%s, %s, %s::jsonb)
                RETURNING id
                """,
                (run_id, event.get('type', 'unknown'), json.dumps(event)),
            )
            event_id = cur.fetchone()[0]
        self.conn.commit()
        return int(event_id)

    def events_after(self, run_id: str, cursor: int = 0) -> List[Tuple[int, Dict[str, Any]]]:
        with self.conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, payload
                FROM vel_events
                WHERE run_id=%s AND id > %s
                ORDER BY id
                """,
                (run_id, cursor),
            )
            rows = cur.fetchall()
        return [(int(row[0]), _coerce_payload(row[1])) for row in rows]

    def close(self) -> None:
        self.conn.close()


class RunManager:
    """Owns detached harness runs, a durable event log, and live pub/sub."""

    def __init__(
        self,
        store_backend: str = 'sqlite',
        db_path: str = ".vel/vel.db",
        dsn: Optional[str] = None,
        event_store: Optional[EventLogStore] = None,
        pubsub: Optional[PubSub] = None,
    ):
        self.store = event_store or self._make_store(store_backend, db_path=db_path, dsn=dsn)
        self.pubsub = pubsub or InProcessPubSub()
        self._tasks: Dict[str, asyncio.Task] = {}
        self._cancels: Dict[str, asyncio.Event] = {}
        self._agents: Dict[str, Any] = {}
        self._harness: Dict[str, Any] = {}
        self._resume_kwargs: Dict[str, Dict[str, Any]] = {}

    def _make_store(
        self,
        store_backend: str,
        *,
        db_path: str,
        dsn: Optional[str],
    ) -> EventLogStore:
        if store_backend == 'sqlite':
            return SQLiteEventLogStore(db_path)
        if store_backend == 'postgres':
            if not dsn:
                raise ValueError("RunManager(store_backend='postgres') requires dsn=")
            return PostgresEventLogStore(dsn)
        raise ValueError(f"Unsupported RunManager store_backend: {store_backend!r}")

    async def start(self, agent, *, input, session_id=None, harness=None, **kw) -> str:
        run_id = kw.pop('run_id', str(uuid.uuid4()))
        self.store.ensure_run(run_id, getattr(agent, 'id', None), 'running')
        self._agents[run_id] = agent
        self._harness[run_id] = harness
        self._resume_kwargs[run_id] = {
            key: value
            for key, value in kw.items()
            if key in {'context', 'generation_config'}
        }
        cancel_token = asyncio.Event()
        self._cancels[run_id] = cancel_token
        task = asyncio.create_task(
            self._drive(
                agent, run_id, input=input, session_id=session_id, harness=harness,
                cancel_token=cancel_token, **kw
            )
        )
        self._tasks[run_id] = task
        return run_id

    async def cancel(self, run_id: str, *, reason: Optional[str] = None) -> bool:
        """Stop a detached run and settle it as cancelled.

        Cooperative first: setting the token lets the run close its open blocks
        and emit `abort` + `finish`, so subscribers see a well-formed ending
        instead of a stream that just stops. `task.cancel()` is the fallback for
        a run wedged somewhere that never reaches an event boundary — it cannot
        emit anything, so it is a last resort rather than the mechanism.

        Five things have to move together or a cancelled run is not really
        cancelled: the task stops, the run status says so, a terminal event
        reaches the durable log, subscribers are woken (otherwise `stream()`
        blocks forever on the queue), and the checkpoint is settled so
        `recover()` does not restart it later.
        """
        token = self._cancels.get(run_id)
        task = self._tasks.get(run_id)
        if token is None and task is None:
            return False

        if token is not None:
            token.set()

        if task is not None and not task.done():
            try:
                await asyncio.wait_for(asyncio.shield(task), timeout=5.0)
            except asyncio.TimeoutError:
                # Never reached an event boundary — stop it the hard way. No
                # abort event is possible in this path; the status and sentinel
                # below are what tell subscribers the run is over.
                task.cancel()
                try:
                    await task
                except (asyncio.CancelledError, Exception):
                    pass
            except Exception:
                pass

        if self.store.get_run_status(run_id) not in TERMINAL_STATUSES:
            self.store.set_run_status(run_id, 'cancelled')
        self._wake_subscribers(run_id)
        self._cancels.pop(run_id, None)
        return True

    async def _drive(
        self, agent, run_id: str, *, input, session_id=None, harness=None,
        cancel_token: Optional[asyncio.Event] = None, **kw
    ) -> None:
        try:
            async for event in agent.run_stream(
                input,
                session_id=session_id,
                harness=harness,
                external_run_id=run_id,
                cancel_token=cancel_token,
                **kw,
            ):
                await self._record_event(run_id, event)
        except Exception as exc:
            await self._record_event(run_id, ErrorEvent(error=str(exc)).to_dict())
            self.store.set_run_status(run_id, 'failed')
        finally:
            self._wake_subscribers(run_id)

    async def stream(self, run_id: str, cursor: int = 0) -> AsyncGenerator[Dict[str, Any], None]:
        last_cursor = cursor
        for event_id, event in self.store.events_after(run_id, last_cursor):
            last_cursor = event_id
            yield self._with_cursor(event, event_id)

        status = self.store.get_run_status(run_id)
        if status in STREAM_END_STATUSES:
            return

        async with self.pubsub.subscribe(run_id) as subscription:
            # Catch-up: yield anything recorded between the initial replay and
            # this subscription. Prevents both missed events and a lost-sentinel
            # hang when a fast run finishes right around subscribe time.
            for event_id, event in self.store.events_after(run_id, last_cursor):
                last_cursor = event_id
                yield self._with_cursor(event, event_id)
            if self.store.get_run_status(run_id) in STREAM_END_STATUSES:
                return
            async for event_id, event in subscription:
                if event is None:
                    # Terminal sentinel: drain any final log rows, then stop.
                    # (Do NOT stop merely because status is terminal — later
                    # events may still be buffered; the sentinel is the end.)
                    for tail_id, tail_event in self.store.events_after(run_id, last_cursor):
                        last_cursor = tail_id
                        yield self._with_cursor(tail_event, tail_id)
                    return
                if event_id <= last_cursor:
                    continue
                last_cursor = event_id
                yield self._with_cursor(event, event_id)

    async def resume(self, run_id: str, decisions: Iterable[Any], **kw) -> None:
        agent = kw.pop('agent', None) or self._agents.get(run_id)
        if agent is None:
            raise ValueError(f"No agent registered for run {run_id!r}")
        harness = kw.pop('harness', self._harness.get(run_id))
        resume_kw = {**self._resume_kwargs.get(run_id, {}), **kw}
        self.store.set_run_status(run_id, 'running')
        task = asyncio.create_task(
            self._drive_resume(agent, run_id, decisions=list(decisions), harness=harness, **resume_kw)
        )
        self._tasks[run_id] = task

    async def _drive_resume(self, agent, run_id: str, *, decisions: List[Any], harness=None, **kw) -> None:
        try:
            async for event in agent.resume(run_id, decisions, harness=harness, **kw):
                await self._record_event(run_id, event)
        except Exception as exc:
            await self._record_event(run_id, ErrorEvent(error=str(exc)).to_dict())
            self.store.set_run_status(run_id, 'failed')
        finally:
            self._wake_subscribers(run_id)

    async def recover(self, run_id: str, **kw) -> None:
        """Recover a run that crashed while ``running`` (drives ``agent.recover``).

        Useful after a process restart: re-attach the agent (implicitly, if it is
        still registered, or via ``agent=``) and continue the run without
        re-running completed tools. New subscribers can ``stream`` it as usual.
        """
        agent = kw.pop('agent', None) or self._agents.get(run_id)
        if agent is None:
            raise ValueError(f"No agent registered for run {run_id!r}")
        harness = kw.pop('harness', self._harness.get(run_id))
        recover_kw = {**self._resume_kwargs.get(run_id, {}), **kw}
        self.store.set_run_status(run_id, 'running')
        task = asyncio.create_task(
            self._drive_recover(agent, run_id, harness=harness, **recover_kw)
        )
        self._tasks[run_id] = task

    async def _drive_recover(self, agent, run_id: str, *, harness=None, **kw) -> None:
        try:
            async for event in agent.recover(run_id, harness=harness, **kw):
                await self._record_event(run_id, event)
        except Exception as exc:
            await self._record_event(run_id, ErrorEvent(error=str(exc)).to_dict())
            self.store.set_run_status(run_id, 'failed')
        finally:
            self._wake_subscribers(run_id)

    async def get_status(self, run_id: str) -> Optional[str]:
        return self.store.get_run_status(run_id)

    async def wait(self, run_id: str) -> None:
        task = self._tasks.get(run_id)
        if task is not None:
            await task

    def close(self) -> None:
        self.store.close()
        if isinstance(self.pubsub, InProcessPubSub):
            self.pubsub._subscribers.clear()

    async def _record_event(self, run_id: str, event: Dict[str, Any]) -> None:
        event_id = self.store.append_event(run_id, event)
        status = self._status_from_event(event)
        if status:
            self.store.set_run_status(run_id, status)
        await self.pubsub.publish(run_id, event_id, event)

    def _with_cursor(self, event: Dict[str, Any], event_id: int) -> Dict[str, Any]:
        return {**event, '_cursor': event_id}

    def _wake_subscribers(self, run_id: str) -> None:
        status = self.store.get_run_status(run_id)
        if status in STREAM_END_STATUSES:
            asyncio.create_task(self.pubsub.publish(run_id, 2**63 - 1, None))

    def _status_from_event(self, event: Dict[str, Any]) -> Optional[str]:
        event_type = event.get('type')
        if event_type == 'data-harness-suspended':
            return 'suspended'
        if event_type == 'data-harness-run-finished':
            return event.get('data', {}).get('status', 'completed')
        if event_type == 'error':
            return 'failed'
        return None


def _load_psycopg():
    try:
        import psycopg
    except ImportError as exc:
        raise ImportError(
            "Postgres RunManager event store requires the optional 'harness-postgres' "
            "extra: pip install 'vel-ai[harness-postgres]'"
        ) from exc
    return psycopg


def _coerce_payload(payload: Any) -> Dict[str, Any]:
    if isinstance(payload, dict):
        return payload
    if isinstance(payload, str):
        return json.loads(payload)
    return dict(payload)


__all__ = ['EventLogStore', 'PostgresEventLogStore', 'RunManager', 'SQLiteEventLogStore']
