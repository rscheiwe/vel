from __future__ import annotations
import json
from typing import Any, Dict, List
import sqlalchemy as sa
from sqlalchemy.ext.asyncio import create_async_engine

# Use psycopg (async) dialect; DSN should be postgresql+psycopg://...
class PGStore:
    def __init__(self, dsn: str):
        # Expect dsn like postgresql+psycopg://user:pass@host:port/db
        self.dsn = dsn
        self.engine = create_async_engine(self.dsn, pool_pre_ping=True, future=True)

    async def ensure_schema(self):
        async with self.engine.begin() as conn:
            await conn.execute(sa.text("""
            create table if not exists vel_runs (
                id text primary key,
                agent_id text,
                status text default 'running',
                created_at timestamptz default now(),
                updated_at timestamptz default now()
            );
            """))
            await conn.execute(sa.text("""
            create table if not exists vel_events (
                id bigserial primary key,
                run_id text references vel_runs(id),
                ts timestamptz default now(),
                kind text not null,
                payload jsonb not null
            );
            """))
            await conn.execute(sa.text("""
            create table if not exists vel_sessions (
                id text primary key,
                context jsonb not null,
                created_at timestamptz default now(),
                updated_at timestamptz default now(),
                expires_at timestamptz
            );
            """))

    async def create_run(self, run_id: str, agent_id: str):
        async with self.engine.begin() as conn:
            await conn.execute(sa.text("insert into vel_runs(id, agent_id) values (:i,:a)"),
                               {"i": run_id, "a": agent_id})

    async def update_status(self, run_id: str, status: str):
        async with self.engine.begin() as conn:
            await conn.execute(sa.text("update vel_runs set status=:s, updated_at=now() where id=:i"),
                               {"i": run_id, "s": status})

    async def append_event(self, run_id: str, event: Dict[str,Any]):
        async with self.engine.begin() as conn:
            await conn.execute(sa.text("insert into vel_events(run_id, kind, payload) values (:r,:k,:p)"),
                               {"r": run_id, "k": event.get("kind","event"), "p": json.dumps(event)})

    async def read_events(self, run_id: str):
        async with self.engine.connect() as conn:
            res = await conn.execute(sa.text("select payload from vel_events where run_id=:r order by id asc"),
                                     {"r": run_id})
            return [row[0] for row in res.fetchall()]

    async def save_session(self, session_id: str, context: List[Dict[str, Any]]):
        """Save session context to database"""
        async with self.engine.begin() as conn:
            await conn.execute(sa.text("""
                insert into vel_sessions(id, context, updated_at)
                values (:id, :ctx, now())
                on conflict (id) do update
                set context = :ctx, updated_at = now()
            """), {"id": session_id, "ctx": json.dumps(context)})

    async def load_session(self, session_id: str) -> List[Dict[str, Any]]:
        """Load session context from database"""
        async with self.engine.connect() as conn:
            res = await conn.execute(sa.text("select context from vel_sessions where id=:id"),
                                     {"id": session_id})
            row = res.fetchone()
            return row[0] if row else []

    async def delete_session(self, session_id: str):
        """Delete a session from database"""
        async with self.engine.begin() as conn:
            await conn.execute(sa.text("delete from vel_sessions where id=:id"),
                               {"id": session_id})
