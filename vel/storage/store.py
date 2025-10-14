from __future__ import annotations
import os, json, uuid
from typing import Any, Dict, List, Optional

try:
    import redis  # type: ignore
except Exception:  # pragma: no cover
    redis = None

from .postgres import PGStore

class RunStore:
    """
    Unified facade over Postgres (durable) and Redis (cache).
    If POSTGRES_DSN is unset, falls back to in-memory.
    """
    def __init__(self, dsn: Optional[str] = None, redis_url: Optional[str] = None):
        self.dsn = dsn or os.getenv('POSTGRES_DSN')
        self.redis_url = redis_url or os.getenv('REDIS_URL')
        self._events: Dict[str, List[Dict[str,Any]]] = {}
        self._pg = PGStore(self.dsn) if self.dsn else None
        self._redis = redis.Redis.from_url(self.redis_url) if (self.redis_url and redis) else None

    @classmethod
    def default(cls) -> 'RunStore':
        return cls()

    async def create_run(self, agent_id: str) -> str:
        run_id = str(uuid.uuid4())
        if self._pg:
            await self._pg.ensure_schema()
            await self._pg.create_run(run_id, agent_id)
        return run_id

    async def update_status(self, run_id: str, status: str):
        if self._pg:
            await self._pg.update_status(run_id, status)

    async def append_event(self, run_id: str, event: Dict[str,Any]):
        if self._pg:
            await self._pg.append_event(run_id, event)
        else:
            self._events.setdefault(run_id, []).append(event)
        if self._redis:
            key = f"vel:events:{run_id}"
            self._redis.rpush(key, json.dumps(event))
            self._redis.expire(key, 3600)

    async def read_events(self, run_id: str) -> List[Dict[str,Any]]:
        if self._pg:
            return await self._pg.read_events(run_id)
        return self._events.get(run_id, [])

    async def save_session(self, session_id: str, context: List[Dict[str, Any]]):
        """Save session context (only if database configured)"""
        if self._pg:
            await self._pg.ensure_schema()
            await self._pg.save_session(session_id, context)

    async def load_session(self, session_id: str) -> List[Dict[str, Any]]:
        """Load session context from database"""
        if self._pg:
            return await self._pg.load_session(session_id)
        return []

    async def delete_session(self, session_id: str):
        """Delete session from database"""
        if self._pg:
            await self._pg.delete_session(session_id)
