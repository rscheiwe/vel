#!/usr/bin/env bash
set -euo pipefail

# Run from the repo root (where pyproject.toml lives)
mkdir -p scripts alembic/versions vel

echo ">>> Writing .env.example"
cat > .env.example <<'ENV'
# Database & cache
POSTGRES_DSN=postgresql+psycopg://user:pass@localhost:5432/vel
REDIS_URL=redis://localhost:6379/0

# OpenAI
OPENAI_API_KEY=sk-...
OPENAI_API_BASE=https://api.openai.com/v1

# Runner mode: local-async | queue
VEL_RUNNER=local-async
ENV

echo ">>> Writing alembic.ini"
cat > alembic.ini <<'INI'
[alembic]
script_location = alembic
prepend_sys_path = .
sqlalchemy.url = %(DB_URL)s

[loggers]
keys = root,sqlalchemy,alembic

[handlers]
keys = console

[formatters]
keys = generic

[logger_root]
level = WARN
handlers = console

[logger_sqlalchemy]
level = WARN
handlers =
qualname = sqlalchemy.engine

[logger_alembic]
level = INFO
handlers = console
qualname = alembic

[handler_console]
class = StreamHandler
args = (sys.stderr,)
level = NOTSET
formatter = generic

[formatter_generic]
format = %(levelname)-5.5s [%(name)s] %(message)s
datefmt = %H:%M:%S
INI

echo ">>> Writing alembic/env.py"
mkdir -p alembic
cat > alembic/env.py <<'PY'
from __future__ import annotations
import os
from logging.config import fileConfig
from sqlalchemy import engine_from_config, pool
from alembic import context

# Alembic Config
config = context.config

db_url = os.getenv("DB_URL") or os.getenv("POSTGRES_DSN")
if db_url:
    config.set_main_option("sqlalchemy.url", db_url)

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = None

def run_migrations_offline():
    url = config.get_main_option("sqlalchemy.url")
    context.configure(url=url, literal_binds=True, dialect_opts={"paramstyle": "named"})
    with context.begin_transaction():
        context.run_migrations()

def run_migrations_online():
    connectable = engine_from_config(
        config.get_section(config.config_ini_section), prefix="sqlalchemy.", poolclass=pool.NullPool
    )
    with connectable.connect() as connection:
        context.configure(connection=connection)
        with context.begin_transaction():
            context.run_migrations()

if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
PY

echo ">>> Writing first migration"
cat > alembic/versions/0001_init_vel_events.py <<'PY'
from __future__ import annotations
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = '0001_init_vel_events'
down_revision = None
branch_labels = None
depends_on = None

def upgrade():
    op.execute("""
    create table if not exists vel_runs (
        id text primary key,
        agent_id text,
        status text default 'running',
        created_at timestamptz default now(),
        updated_at timestamptz default now()
    );
    """)
    op.execute("""
    create table if not exists vel_events (
        id bigserial primary key,
        run_id text references vel_runs(id),
        ts timestamptz default now(),
        kind text not null,
        payload jsonb not null
    );
    """)

def downgrade():
    op.execute("drop table if exists vel_events;")
    op.execute("drop table if exists vel_runs;")
PY

echo ">>> Backing up and writing vel/storage_pg.py (async SQLAlchemy over psycopg)"
[ -f vel/storage_pg.py ] && cp vel/storage_pg.py vel/storage_pg.py.bak || true
cat > vel/storage_pg.py <<'PY'
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
PY

echo ">>> Backing up and writing vel/storage.py (PG facade + Redis cache)"
[ -f vel/storage.py ] && cp vel/storage.py vel/storage.py.bak || true
cat > vel/storage.py <<'PY'
from __future__ import annotations
import os, json, uuid
from typing import Any, Dict, List, Optional

try:
    import redis  # type: ignore
except Exception:  # pragma: no cover
    redis = None

from .storage_pg import PGStore

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
PY

echo ">>> Patching vel/agent.py to persist runs (create_run + update_status)"
# Create a backup
cp vel/agent.py vel/agent.py.bak

# Overwrite file with a persistence-aware run loop
cat > vel/agent.py <<'PY'
from __future__ import annotations
import asyncio, uuid
from typing import Any, AsyncGenerator, Dict, List
from .reducer import State, reduce
from .providers import ProviderRegistry
from .tools import ToolRegistry, validate_io
from .context import ContextManager
from .storage import RunStore

class Agent:
    def __init__(self, id: str, model: Dict[str, Any], prompt_env: str='prod',
                 tools: List[str]|None=None, policies: Dict[str, Any]|None=None):
        self.id = id
        self.model_cfg = model
        self.prompt_env = prompt_env
        self.tools = tools or []
        self.policies = policies or {'max_steps': 24, 'retry': {'attempts': 2}}
        self.providers = ProviderRegistry.default()
        self.toolreg = ToolRegistry.default()
        self.ctxmgr = ContextManager()
        self.store = RunStore.default()

    async def _call_llm_plan(self, run_id: str) -> Dict[str, Any]:
        messages = self.ctxmgr.messages_for_llm(run_id)
        provider = self.providers.get(self.model_cfg['provider'])
        step = await provider.plan(messages, model=self.model_cfg['model'], tools=self.toolreg.schemas())
        return step

    async def _call_tool(self, step: Dict[str, Any]) -> Dict[str, Any]:
        tname = step['tool']
        args = step.get('args', {})
        tool = self.toolreg.get(tname)
        validate_io(tool.input_schema, args)
        result = await tool.run(args, ctx={})
        validate_io(tool.output_schema, result)
        return result

    async def run_stream(self, input: Dict[str, Any]) -> AsyncGenerator[Dict[str, Any], None]:
        run_id = await self.store.create_run(self.id)
        state = State(run_id=run_id)
        await self.store.append_event(run_id, {'kind':'start', 'agent_id': self.id, 'input':input})
        event: Dict[str, Any] = {'kind':'start'}
        steps = 0
        try:
            while True:
                state, effects = reduce(state, event)
                for eff in effects:
                    if eff.kind == 'emit':
                        yield eff.payload
                    elif eff.kind == 'call_llm':
                        step = await self._call_llm_plan(run_id)
                        event = {'kind':'llm_step', 'step': step}
                        await self.store.append_event(run_id, event)
                        break
                    elif eff.kind == 'call_tool':
                        result = await self._call_tool(eff.payload)
                        event = {'kind':'tool_result', 'result': result}
                        await self.store.append_event(run_id, event)
                        break
                    elif eff.kind == 'halt':
                        final = eff.payload.get('final','')
                        await self.store.append_event(run_id, {'kind':'final','answer':final})
                        await self.store.update_status(run_id, 'completed')
                        yield {'kind':'final','answer':final}
                        return
                steps += 1
                if steps > self.policies.get('max_steps', 24):
                    msg = 'max steps exceeded'
                    await self.store.append_event(run_id, {'kind':'error','message': msg})
                    await self.store.update_status(run_id, 'failed')
                    yield {'kind':'error', 'message': msg}
                    return
        except asyncio.CancelledError:
            await self.store.update_status(run_id, 'canceled')
            raise
        except Exception as e:
            await self.store.append_event(run_id, {'kind':'error','message': str(e)})
            await self.store.update_status(run_id, 'failed')
            raise

async def run_stream(agent: 'Agent', input: Dict[str, Any]):
    async for e in agent.run_stream(input):
        yield e
PY

echo ">>> Done. Next steps:"
echo "1) Add/verify dependencies: pip install 'sqlalchemy>=2.0' 'alembic>=1.13' 'redis>=5.0' 'psycopg[binary,pool]>=3.2'"
echo "2) Set POSTGRES_DSN (see .env.example), then run: DB_URL=\$POSTGRES_DSN alembic upgrade head"
echo "3) Start your service and exercise a run."

exit 0
