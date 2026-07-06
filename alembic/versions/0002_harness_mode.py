"""Harness Mode durability tables: checkpoints + approvals.

Additive per the backwards-compatibility contract (§8): only ``CREATE TABLE IF
NOT EXISTS`` / new indexes; existing ``vel_runs``/``vel_events`` rows are
untouched. DDL targets Postgres (the ``persistent`` backend via alembic). The
``transient`` SQLite backend self-initializes equivalent tables in
``vel/harness/checkpoint.py`` and does not run alembic.
"""
from __future__ import annotations
from alembic import op

# revision identifiers, used by Alembic.
revision = '0002_harness_mode'
down_revision = '0001_init_vel_events'
branch_labels = None
depends_on = None


def upgrade():
    op.execute("""
    create table if not exists vel_checkpoints (
        run_id text primary key references vel_runs(id),
        agent_id text not null,
        session_id text,
        status text not null default 'running',
        step int not null default 0,
        snapshot jsonb not null,
        config_hash text not null,
        created_at timestamptz default now(),
        updated_at timestamptz default now()
    );
    """)
    op.execute("create index if not exists idx_ckpt_session on vel_checkpoints(session_id);")
    op.execute("create index if not exists idx_ckpt_status  on vel_checkpoints(status);")

    op.execute("""
    create table if not exists vel_approvals (
        approval_id text primary key,
        run_id text not null references vel_runs(id),
        tool_call_id text not null,
        tool_name text not null,
        args jsonb not null,
        reason text,
        status text not null default 'pending',
        decision jsonb,
        created_at timestamptz default now(),
        decided_at timestamptz
    );
    """)
    op.execute("create index if not exists idx_appr_run on vel_approvals(run_id);")
    op.execute("create index if not exists idx_appr_tool_call on vel_approvals(tool_call_id);")


def downgrade():
    op.execute("drop table if exists vel_approvals;")
    op.execute("drop table if exists vel_checkpoints;")
