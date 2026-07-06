"""Approve-once memory table + crash-recovery checkpoint fields.

Additive per the backwards-compatibility contract (§8): only ``CREATE TABLE IF
NOT EXISTS`` for the new ``vel_approved_tools`` table (session-scoped
approve-once memory). The crash-recovery field ``completed_tool_calls`` lives
inside the existing ``vel_checkpoints.snapshot`` JSONB and therefore needs no
DDL. DDL targets Postgres (the ``persistent`` backend via alembic); the
``transient`` SQLite backend self-initializes equivalent tables in
``vel/harness/checkpoint.py`` and does not run alembic.
"""
from __future__ import annotations
from alembic import op

# revision identifiers, used by Alembic.
revision = '0003_approval_memory_and_recovery'
down_revision = '0002_harness_mode'
branch_labels = None
depends_on = None


def upgrade():
    op.execute("""
    create table if not exists vel_approved_tools (
        session_id text not null,
        tool_name text not null,
        created_at timestamptz default now(),
        primary key (session_id, tool_name)
    );
    """)
    # completed_tool_calls is stored inside vel_checkpoints.snapshot (jsonb);
    # no column change is required — old rows read back as an empty list.


def downgrade():
    op.execute("drop table if exists vel_approved_tools;")
