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
