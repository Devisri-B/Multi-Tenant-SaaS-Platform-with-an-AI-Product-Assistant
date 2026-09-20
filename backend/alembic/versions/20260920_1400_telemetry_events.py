"""Add telemetry events table with PostgreSQL Row Level Security.

Revision ID: 0005_telemetry
Revises: 0004_rls
Create Date: 2026-09-20 14:00:00

"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0005_telemetry"
down_revision = "0004_rls"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    is_postgres = bind.dialect.name == "postgresql"

    uuid_type = postgresql.UUID(as_uuid=True) if is_postgres else sa.String(36)
    json_type = postgresql.JSONB() if is_postgres else sa.JSON()

    op.create_table(
        "telemetry_events",
        sa.Column("id", uuid_type, nullable=False),
        sa.Column("tenant_id", uuid_type, nullable=False),
        sa.Column("user_id", uuid_type, nullable=True),
        sa.Column("conversation_id", uuid_type, nullable=True),
        sa.Column("message_id", uuid_type, nullable=True),
        sa.Column("event_type", sa.String(length=64), nullable=False),
        sa.Column(
            "event_data",
            json_type,
            nullable=False,
            server_default="{}",
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id", name="pk_telemetry_events"),
        sa.ForeignKeyConstraint(["tenant_id"], ["tenants.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="SET NULL"),
        sa.ForeignKeyConstraint(["conversation_id"], ["conversations.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["message_id"], ["messages.id"], ondelete="CASCADE"),
    )
    op.create_index("ix_telemetry_events_tenant_id", "telemetry_events", ["tenant_id"])
    op.create_index("ix_telemetry_tenant_event", "telemetry_events", ["tenant_id", "event_type"])
    op.create_index("ix_telemetry_conversation", "telemetry_events", ["conversation_id"])
    op.create_index("ix_telemetry_created_at", "telemetry_events", ["created_at"])

    if is_postgres:
        op.execute("ALTER TABLE telemetry_events ENABLE ROW LEVEL SECURITY;")
        op.execute("ALTER TABLE telemetry_events FORCE ROW LEVEL SECURITY;")
        op.execute("DROP POLICY IF EXISTS tenant_isolation_telemetry_events ON telemetry_events;")
        op.execute("""
            CREATE POLICY tenant_isolation_telemetry_events ON telemetry_events
            FOR ALL
            USING (tenant_id = NULLIF(current_setting('app.tenant_id', true), '')::uuid)
            WITH CHECK (tenant_id = NULLIF(current_setting('app.tenant_id', true), '')::uuid);
        """)


def downgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name == "postgresql":
        op.execute("DROP POLICY IF EXISTS tenant_isolation_telemetry_events ON telemetry_events;")
        op.execute("ALTER TABLE telemetry_events NO FORCE ROW LEVEL SECURITY;")
        op.execute("ALTER TABLE telemetry_events DISABLE ROW LEVEL SECURITY;")

    op.drop_table("telemetry_events")
