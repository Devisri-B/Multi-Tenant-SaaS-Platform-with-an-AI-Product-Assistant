"""Add PostgreSQL Row Level Security (RLS) for tenant-scoped tables.

Revision ID: 0004_rls
Revises: 0003_conversations
Create Date: 2026-09-20 12:00:00

Enables and forces row-level security on `documents`, `document_chunks`,
`conversations`, and `messages` using `current_setting('app.tenant_id', true)::uuid`.
Provides defense-in-depth against unauthorized cross-tenant queries and SQL injections.
"""
from __future__ import annotations

from alembic import op

revision = "0004_rls"
down_revision = "0003_conversations"
branch_labels = None
depends_on = None

TARGET_TABLES = ["documents", "document_chunks", "conversations", "messages"]


def upgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name != "postgresql":
        return

    for table in TARGET_TABLES:
        op.execute(f"ALTER TABLE {table} ENABLE ROW LEVEL SECURITY;")
        op.execute(f"ALTER TABLE {table} FORCE ROW LEVEL SECURITY;")
        op.execute(f"DROP POLICY IF EXISTS tenant_isolation_{table} ON {table};")
        op.execute(f"""
            CREATE POLICY tenant_isolation_{table} ON {table}
            FOR ALL
            USING (tenant_id = NULLIF(current_setting('app.tenant_id', true), '')::uuid)
            WITH CHECK (tenant_id = NULLIF(current_setting('app.tenant_id', true), '')::uuid);
        """)


def downgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name != "postgresql":
        return

    for table in TARGET_TABLES:
        op.execute(f"DROP POLICY IF EXISTS tenant_isolation_{table} ON {table};")
        op.execute(f"ALTER TABLE {table} NO FORCE ROW LEVEL SECURITY;")
        op.execute(f"ALTER TABLE {table} DISABLE ROW LEVEL SECURITY;")
