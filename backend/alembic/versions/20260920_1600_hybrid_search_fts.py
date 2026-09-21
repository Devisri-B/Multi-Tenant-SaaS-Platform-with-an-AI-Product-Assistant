"""Add PostgreSQL Full-Text Search (FTS) GIN index for hybrid retrieval.

Revision ID: 0006_hybrid_fts
Revises: 0005_telemetry
Create Date: 2026-09-20 16:00:00

"""
from __future__ import annotations

from alembic import op

revision = "0006_hybrid_fts"
down_revision = "0005_telemetry"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name == "postgresql":
        op.execute(
            "CREATE INDEX IF NOT EXISTS ix_document_chunks_content_fts "
            "ON document_chunks USING gin (to_tsvector('english', content));"
        )


def downgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name == "postgresql":
        op.execute("DROP INDEX IF EXISTS ix_document_chunks_content_fts;")
