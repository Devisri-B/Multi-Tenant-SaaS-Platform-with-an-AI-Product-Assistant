"""Add conversation summary column for memory outside sliding window.

Revision ID: 0007_conversation_summary
Revises: 0006_hybrid_fts
Create Date: 2026-09-23 10:00:00

"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0007_conversation_summary"
down_revision = "0006_hybrid_fts"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("conversations", sa.Column("summary", sa.Text(), nullable=True))


def downgrade() -> None:
    op.drop_column("conversations", "summary")
