"""PostgreSQL Row Level Security (RLS) tenant context management.

Provides helpers and event listeners to set the transaction-local `app.tenant_id`
configuration parameter in PostgreSQL sessions. This underpins database-level
RLS policies for multi-tenant isolation, acting as defense-in-depth alongside
application-level filtering in repositories.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING

from sqlalchemy import DDL, Table, event, text
from sqlalchemy.orm import Session

if TYPE_CHECKING:
    from sqlalchemy.engine import Connection
    from sqlalchemy.orm import SessionTransaction


def set_tenant_context(session: Session, tenant_id: uuid.UUID | str | None) -> None:
    """Set the PostgreSQL row-level security tenant context.

    Stores the tenant ID in ``session.info["tenant_id"]`` and, if connected
    to PostgreSQL and a transaction is in progress, executes
    ``SELECT set_config('app.tenant_id', :val, true)``.

    ``is_local = true`` ensures the variable is scoped strictly to the current
    transaction, automatically clearing on transaction commit or rollback to
    prevent connection pool pollution.

    On non-PostgreSQL dialects (e.g. SQLite in unit tests), this safely updates
    ``session.info`` without attempting database-level configuration.
    """
    normalized = str(tenant_id) if tenant_id else None
    session.info["tenant_id"] = normalized

    try:
        bind = session.get_bind()
        if bind is not None and bind.dialect.name == "postgresql":
            val = normalized or ""
            session.execute(
                text("SELECT set_config('app.tenant_id', :val, true)"),
                {"val": val},
            )
    except Exception:
        # If the session is not currently bound to an open connection or is outside
        # an active transaction, after_begin will apply it as soon as a connection is acquired.
        pass


def get_tenant_context(session: Session) -> str | None:
    """Return the active tenant ID recorded on the session, if any."""
    return session.info.get("tenant_id")


def reset_tenant_context(session: Session) -> None:
    """Reset the tenant context on the session."""
    set_tenant_context(session, None)


@contextmanager
def tenant_scope(session: Session, tenant_id: uuid.UUID | str | None) -> Iterator[None]:
    """Context manager to temporarily scope a session to a specific tenant."""
    previous = get_tenant_context(session)
    set_tenant_context(session, tenant_id)
    try:
        yield
    finally:
        set_tenant_context(session, previous)


@event.listens_for(Session, "after_begin")
def _on_session_begin(
    session: Session,
    _transaction: SessionTransaction,
    connection: Connection,
) -> None:
    """Ensure every new PostgreSQL transaction receives the active tenant context.

    This ensures that even if a transaction is committed and a new transaction
    begins within the same session, `app.tenant_id` is re-applied before any query runs.
    """
    if connection.dialect.name != "postgresql":
        return

    tenant_id = session.info.get("tenant_id")
    val = str(tenant_id) if tenant_id else ""
    connection.execute(
        text("SELECT set_config('app.tenant_id', :val, true)"),
        {"val": val},
    )


def attach_rls_policies(table: Table) -> None:
    """Attach PostgreSQL DDL statements to enable and force RLS on a table.

    Applied via SQLAlchemy ``after_create`` table event listener, guarded with
    ``.execute_if(dialect="postgresql")`` so SQLite runs skip it automatically.
    """
    table_name = table.name
    ddl = DDL(
        f"ALTER TABLE {table_name} ENABLE ROW LEVEL SECURITY;\n"
        f"ALTER TABLE {table_name} FORCE ROW LEVEL SECURITY;\n"
        f"DROP POLICY IF EXISTS tenant_isolation_{table_name} ON {table_name};\n"
        f"CREATE POLICY tenant_isolation_{table_name} ON {table_name} "
        f"FOR ALL "
        f"USING (tenant_id = NULLIF(current_setting('app.tenant_id', true), '')::uuid) "
        f"WITH CHECK (tenant_id = NULLIF(current_setting('app.tenant_id', true), '')::uuid);"
    ).execute_if(dialect="postgresql")

    event.listen(table, "after_create", ddl)
