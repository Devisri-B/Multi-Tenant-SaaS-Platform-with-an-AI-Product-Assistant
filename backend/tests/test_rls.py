"""Unit tests for PostgreSQL Row Level Security (RLS) helpers and tenancy context."""

from __future__ import annotations

import uuid

from sqlalchemy import Column, Integer, MetaData, Table, create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.db.tenancy import (
    attach_rls_policies,
    get_tenant_context,
    reset_tenant_context,
    set_tenant_context,
    tenant_scope,
)
from app.services.document import DocumentRepository


def test_set_and_get_tenant_context():
    engine = create_engine("sqlite:///:memory:")
    SessionLocal = sessionmaker(bind=engine)
    session = SessionLocal()

    try:
        assert get_tenant_context(session) is None

        tenant_id = uuid.uuid4()
        set_tenant_context(session, tenant_id)
        assert get_tenant_context(session) == str(tenant_id)

        reset_tenant_context(session)
        assert get_tenant_context(session) is None
    finally:
        session.close()


def test_tenant_scope_context_manager():
    engine = create_engine("sqlite:///:memory:")
    SessionLocal = sessionmaker(bind=engine)
    session = SessionLocal()

    try:
        orig_id = uuid.uuid4()
        scoped_id = uuid.uuid4()

        set_tenant_context(session, orig_id)
        assert get_tenant_context(session) == str(orig_id)

        with tenant_scope(session, scoped_id):
            assert get_tenant_context(session) == str(scoped_id)

        assert get_tenant_context(session) == str(orig_id)
    finally:
        session.close()


def test_attach_rls_policies_ddl_compilation():
    metadata = MetaData()
    tbl = Table("test_table", metadata, Column("id", Integer, primary_key=True))
    attach_rls_policies(tbl)

    found_ddl = False
    for listener in tbl.dispatch.after_create:
        text_repr = getattr(listener, "statement", str(listener))
        if "ENABLE ROW LEVEL SECURITY" in text_repr:
            found_ddl = True
            assert "ALTER TABLE test_table ENABLE ROW LEVEL SECURITY;" in text_repr
            assert "ALTER TABLE test_table FORCE ROW LEVEL SECURITY;" in text_repr
            assert "CREATE POLICY tenant_isolation_test_table ON test_table" in text_repr
            assert "current_setting('app.tenant_id', true)" in text_repr
            assert "WITH CHECK" in text_repr
            break

    assert found_ddl, "RLS DDL listener was not properly registered on the table."


def test_repository_sets_tenant_context_on_session(db: Session, owner):
    repo = DocumentRepository(db, owner.tenant_id)
    assert get_tenant_context(db) == str(owner.tenant_id)

    # Calling scoped query or count also ensures context is set
    repo.count()
    assert get_tenant_context(db) == str(owner.tenant_id)
