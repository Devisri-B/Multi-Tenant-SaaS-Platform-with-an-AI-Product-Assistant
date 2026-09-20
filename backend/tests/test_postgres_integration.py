"""End-to-end retrieval against a real pgvector-enabled Postgres.

Skipped unless ``TEST_DATABASE_URL`` points at Postgres, so local runs and the
default CI job stay service-free. The migrations job in CI sets it, which is
what stops a Postgres-only regression from shipping again: the compile-level
checks in ``test_pgvector_query.py`` catch a missing operator, but only a live
round-trip catches a bad bind or a broken index.
"""

from __future__ import annotations

import os
import uuid
from collections.abc import Iterator

import pytest
from sqlalchemy import create_engine, event, text
from sqlalchemy.orm import Session, sessionmaker

POSTGRES_URL = os.environ.get("TEST_DATABASE_URL", "")

pytestmark = pytest.mark.skipif(
    not POSTGRES_URL.startswith("postgresql"),
    reason="Set TEST_DATABASE_URL to a pgvector Postgres to run these.",
)

DOC = """\
# Billing and Refunds

Refunds are issued to the original payment method within ten business days.
Invoices are generated on the first of every month.
"""

OTHER_DOC = """\
# Deployment Guide

Deployments run through GitHub Actions with a rolling restart of the API pods.
"""


@pytest.fixture(scope="module")
def pg_session() -> Iterator[Session]:
    import app.models  # noqa: F401  (registers every table on the metadata)
    from app.db.base import Base

    admin_engine = create_engine(POSTGRES_URL, future=True)
    with admin_engine.begin() as connection:
        connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        connection.execute(text("""
            DO $$
            BEGIN
                IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = 'nimbus_app') THEN
                    CREATE ROLE nimbus_app LOGIN NOSUPERUSER NOBYPASSRLS;
                END IF;
            END
            $$;
        """))
    Base.metadata.drop_all(bind=admin_engine)
    Base.metadata.create_all(bind=admin_engine)

    with admin_engine.begin() as connection:
        connection.execute(text("""
            GRANT ALL ON ALL TABLES IN SCHEMA public TO nimbus_app;
            GRANT ALL ON ALL SEQUENCES IN SCHEMA public TO nimbus_app;
            ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON TABLES TO nimbus_app;
            ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON SEQUENCES TO nimbus_app;
        """))
    admin_engine.dispose()

    test_engine = create_engine(POSTGRES_URL, future=True)

    @event.listens_for(test_engine, "connect")
    def _set_role(dbapi_connection, _record):
        cursor = dbapi_connection.cursor()
        cursor.execute("SET ROLE nimbus_app")
        cursor.close()

    session = sessionmaker(bind=test_engine, expire_on_commit=False)()
    try:
        yield session
    finally:
        session.close()
        cleanup_engine = create_engine(POSTGRES_URL, future=True)
        try:
            with cleanup_engine.begin() as connection:
                for table in reversed(Base.metadata.sorted_tables):
                    connection.execute(table.delete())
        except Exception:
            pass
        finally:
            cleanup_engine.dispose()
        test_engine.dispose()


@pytest.fixture(scope="module")
def seeded(pg_session: Session):
    """Two workspaces, each with its own document, indexed for real."""
    from app.services import auth as auth_service
    from app.services import document as document_service
    from app.services import tenant as tenant_service

    suffix = uuid.uuid4().hex[:8]
    owner = auth_service.create_user(
        pg_session, email=f"pg-{suffix}@acme.io", password="OwnerPassw0rd"
    )
    mine = tenant_service.create_tenant(pg_session, name=f"Acme {suffix}", owner=owner)
    theirs = tenant_service.create_tenant(pg_session, name=f"Globex {suffix}", owner=owner)

    document_service.create_document_from_text(
        pg_session, tenant=mine, uploader=owner, title="Billing", content=DOC
    )
    document_service.create_document_from_text(
        pg_session, tenant=theirs, uploader=owner, title="Deployment", content=OTHER_DOC
    )
    pg_session.commit()
    return mine, theirs


def test_documents_index_on_postgres(pg_session, seeded):
    """Embeddings must round-trip through a real vector column."""
    from app.db.tenancy import set_tenant_context
    from app.models.document import DocumentChunk

    mine, _ = seeded
    set_tenant_context(pg_session, mine.id)
    chunk = (
        pg_session.query(DocumentChunk).filter(DocumentChunk.tenant_id == mine.id).first()
    )
    assert chunk is not None
    assert chunk.embedding is not None
    assert len(chunk.embedding) > 0


def test_pgvector_retrieval_returns_ranked_hits(pg_session, seeded):
    """The <=> operator path - the one SQLite never exercises."""
    from app.rag.chain import semantic_search

    mine, _ = seeded
    hits = semantic_search(
        pg_session, tenant_id=mine.id, query="refunds original payment method", top_k=5
    )
    assert hits, "pgvector retrieval returned nothing"
    assert hits[0].document_title == "Billing"
    scores = [hit.score for hit in hits]
    assert scores == sorted(scores, reverse=True)


def test_pgvector_retrieval_is_tenant_isolated(pg_session, seeded):
    from app.rag.chain import semantic_search

    _, theirs = seeded
    hits = semantic_search(
        pg_session, tenant_id=theirs.id, query="refunds original payment method", top_k=5
    )
    titles = {hit.document_title for hit in hits}
    assert "Billing" not in titles


def test_answer_question_end_to_end_on_postgres(pg_session, seeded):
    from app.rag.chain import answer_question

    mine, _ = seeded
    result = answer_question(
        pg_session, tenant=mine, question="How long do refunds take?"
    )
    assert result.used_context is True
    assert result.citations
    assert result.citations[0]["document_title"] == "Billing"


def test_rls_blocks_raw_sql_injection(pg_session, seeded):
    """Even a 1=1 SQL injection cannot leak rows across tenant boundaries under RLS."""
    from app.db.tenancy import set_tenant_context

    mine, theirs = seeded
    set_tenant_context(pg_session, mine.id)

    # Simulated SQL injection bypass: WHERE title = 'anything' OR 1=1
    injected_query = text(
        "SELECT id, tenant_id, title FROM documents WHERE title = 'Deployment' OR 1=1"
    )
    rows = pg_session.execute(injected_query).fetchall()

    # RLS ensures that ONLY documents belonging to `mine.id` are returned
    assert len(rows) > 0
    for row in rows:
        assert str(row.tenant_id) == str(mine.id)
        assert row.title != "Deployment"  # "Deployment" belongs to `theirs`


def test_rls_blocks_read_without_tenant_context(pg_session, seeded):
    """Queries executed without setting app.tenant_id return zero rows."""
    from app.db.tenancy import reset_tenant_context

    reset_tenant_context(pg_session)
    count = pg_session.execute(text("SELECT count(*) FROM documents")).scalar_one()
    assert count == 0


def test_rls_with_check_blocks_insert_mismatch(pg_session, seeded):
    """PostgreSQL WITH CHECK policy blocks inserting a row for another tenant."""
    from app.db.tenancy import set_tenant_context

    mine, theirs = seeded
    set_tenant_context(pg_session, mine.id)

    # Attempting to insert a document owned by `theirs` while scoped to `mine`
    with pytest.raises(Exception) as exc_info:
        pg_session.execute(
            text(
                "INSERT INTO documents "
                "(id, tenant_id, title, source_name, content_type, checksum, "
                "byte_size, status, chunk_count, doc_metadata) "
                "VALUES (:id, :tenant_id, 'Evil', 'evil.txt', 'text/plain', "
                "'deadbeef', 10, 'indexed', 0, '{}')"
            ),
            {"id": uuid.uuid4(), "tenant_id": theirs.id},
        )
        pg_session.flush()

    assert "violates row-level security policy" in str(exc_info.value).lower()
    pg_session.rollback()
