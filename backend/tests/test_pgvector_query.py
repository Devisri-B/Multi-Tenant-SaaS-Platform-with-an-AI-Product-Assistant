"""Postgres-dialect checks for the vector search path.

The rest of the suite runs on SQLite, which takes the in-Python cosine
fallback, so nothing else here executes ``build_pgvector_query``. A missing
operator or a broken bind chain fails at query-build time rather than at
import, which makes it exactly the kind of bug that reaches production. These
tests compile against the Postgres dialect and need no server.
"""

from __future__ import annotations

import uuid

import pytest
from sqlalchemy.dialects import postgresql

from app.db.types import Vector
from app.models.document import DocumentChunk
from app.rag.retriever import build_pgvector_query

DIALECT = postgresql.dialect()


def compiled(embedding: list[float] | None = None) -> str:
    return str(
        build_pgvector_query(uuid.uuid4(), embedding or [0.1] * 8, 5).compile(
            dialect=DIALECT
        )
    )


@pytest.mark.parametrize(
    "operator",
    ["cosine_distance", "l2_distance", "max_inner_product"],
)
def test_vector_column_exposes_distance_operators(operator):
    """TypeDecorator does not inherit pgvector's comparator; we re-declare it."""
    assert hasattr(DocumentChunk.embedding, operator)


def test_cosine_distance_emits_the_pgvector_operator():
    expression = DocumentChunk.embedding.cosine_distance([0.1] * 8)
    assert "<=>" in str(expression.compile(dialect=DIALECT))


def test_l2_and_inner_product_emit_their_operators():
    assert "<->" in str(
        DocumentChunk.embedding.l2_distance([0.1] * 8).compile(dialect=DIALECT)
    )
    assert "<#>" in str(
        DocumentChunk.embedding.max_inner_product([0.1] * 8).compile(dialect=DIALECT)
    )


def test_retrieval_query_pushes_distance_into_sql():
    sql = compiled()
    assert sql.count("<=>") >= 2, "distance must appear in both SELECT and ORDER BY"
    assert "LIMIT" in sql


def test_retrieval_query_is_tenant_filtered():
    """The isolation predicate must live in SQL, not in Python."""
    sql = compiled()
    assert "document_chunks.tenant_id = " in sql
    assert "embedding IS NOT NULL" in sql


def test_retrieval_query_excludes_unindexed_documents():
    assert "documents.status" in compiled()


def test_bind_chain_produces_a_vector_literal():
    """TypeDecorator -> pgvector bind processor must yield '[a, b, c]'."""
    processor = Vector(3).bind_processor(DIALECT)
    assert processor([0.1, 0.2, 0.3]) == "[0.1, 0.2, 0.3]"


def test_bind_chain_rejects_a_wrong_width_embedding():
    processor = Vector(3).bind_processor(DIALECT)
    with pytest.raises(ValueError, match="3 dimensions"):
        processor([0.1, 0.2])


def test_result_value_round_trips_to_a_float_list():
    vector_type = Vector(3)
    processor = vector_type.result_processor(DIALECT, None)
    assert processor("[0.1,0.2,0.3]") == [0.1, 0.2, 0.3]
