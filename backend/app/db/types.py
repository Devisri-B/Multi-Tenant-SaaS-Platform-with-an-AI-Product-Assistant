"""Portable column types.

The production database is Postgres with the ``pgvector`` extension, but the
test-suite runs on SQLite so that CI needs no services.  ``Vector`` below binds
to a real ``vector`` column on Postgres and degrades to a JSON array elsewhere,
which keeps the ORM models identical across both environments.
"""

from __future__ import annotations

import json
import uuid
from typing import Any

from sqlalchemy import CHAR, JSON, Float, TypeDecorator
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import UUID as PGUUID

from app.core.config import settings


class GUID(TypeDecorator):
    """UUID column: native ``uuid`` on Postgres, 32-char hex string elsewhere."""

    impl = CHAR
    cache_ok = True

    def load_dialect_impl(self, dialect):
        if dialect.name == "postgresql":
            return dialect.type_descriptor(PGUUID(as_uuid=True))
        return dialect.type_descriptor(CHAR(32))

    def process_bind_param(self, value: Any, dialect) -> Any:
        if value is None:
            return None
        if not isinstance(value, uuid.UUID):
            value = uuid.UUID(str(value))
        if dialect.name == "postgresql":
            return value
        return value.hex

    def process_result_value(self, value: Any, dialect) -> uuid.UUID | None:
        if value is None:
            return None
        if isinstance(value, uuid.UUID):
            return value
        return uuid.UUID(str(value))


class JSONType(TypeDecorator):
    """JSONB on Postgres, plain JSON elsewhere."""

    impl = JSON
    cache_ok = True

    def load_dialect_impl(self, dialect):
        if dialect.name == "postgresql":
            return dialect.type_descriptor(JSONB())
        return dialect.type_descriptor(JSON())


class Vector(TypeDecorator):
    """Embedding column backed by pgvector on Postgres, JSON text on SQLite.

    A ``TypeDecorator`` does not inherit the comparator of the type it wraps,
    so pgvector's distance operators have to be re-declared here. Without this
    ``DocumentChunk.embedding.cosine_distance(...)`` raises ``AttributeError``
    at query-build time on Postgres.
    """

    impl = JSON
    cache_ok = True

    class comparator_factory(TypeDecorator.Comparator):  # noqa: N801
        """Expose pgvector's distance operators on the decorated column."""

        def l2_distance(self, other: Any):
            return self.op("<->", return_type=Float)(other)

        def max_inner_product(self, other: Any):
            return self.op("<#>", return_type=Float)(other)

        def cosine_distance(self, other: Any):
            return self.op("<=>", return_type=Float)(other)

    def __init__(self, dimensions: int | None = None, **kwargs: Any) -> None:
        self.dimensions = dimensions or settings.EMBEDDING_DIMENSIONS
        super().__init__(**kwargs)

    def load_dialect_impl(self, dialect):
        if dialect.name == "postgresql":
            from pgvector.sqlalchemy import Vector as PGVector

            return dialect.type_descriptor(PGVector(self.dimensions))
        return dialect.type_descriptor(JSON())

    def process_bind_param(self, value: Any, dialect) -> Any:
        if value is None:
            return None
        vector = [float(component) for component in value]
        if len(vector) != self.dimensions:
            raise ValueError(
                f"Expected an embedding of {self.dimensions} dimensions, "
                f"received {len(vector)}."
            )
        if dialect.name == "postgresql":
            return vector
        return vector

    def process_result_value(self, value: Any, dialect) -> list[float] | None:
        if value is None:
            return None
        if isinstance(value, str):
            return json.loads(value)
        return list(value)
