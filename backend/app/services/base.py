"""Tenant-scoped repository base.

Every query for tenant-owned data goes through :class:`TenantScopedRepository`.
Centralising the ``WHERE tenant_id = :tenant_id`` predicate here means a route
cannot accidentally read across the isolation boundary — there is no code path
to a tenant table that does not pass through ``_scoped``.
"""

from __future__ import annotations

import uuid
from typing import Generic, TypeVar

from sqlalchemy import Select, func, select
from sqlalchemy.orm import Session

from app.core.exceptions import NotFoundError
from app.db.base import Base

ModelT = TypeVar("ModelT", bound=Base)


class TenantScopedRepository(Generic[ModelT]):
    model: type[ModelT]

    def __init__(self, db: Session, tenant_id: uuid.UUID) -> None:
        self.db = db
        self.tenant_id = tenant_id

    # -- query building -----------------------------------------------------
    def _scoped(self) -> Select:
        return select(self.model).where(self.model.tenant_id == self.tenant_id)

    # -- reads --------------------------------------------------------------
    def get(self, entity_id: uuid.UUID) -> ModelT | None:
        stmt = self._scoped().where(self.model.id == entity_id)
        return self.db.execute(stmt).scalars().first()

    def get_or_404(self, entity_id: uuid.UUID) -> ModelT:
        entity = self.get(entity_id)
        if entity is None:
            raise NotFoundError(f"{self.model.__name__} {entity_id} was not found.")
        return entity

    def list(self, *, offset: int = 0, limit: int = 20, order_desc: bool = True) -> list[ModelT]:
        order = self.model.created_at.desc() if order_desc else self.model.created_at.asc()
        stmt = self._scoped().order_by(order).offset(offset).limit(limit)
        return list(self.db.execute(stmt).scalars().all())

    def count(self) -> int:
        stmt = (
            select(func.count())
            .select_from(self.model)
            .where(self.model.tenant_id == self.tenant_id)
        )
        return int(self.db.execute(stmt).scalar_one())

    # -- writes -------------------------------------------------------------
    def add(self, entity: ModelT) -> ModelT:
        if getattr(entity, "tenant_id", None) is None:
            entity.tenant_id = self.tenant_id
        if entity.tenant_id != self.tenant_id:
            raise ValueError("Refusing to persist an entity owned by another tenant.")
        self.db.add(entity)
        self.db.flush()
        return entity

    def delete(self, entity: ModelT) -> None:
        if entity.tenant_id != self.tenant_id:
            raise ValueError("Refusing to delete an entity owned by another tenant.")
        self.db.delete(entity)
        self.db.flush()
