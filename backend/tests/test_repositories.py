"""Direct tests of the tenant-scoped repository guarantees."""

from __future__ import annotations

import uuid

import pytest

from app.core.exceptions import NotFoundError
from app.models.document import Document
from app.models.enums import DocumentStatus
from app.services.document import DocumentRepository


def _document(tenant_id, title="Doc", checksum=None) -> Document:
    return Document(
        tenant_id=tenant_id,
        title=title,
        source_name=f"{title}.md",
        checksum=checksum or uuid.uuid4().hex,
        byte_size=10,
        status=DocumentStatus.INDEXED,
    )


def test_get_or_404_raises_for_missing(db, owner):
    repo = DocumentRepository(db, owner.tenant_id)
    with pytest.raises(NotFoundError):
        repo.get_or_404(uuid.uuid4())


def test_repository_refuses_foreign_tenant_writes(db, owner, other_owner):
    repo = DocumentRepository(db, owner.tenant_id)
    with pytest.raises(ValueError):
        repo.add(_document(other_owner.tenant_id))


def test_repository_stamps_tenant_id_when_absent(db, owner):
    repo = DocumentRepository(db, owner.tenant_id)
    document = Document(
        title="Stamped",
        source_name="stamped.md",
        checksum=uuid.uuid4().hex,
        byte_size=1,
    )
    repo.add(document)
    assert document.tenant_id == owner.tenant_id


def test_count_is_scoped(db, owner, other_owner):
    DocumentRepository(db, owner.tenant_id).add(_document(owner.tenant_id, "Mine"))
    DocumentRepository(db, other_owner.tenant_id).add(
        _document(other_owner.tenant_id, "Theirs")
    )
    db.flush()
    assert DocumentRepository(db, owner.tenant_id).count() == 1
    assert DocumentRepository(db, other_owner.tenant_id).count() == 1


def test_get_returns_none_for_other_tenant(db, owner, other_owner):
    theirs = _document(other_owner.tenant_id, "Theirs")
    DocumentRepository(db, other_owner.tenant_id).add(theirs)
    db.flush()
    assert DocumentRepository(db, owner.tenant_id).get(theirs.id) is None


def test_delete_refuses_foreign_tenant(db, owner, other_owner):
    theirs = _document(other_owner.tenant_id, "Theirs")
    DocumentRepository(db, other_owner.tenant_id).add(theirs)
    db.flush()
    with pytest.raises(ValueError):
        DocumentRepository(db, owner.tenant_id).delete(theirs)
