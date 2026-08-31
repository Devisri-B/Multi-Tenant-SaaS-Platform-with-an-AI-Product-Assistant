"""Document ingestion and lifecycle, scoped to a tenant."""

from __future__ import annotations

import uuid

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.exceptions import ConflictError, ValidationError
from app.models.document import Document, DocumentChunk
from app.models.enums import DocumentStatus
from app.models.tenant import Tenant
from app.models.user import User
from app.rag import ingest
from app.services import tenant as tenant_service
from app.services.base import TenantScopedRepository


class DocumentRepository(TenantScopedRepository[Document]):
    model = Document

    def by_checksum(self, digest: str) -> Document | None:
        stmt = self._scoped().where(Document.checksum == digest)
        return self.db.execute(stmt).scalars().first()

    def list_with_total(
        self, *, offset: int = 0, limit: int = 20, status: DocumentStatus | None = None
    ) -> tuple[list[Document], int]:
        stmt = self._scoped()
        count_stmt = (
            select(func.count())
            .select_from(Document)
            .where(Document.tenant_id == self.tenant_id)
        )
        if status is not None:
            stmt = stmt.where(Document.status == status)
            count_stmt = count_stmt.where(Document.status == status)
        stmt = stmt.order_by(Document.created_at.desc()).offset(offset).limit(limit)
        items = list(self.db.execute(stmt).scalars().all())
        total = int(self.db.execute(count_stmt).scalar_one())
        return items, total

    def chunks(self, document_id: uuid.UUID) -> list[DocumentChunk]:
        stmt = (
            select(DocumentChunk)
            .where(
                DocumentChunk.tenant_id == self.tenant_id,
                DocumentChunk.document_id == document_id,
            )
            .order_by(DocumentChunk.ordinal.asc())
        )
        return list(self.db.execute(stmt).scalars().all())


def create_document(
    db: Session,
    *,
    tenant: Tenant,
    uploader: User,
    title: str,
    payload: bytes,
    source_name: str,
    content_type: str = "text/plain",
) -> Document:
    """Persist an upload and index it in one unit of work."""
    if not payload:
        raise ValidationError("The uploaded document is empty.")
    if len(payload) > settings.MAX_UPLOAD_BYTES:
        raise ValidationError(
            f"Document exceeds the {settings.MAX_UPLOAD_BYTES // 1024 // 1024}MB limit."
        )

    tenant_service.assert_document_quota(db, tenant)

    repo = DocumentRepository(db, tenant.id)
    digest = ingest.checksum(payload)
    if repo.by_checksum(digest):
        raise ConflictError("An identical document has already been uploaded.")

    text = ingest.extract_text(payload, content_type, source_name)

    document = Document(
        tenant_id=tenant.id,
        uploaded_by_id=uploader.id,
        title=title.strip() or source_name,
        source_name=source_name,
        content_type=content_type or "text/plain",
        checksum=digest,
        byte_size=len(payload),
        status=DocumentStatus.PENDING,
        doc_metadata={"characters": len(text)},
    )
    repo.add(document)

    ingest.index_document(db, document, text)
    return document


def create_document_from_text(
    db: Session,
    *,
    tenant: Tenant,
    uploader: User,
    title: str,
    content: str,
    metadata: dict | None = None,
) -> Document:
    document = create_document(
        db,
        tenant=tenant,
        uploader=uploader,
        title=title,
        payload=content.encode("utf-8"),
        source_name=f"{title}.md",
        content_type="text/markdown",
    )
    if metadata:
        document.doc_metadata = {**document.doc_metadata, **metadata}
        db.flush()
    return document


def reindex_document(db: Session, *, tenant: Tenant, document: Document) -> Document:
    """Rebuild a document's chunks from the text already stored in them."""
    repo = DocumentRepository(db, tenant.id)
    existing = repo.chunks(document.id)
    if not existing:
        raise ConflictError(
            "This document has no stored text to re-index. Upload it again."
        )
    text = "\n\n".join(chunk.content for chunk in existing)
    return ingest.index_document(db, document, text)


def get_document_content(db: Session, *, tenant: Tenant, document: Document) -> str:
    """Retrieve full reconstructed text from stored document chunks."""
    repo = DocumentRepository(db, tenant.id)
    chunks = repo.chunks(document.id)
    return "\n\n".join(chunk.content for chunk in chunks)


def update_document(
    db: Session,
    *,
    tenant: Tenant,
    document: Document,
    title: str | None = None,
    content: str | None = None,
) -> Document:
    """Update title and/or re-chunk & re-embed updated document text."""
    repo = DocumentRepository(db, tenant.id)
    if title is not None and title.strip():
        document.title = title.strip()

    if content is not None:
        cleaned_content = content.strip()
        if not cleaned_content:
            raise ValidationError("Document content cannot be empty.")
        raw_bytes = cleaned_content.encode("utf-8")
        if len(raw_bytes) > settings.MAX_UPLOAD_BYTES:
            raise ValidationError(
                f"Document exceeds the {settings.MAX_UPLOAD_BYTES // 1024 // 1024}MB limit."
            )
        digest = ingest.checksum(raw_bytes)
        existing_doc = repo.by_checksum(digest)
        if existing_doc and existing_doc.id != document.id:
            raise ConflictError("An identical document already exists in this workspace.")

        document.checksum = digest
        document.byte_size = len(raw_bytes)
        document.doc_metadata = {**document.doc_metadata, "characters": len(cleaned_content)}
        ingest.index_document(db, document, cleaned_content)
    else:
        db.flush()

    return document


def delete_document(db: Session, *, tenant: Tenant, document: Document) -> None:
    DocumentRepository(db, tenant.id).delete(document)
