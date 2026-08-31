"""Product-documentation endpoints."""

from __future__ import annotations

import uuid

from fastapi import APIRouter, File, Form, Query, UploadFile, status

from app.api.deps import DbSession, RequireMember, RequireViewer
from app.models.enums import DocumentStatus
from app.schemas.common import MessageResponse, Page
from app.schemas.document import (
    DocumentChunkRead,
    DocumentCreate,
    DocumentDetail,
    DocumentRead,
    DocumentUpdate,
    ReindexResponse,
)
from app.services import audit as audit_service
from app.services import document as document_service

router = APIRouter(prefix="/workspaces/{tenant_id}/documents", tags=["documents"])


@router.get("", response_model=Page[DocumentRead])
def list_documents(
    tenant_id: uuid.UUID,
    db: DbSession,
    context: RequireViewer,
    page: int = Query(1, ge=1),
    size: int = Query(20, ge=1, le=100),
    document_status: DocumentStatus | None = Query(None, alias="status"),
) -> Page[DocumentRead]:
    repo = document_service.DocumentRepository(db, context.tenant_id)
    items, total = repo.list_with_total(
        offset=(page - 1) * size, limit=size, status=document_status
    )
    return Page(
        items=[DocumentRead.model_validate(item) for item in items],
        total=total,
        page=page,
        size=size,
    )


@router.post("", response_model=DocumentRead, status_code=status.HTTP_201_CREATED)
def create_document_from_text(
    tenant_id: uuid.UUID,
    payload: DocumentCreate,
    db: DbSession,
    context: RequireMember,
) -> DocumentRead:
    document = document_service.create_document_from_text(
        db,
        tenant=context.tenant,
        uploader=context.user,
        title=payload.title,
        content=payload.content,
        metadata=payload.metadata,
    )
    audit_service.record(
        db,
        action="document.created",
        tenant_id=context.tenant_id,
        actor_id=context.user.id,
        target_type="document",
        target_id=document.id,
    )
    return DocumentRead.model_validate(document)


@router.post("/upload", response_model=DocumentRead, status_code=status.HTTP_201_CREATED)
async def upload_document(
    tenant_id: uuid.UUID,
    db: DbSession,
    context: RequireMember,
    file: UploadFile = File(...),
    title: str | None = Form(default=None),
) -> DocumentRead:
    payload = await file.read()
    document = document_service.create_document(
        db,
        tenant=context.tenant,
        uploader=context.user,
        title=title or file.filename or "Untitled document",
        payload=payload,
        source_name=file.filename or "upload.txt",
        content_type=file.content_type or "text/plain",
    )
    audit_service.record(
        db,
        action="document.uploaded",
        tenant_id=context.tenant_id,
        actor_id=context.user.id,
        target_type="document",
        target_id=document.id,
        context={"bytes": len(payload)},
    )
    return DocumentRead.model_validate(document)


@router.get("/{document_id}", response_model=DocumentDetail)
def get_document(
    tenant_id: uuid.UUID, document_id: uuid.UUID, db: DbSession, context: RequireViewer
) -> DocumentDetail:
    repo = document_service.DocumentRepository(db, context.tenant_id)
    document = repo.get_or_404(document_id)
    chunks = repo.chunks(document_id)
    content = "\n\n".join(chunk.content for chunk in chunks)
    return DocumentDetail(
        id=document.id,
        tenant_id=document.tenant_id,
        title=document.title,
        source_name=document.source_name,
        content_type=document.content_type,
        status=document.status,
        byte_size=document.byte_size,
        chunk_count=document.chunk_count,
        error_message=document.error_message,
        doc_metadata=document.doc_metadata,
        created_at=document.created_at,
        content=content,
        chunks=[DocumentChunkRead.model_validate(chunk) for chunk in chunks],
    )


@router.patch("/{document_id}", response_model=DocumentDetail)
def update_document(
    tenant_id: uuid.UUID,
    document_id: uuid.UUID,
    payload: DocumentUpdate,
    db: DbSession,
    context: RequireMember,
) -> DocumentDetail:
    repo = document_service.DocumentRepository(db, context.tenant_id)
    document = repo.get_or_404(document_id)
    updated_doc = document_service.update_document(
        db,
        tenant=context.tenant,
        document=document,
        title=payload.title,
        content=payload.content,
    )
    chunks = repo.chunks(document_id)
    content = "\n\n".join(chunk.content for chunk in chunks)
    audit_service.record(
        db,
        action="document.updated",
        tenant_id=context.tenant_id,
        actor_id=context.user.id,
        target_type="document",
        target_id=document.id,
        context={
            "title_updated": payload.title is not None,
            "content_updated": payload.content is not None,
        },
    )
    return DocumentDetail(
        id=updated_doc.id,
        tenant_id=updated_doc.tenant_id,
        title=updated_doc.title,
        source_name=updated_doc.source_name,
        content_type=updated_doc.content_type,
        status=updated_doc.status,
        byte_size=updated_doc.byte_size,
        chunk_count=updated_doc.chunk_count,
        error_message=updated_doc.error_message,
        doc_metadata=updated_doc.doc_metadata,
        created_at=updated_doc.created_at,
        content=content,
        chunks=[DocumentChunkRead.model_validate(chunk) for chunk in chunks],
    )


@router.get("/{document_id}/chunks", response_model=list[DocumentChunkRead])
def get_document_chunks(
    tenant_id: uuid.UUID, document_id: uuid.UUID, db: DbSession, context: RequireViewer
) -> list[DocumentChunkRead]:
    repo = document_service.DocumentRepository(db, context.tenant_id)
    repo.get_or_404(document_id)
    return [DocumentChunkRead.model_validate(chunk) for chunk in repo.chunks(document_id)]


@router.post("/{document_id}/reindex", response_model=ReindexResponse)
def reindex_document(
    tenant_id: uuid.UUID, document_id: uuid.UUID, db: DbSession, context: RequireMember
) -> ReindexResponse:
    repo = document_service.DocumentRepository(db, context.tenant_id)
    document = repo.get_or_404(document_id)
    document = document_service.reindex_document(
        db, tenant=context.tenant, document=document
    )
    return ReindexResponse(
        document_id=document.id, status=document.status, chunk_count=document.chunk_count
    )


@router.delete("/{document_id}", response_model=MessageResponse)
def delete_document(
    tenant_id: uuid.UUID, document_id: uuid.UUID, db: DbSession, context: RequireMember
) -> MessageResponse:
    repo = document_service.DocumentRepository(db, context.tenant_id)
    document = repo.get_or_404(document_id)
    document_service.delete_document(db, tenant=context.tenant, document=document)
    audit_service.record(
        db,
        action="document.deleted",
        tenant_id=context.tenant_id,
        actor_id=context.user.id,
        target_type="document",
        target_id=document_id,
    )
    return MessageResponse(message="Document deleted.")
