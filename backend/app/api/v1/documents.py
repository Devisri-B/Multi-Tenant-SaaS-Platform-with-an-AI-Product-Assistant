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
    DocumentRead,
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


@router.get("/{document_id}", response_model=DocumentRead)
def get_document(
    tenant_id: uuid.UUID, document_id: uuid.UUID, db: DbSession, context: RequireViewer
) -> DocumentRead:
    repo = document_service.DocumentRepository(db, context.tenant_id)
    return DocumentRead.model_validate(repo.get_or_404(document_id))


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
