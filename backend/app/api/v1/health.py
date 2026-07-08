"""Liveness and readiness probes."""

from __future__ import annotations

from fastapi import APIRouter
from sqlalchemy import text

from app.api.deps import DbSession
from app.core.config import settings

router = APIRouter(tags=["health"])


@router.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "app": settings.APP_NAME,
        "environment": settings.ENVIRONMENT,
    }


@router.get("/health/ready")
def readiness(db: DbSession) -> dict:
    try:
        db.execute(text("SELECT 1"))
        database_ok = True
    except Exception:
        database_ok = False
    return {
        "status": "ok" if database_ok else "degraded",
        "database": "up" if database_ok else "down",
        "llm_provider": settings.LLM_PROVIDER,
    }
