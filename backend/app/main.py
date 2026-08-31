"""FastAPI application factory."""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.api.middleware import RequestContextMiddleware, SecurityHeadersMiddleware
from app.api.v1.router import api_router
from app.core.config import settings
from app.core.exceptions import AppError
from app.core.logging import configure_logging, get_logger

log = get_logger(__name__)

DESCRIPTION = """
Multi-tenant SaaS platform with an AI-powered product assistant.

* **Workspaces** — every product lives in an isolated workspace backed by a
  shared Postgres schema and a tenant-scoped repository layer.
* **Auth** — JWT access/refresh tokens with role-based authorization
  (`viewer` < `member` < `admin` < `owner`).
* **Assistant** — a LangChain RAG pipeline that answers questions from the
  workspace's own product documentation, with citations.
"""


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()
    log.info(
        "app.startup",
        environment=settings.ENVIRONMENT,
        llm_provider=settings.LLM_PROVIDER,
    )
    yield
    log.info("app.shutdown")


def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.APP_NAME,
        description=DESCRIPTION,
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url=f"{settings.API_V1_PREFIX}/openapi.json",
        lifespan=lifespan,
    )

    app.add_middleware(SecurityHeadersMiddleware)
    app.add_middleware(RequestContextMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.BACKEND_CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["X-Request-Id"],
    )

    app.include_router(api_router, prefix=settings.API_V1_PREFIX)

    @app.exception_handler(AppError)
    async def handle_app_error(request: Request, exc: AppError) -> JSONResponse:
        return JSONResponse(status_code=exc.status_code, content=exc.to_dict())

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        errors = [
            {
                "loc": [str(part) for part in error.get("loc", ())],
                "type": error.get("type", "value_error"),
                # Pydantic prefixes custom validator failures with "Value error, ";
                # the raw sentence reads better in a form field.
                "message": str(error.get("msg", "Invalid value.")).removeprefix(
                    "Value error, "
                ),
            }
            for error in exc.errors()
        ]
        return JSONResponse(
            status_code=422,
            content={
                "code": "validation_error",
                "message": "The submitted payload is invalid.",
                "detail": {"errors": errors},
            },
        )

    static_dir = Path(os.getenv("STATIC_DIR", "/app/static"))
    if not static_dir.exists():
        local_dist = Path(__file__).resolve().parent.parent.parent / "frontend" / "dist"
        if local_dist.exists() and (local_dist / "index.html").exists():
            static_dir = local_dist

    has_spa = (
        settings.ENVIRONMENT != "test"
        and static_dir.exists()
        and (static_dir / "index.html").exists()
    )
    if has_spa:
        assets_dir = static_dir / "assets"
        if assets_dir.exists():
            app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="assets")

        @app.get("/{full_path:path}", include_in_schema=False)
        async def serve_spa(full_path: str) -> FileResponse | JSONResponse:
            if (
                full_path.startswith("api/")
                or full_path.startswith("docs")
                or full_path.startswith("redoc")
            ):
                return JSONResponse(status_code=404, content={"message": "Not Found"})
            target = static_dir / full_path
            if target.is_file():
                return FileResponse(target)
            return FileResponse(static_dir / "index.html")
    else:
        @app.get("/", include_in_schema=False)
        async def root() -> dict:
            return {
                "name": settings.APP_NAME,
                "version": "0.1.0",
                "docs": "/docs",
                "api": settings.API_V1_PREFIX,
            }

    return app


app = create_app()
