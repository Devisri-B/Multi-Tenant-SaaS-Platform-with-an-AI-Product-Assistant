"""Cross-cutting HTTP middleware."""

from __future__ import annotations

import time
import uuid

import structlog
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from app.core.logging import get_logger

log = get_logger("http")


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Attach a request id, bind log context, and record latency."""

    async def dispatch(self, request: Request, call_next) -> Response:
        request_id = request.headers.get("X-Request-Id") or uuid.uuid4().hex
        request.state.request_id = request_id

        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(
            request_id=request_id, path=request.url.path, method=request.method
        )

        started = time.perf_counter()
        response = await call_next(request)
        duration_ms = int((time.perf_counter() - started) * 1000)

        response.headers["X-Request-Id"] = request_id
        response.headers["X-Response-Time-Ms"] = str(duration_ms)

        log.info(
            "request.completed",
            status_code=response.status_code,
            duration_ms=duration_ms,
            tenant_id=getattr(request.state, "tenant_id", None),
        )
        return response


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Conservative security headers for the API surface."""

    async def dispatch(self, request: Request, call_next) -> Response:
        response = await call_next(request)
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("X-Frame-Options", "DENY")
        response.headers.setdefault("Referrer-Policy", "no-referrer")
        response.headers.setdefault(
            "Strict-Transport-Security", "max-age=31536000; includeSubDomains"
        )
        return response
