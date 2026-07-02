"""Domain-level exceptions mapped to HTTP responses by the app error handlers."""

from __future__ import annotations

from fastapi import status


class AppError(Exception):
    """Base class for every error the application raises deliberately."""

    status_code: int = status.HTTP_400_BAD_REQUEST
    code: str = "app_error"
    message: str = "Something went wrong."

    def __init__(self, message: str | None = None, *, code: str | None = None) -> None:
        self.message = message or self.message
        self.code = code or self.code
        super().__init__(self.message)

    def to_dict(self) -> dict[str, str]:
        return {"code": self.code, "message": self.message}


class NotFoundError(AppError):
    status_code = status.HTTP_404_NOT_FOUND
    code = "not_found"
    message = "The requested resource does not exist."


class ConflictError(AppError):
    status_code = status.HTTP_409_CONFLICT
    code = "conflict"
    message = "The resource already exists."


class AuthenticationError(AppError):
    status_code = status.HTTP_401_UNAUTHORIZED
    code = "unauthenticated"
    message = "Could not validate credentials."


class PermissionDeniedError(AppError):
    status_code = status.HTTP_403_FORBIDDEN
    code = "permission_denied"
    message = "You do not have permission to perform this action."


class ValidationError(AppError):
    status_code = status.HTTP_422_UNPROCESSABLE_ENTITY
    code = "validation_error"
    message = "The submitted payload is invalid."


class RateLimitError(AppError):
    status_code = status.HTTP_429_TOO_MANY_REQUESTS
    code = "rate_limited"
    message = "Too many requests. Please slow down."


class ProviderError(AppError):
    status_code = status.HTTP_502_BAD_GATEWAY
    code = "provider_error"
    message = "An upstream provider failed to respond."
