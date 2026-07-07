"""Registration, login, token refresh and password management."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.exceptions import AuthenticationError, ConflictError, NotFoundError
from app.core.security import (
    create_access_token,
    create_refresh_token,
    decode_token,
    hash_password,
    verify_password,
)
from app.models.user import User
from app.schemas.auth import TokenPair

# A real bcrypt hash of a throwaway string.  Verifying against it when the
# account does not exist keeps the response time of a failed login independent
# of whether the email address is registered.
_TIMING_PLACEHOLDER_HASH = (
    "$2b$12$y9eCKxfpW9aRjBdwkYhkQOb3E9Y6qWnSI5.Udv8mx3Am4Iz9DdruK"
)


def get_user_by_email(db: Session, email: str) -> User | None:
    normalised = email.strip().lower()
    stmt = select(User).where(User.email == normalised)
    return db.execute(stmt).scalars().first()


def get_user(db: Session, user_id: uuid.UUID) -> User:
    user = db.get(User, user_id)
    if user is None or user.is_deleted:
        raise NotFoundError("User not found.")
    return user


def create_user(
    db: Session,
    *,
    email: str,
    password: str,
    full_name: str | None = None,
    is_superuser: bool = False,
) -> User:
    normalised = email.strip().lower()
    if get_user_by_email(db, normalised):
        raise ConflictError("An account with that email address already exists.")
    user = User(
        email=normalised,
        hashed_password=hash_password(password),
        full_name=full_name.strip() if full_name else None,
        is_superuser=is_superuser,
    )
    db.add(user)
    db.flush()
    return user


def authenticate(db: Session, *, email: str, password: str) -> User:
    """Verify credentials.

    The password check runs even when the account is missing so that response
    timing does not reveal whether an email address is registered.
    """
    user = get_user_by_email(db, email)
    hashed = user.hashed_password if user else _TIMING_PLACEHOLDER_HASH
    password_ok = verify_password(password, hashed)

    if not user or not password_ok:
        raise AuthenticationError("Incorrect email address or password.")
    if not user.is_active or user.is_deleted:
        raise AuthenticationError("This account has been deactivated.")

    user.last_login_at = datetime.now(timezone.utc)
    db.flush()
    return user


def issue_tokens(user: User) -> TokenPair:
    subject = str(user.id)
    return TokenPair(
        access_token=create_access_token(subject, {"email": user.email}),
        refresh_token=create_refresh_token(subject),
        expires_in=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
    )


def refresh_tokens(db: Session, refresh_token: str) -> TokenPair:
    payload = decode_token(refresh_token, expected_type="refresh")
    try:
        user_id = uuid.UUID(payload["sub"])
    except (KeyError, ValueError):
        raise AuthenticationError("Refresh token has an invalid subject.")

    user = db.get(User, user_id)
    if user is None or not user.is_active or user.is_deleted:
        raise AuthenticationError("The account for this token is no longer active.")
    return issue_tokens(user)


def change_password(
    db: Session, user: User, *, current_password: str, new_password: str
) -> None:
    if not verify_password(current_password, user.hashed_password):
        raise AuthenticationError("Current password is incorrect.")
    user.hashed_password = hash_password(new_password)
    db.flush()
