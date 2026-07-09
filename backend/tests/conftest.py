"""Test fixtures: an in-memory database, an app client, and seeded tenants."""

from __future__ import annotations

import os
import uuid
from collections.abc import Iterator

import pytest

os.environ.setdefault("ENVIRONMENT", "test")
os.environ.setdefault("DATABASE_URL", "sqlite+pysqlite:///:memory:")
os.environ.setdefault("LLM_PROVIDER", "fake")
os.environ.setdefault("SECRET_KEY", "test-secret-key-not-used-in-production")
os.environ.setdefault("EMBEDDING_DIMENSIONS", "128")
os.environ.setdefault("RAG_MIN_SCORE", "0.0")

from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy.orm import Session  # noqa: E402

from app.db.base import Base  # noqa: E402
from app.db.session import SessionLocal, engine, get_db  # noqa: E402
from app.main import create_app  # noqa: E402
from app.models import *  # noqa: E402,F401,F403
from app.models.enums import Role  # noqa: E402
from app.services import auth as auth_service  # noqa: E402
from app.services import member as member_service  # noqa: E402
from app.services import tenant as tenant_service  # noqa: E402


@pytest.fixture(scope="session", autouse=True)
def _create_schema() -> Iterator[None]:
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)


@pytest.fixture(autouse=True)
def _clean_tables() -> Iterator[None]:
    yield
    with engine.begin() as connection:
        for table in reversed(Base.metadata.sorted_tables):
            connection.execute(table.delete())


@pytest.fixture
def db() -> Iterator[Session]:
    session = SessionLocal()
    try:
        yield session
        session.commit()
    finally:
        session.close()


@pytest.fixture
def client(db: Session) -> Iterator[TestClient]:
    app = create_app()

    def _override_get_db() -> Iterator[Session]:
        try:
            yield db
            db.commit()
        except Exception:
            db.rollback()
            raise

    app.dependency_overrides[get_db] = _override_get_db
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# Domain fixtures
# ---------------------------------------------------------------------------
class Actor:
    """A user plus the headers needed to act as them."""

    def __init__(self, user, tenant, token: str, role: Role) -> None:
        self.user = user
        self.tenant = tenant
        self.token = token
        self.role = role

    @property
    def headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.token}",
            "X-Workspace-Id": str(self.tenant.id),
        }

    @property
    def tenant_id(self) -> uuid.UUID:
        return self.tenant.id


def make_actor(db: Session, tenant, email: str, role: Role, inviter=None) -> Actor:
    if role is Role.OWNER:
        user = auth_service.create_user(
            db, email=email, password="OwnerPassw0rd", full_name="Owner"
        )
        return Actor(user, tenant, auth_service.issue_tokens(user).access_token, role)

    membership, _, _ = member_service.invite_member(
        db, tenant=tenant, email=email, role=role, full_name=None, invited_by=inviter
    )
    member_service.activate_membership(db, membership)
    user = membership.user
    return Actor(user, tenant, auth_service.issue_tokens(user).access_token, role)


@pytest.fixture
def owner(db: Session) -> Actor:
    user = auth_service.create_user(
        db, email="owner@acme.io", password="OwnerPassw0rd", full_name="Ada Owner"
    )
    tenant = tenant_service.create_tenant(db, name="Acme Docs", owner=user)
    db.commit()
    token = auth_service.issue_tokens(user).access_token
    return Actor(user, tenant, token, Role.OWNER)


@pytest.fixture
def admin(db: Session, owner: Actor) -> Actor:
    actor = make_actor(db, owner.tenant, "admin@acme.io", Role.ADMIN, owner.user)
    db.commit()
    return actor


@pytest.fixture
def member(db: Session, owner: Actor) -> Actor:
    actor = make_actor(db, owner.tenant, "member@acme.io", Role.MEMBER, owner.user)
    db.commit()
    return actor


@pytest.fixture
def viewer(db: Session, owner: Actor) -> Actor:
    actor = make_actor(db, owner.tenant, "viewer@acme.io", Role.VIEWER, owner.user)
    db.commit()
    return actor


@pytest.fixture
def other_owner(db: Session) -> Actor:
    """An owner of a completely separate workspace — used for isolation tests."""
    user = auth_service.create_user(
        db, email="rival@globex.io", password="RivalPassw0rd", full_name="Rival"
    )
    tenant = tenant_service.create_tenant(db, name="Globex Docs", owner=user)
    db.commit()
    return Actor(user, tenant, auth_service.issue_tokens(user).access_token, Role.OWNER)
