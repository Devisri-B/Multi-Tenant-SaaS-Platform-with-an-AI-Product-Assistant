"""Seed a demo workspace with users and product documentation.

    python -m scripts.seed
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.db.base import Base  # noqa: E402
from app.db.session import engine, session_scope  # noqa: E402
from app.models import *  # noqa: E402,F401,F403
from app.models.enums import Role  # noqa: E402
from app.services import auth as auth_service  # noqa: E402
from app.services import document as document_service  # noqa: E402
from app.services import member as member_service  # noqa: E402
from app.services import tenant as tenant_service  # noqa: E402

DEMO_DOCS = {
    "Getting Started": """\
# Getting Started with Nimbus

## Creating your workspace
Every product you support lives in its own workspace. Sign up, name the
workspace after the product, and Nimbus provisions an isolated tenant for it.

## Inviting your team
Open Settings -> Members and choose Invite. Roles are viewer, member, admin and
owner. Viewers can read documentation and ask the assistant; members can also
upload documentation; admins manage members; owners control billing.

## Free plan limits
The free plan includes 5 seats and 50 documents per workspace. Upgrade to Pro
to raise both limits.
""",
    "Uploading Documentation": """\
# Uploading Documentation

## Supported formats
Nimbus accepts Markdown, plain text, CSV, JSON and PDF files up to 10 MB each.

## How indexing works
On upload, a document is split into overlapping chunks of roughly 900
characters. Each chunk is embedded and stored alongside your workspace id, so
retrieval never crosses a workspace boundary.

## Re-indexing
If you change your chunking settings, use the Reindex action on a document to
rebuild its embeddings. Indexing status is shown as pending, processing,
indexed or failed.
""",
    "Assistant FAQ": """\
# Product Assistant FAQ

## Why did the assistant say it could not find an answer?
The assistant only answers from documentation you have uploaded to that
workspace. If nothing scores above the relevance threshold it will tell you so
rather than guess.

## Where do the citations come from?
Every answer lists the document chunks used to produce it, with a relevance
score. Click a citation to open the source document at that position.

## Can the assistant see other workspaces?
No. Retrieval is filtered by workspace id in SQL, and the API rejects requests
for a workspace the caller is not a member of.
""",
}


def main() -> None:
    Base.metadata.create_all(bind=engine)

    with session_scope() as db:
        if auth_service.get_user_by_email(db, "owner@nimbus.dev"):
            print("Demo data already present — nothing to do.")
            return

        owner = auth_service.create_user(
            db,
            email="owner@nimbus.dev",
            password="DemoPassw0rd",
            full_name="Ada Owner",
            is_superuser=True,
        )
        workspace = tenant_service.create_tenant(
            db, name="Nimbus Analytics", owner=owner
        )

        for email, name, role in [
            ("admin@nimbus.dev", "Ravi Admin", Role.ADMIN),
            ("member@nimbus.dev", "Mia Member", Role.MEMBER),
            ("viewer@nimbus.dev", "Vic Viewer", Role.VIEWER),
        ]:
            member_service.invite_member(
                db,
                tenant=workspace,
                email=email,
                role=role,
                full_name=name,
                invited_by=owner,
            )

        for title, content in DEMO_DOCS.items():
            document_service.create_document_from_text(
                db, tenant=workspace, uploader=owner, title=title, content=content
            )

        second = tenant_service.create_tenant(db, name="Orbit CRM", owner=owner)
        document_service.create_document_from_text(
            db,
            tenant=second,
            uploader=owner,
            title="Orbit Pipeline Stages",
            content=(
                "# Orbit Pipeline Stages\n\nOrbit CRM deals move through "
                "Prospect, Qualified, Proposal, Negotiation and Closed Won.\n"
            ),
        )

        print(f"Seeded workspace '{workspace.name}' ({workspace.slug}).")
        print(f"Seeded workspace '{second.name}' ({second.slug}).")
        print("Login with owner@nimbus.dev / DemoPassw0rd")


if __name__ == "__main__":
    main()
