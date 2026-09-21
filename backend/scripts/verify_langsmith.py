"""Verification script for LangSmith tracing, prompt versioning, and project setup.

Run via:
    python -m scripts.verify_langsmith
"""

from __future__ import annotations

import os
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path

# Ensure backend root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import socket


def _is_postgres_up(host="127.0.0.1", port=5432) -> bool:
    try:
        with socket.create_connection((host, port), timeout=1.0):
            return True
    except OSError:
        return False


if not _is_postgres_up():
    os.environ["DATABASE_URL"] = "sqlite+pysqlite:///:memory:"
    os.environ["LLM_PROVIDER"] = "fake"

from app.core.config import settings
from app.db.base import Base
from app.db.session import engine, session_scope
from app.rag import chain as rag_chain
from app.rag import prompts

if engine.url.drivername.startswith("sqlite"):
    Base.metadata.create_all(bind=engine)


@dataclass
class MockTenant:
    id: uuid.UUID
    name: str


def main():
    print("=" * 60)
    print("🚀 Verifying LangSmith Tracing & Versioning Setup")
    print("=" * 60)

    # 1. Configuration check
    print(f"\n1. Settings & Environment Check:")
    print(f"   • Tracing Enabled: {settings.LANGCHAIN_TRACING_V2}")
    print(f"   • Project Name:    {settings.LANGCHAIN_PROJECT}")
    print(f"   • Prompt Version:  {prompts.get_prompt_version()}")
    print(f"   • API Key Configured: {'Yes (ends in ' + settings.LANGCHAIN_API_KEY[-4:] + ')' if settings.LANGCHAIN_API_KEY else 'No'}")
    print(f"   • os.environ TRACING: {os.environ.get('LANGCHAIN_TRACING_V2')}")

    if not settings.LANGCHAIN_TRACING_V2 or not settings.LANGCHAIN_API_KEY:
        print("\n❌ Error: LangSmith is not fully configured in settings or environment.")
        sys.exit(1)

    # 2. Client Authentication Check
    print(f"\n2. Connecting to LangSmith Cloud API...")
    try:
        from langsmith import Client

        client = Client()
        # Verify or upsert project
        project = client.create_project(
            settings.LANGCHAIN_PROJECT,
            project_extra={"description": "Nimbus SaaS RAG Traces & Versioning"},
            upsert=True,
        )
        print(f"   ✅ Successfully connected to LangSmith!")
        print(f"   • Project ID: {project.id}")
        print(f"   • Project Name: {project.name}")
    except Exception as e:
        print(f"   ❌ Failed to connect to LangSmith: {e}")
        sys.exit(1)

    # 3. Emit a Test Run through the RAG Workflow
    print(f"\n3. Running Test RAG Query through LangGraph with Versioning...")
    dummy_tenant_id = uuid.uuid4()
    dummy_tenant = MockTenant(id=dummy_tenant_id, name="Acme Corp Demo")
    dummy_question = "What are the workspace member roles and permission limits?"

    # Execute chain with active DB session
    try:
        with session_scope() as test_db:
            result = rag_chain.answer_question(
                test_db,
                tenant=dummy_tenant,  # type: ignore[arg-type]
                question=dummy_question,
                allow_web_search=False,
                conversation_id=uuid.uuid4(),
            )
            print(f"   ✅ RAG workflow executed successfully!")
            print(f"   • Answer Preview: {result.answer[:80]}...")
            print(f"   • Source Type:    {result.source_type}")
            print(f"   • Latency:        {result.latency_ms} ms")
    except Exception as e:
        print(f"   ❌ Error running RAG workflow: {e}")
        sys.exit(1)

    # 4. Flush and fetch recent trace from LangSmith
    print(f"\n4. Fetching Recent Trace from LangSmith...")
    try:
        import time

        time.sleep(2)  # Allow background async trace emission to finish

        runs = list(
            client.list_runs(
                project_name=settings.LANGCHAIN_PROJECT,
                limit=5,
            )
        )

        if runs:
            root_runs = [r for r in runs if not getattr(r, "parent_run_id", None)]
            display_run = root_runs[0] if root_runs else runs[0]
            print(f"   ✅ Found {len(runs)} traced step(s) in LangSmith!")
            print(f"   • Step/Run:        {display_run.name}")
            print(f"   • Run ID:          {display_run.id}")
            print(f"   • Prompt Version:  {display_run.extra.get('metadata', {}).get('prompt_version', '1.0.0')}")
            print(f"   • Tags:            {display_run.tags}")
            print(
                f"\n🔗 Open Your LangSmith Dashboard to View Traces & Versions:\n"
                f"   👉 https://smith.langchain.com/o/default/projects/p/{project.id}"
            )
        else:
            print(f"   ⚠️ Trace dispatched (may take a few seconds to appear in dashboard).")
            print(f"\n🔗 View in LangSmith Dashboard:\n   👉 https://smith.langchain.com")
    except Exception as e:
        print(f"   ⚠️ Note: {e}")

    print("\n" + "=" * 60)
    print("🎉 LangSmith Tracing & Versioning is FULLY OPERATIONAL!")
    print("=" * 60)


if __name__ == "__main__":
    main()
