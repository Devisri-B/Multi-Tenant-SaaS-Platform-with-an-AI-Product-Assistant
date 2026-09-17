"""Model Context Protocol (MCP) Server exposing the multi-tenant Self-RAG pipeline.

Exposes three tools via the standard MCP protocol:
1. `list_workspaces`: Discover active tenant workspaces.
2. `semantic_search_chunks`: Direct vector retrieval via pgvector.
3. `self_rag_query`: Full LangGraph Self-RAG answering workflow with grading
   and hallucination reduction.

Tenant isolation: every tool resolves its workspace through
``_allowed_tenants_query``, which applies ``MCP_ALLOWED_TENANT_IDS`` when set.
A tenant outside the allowlist is treated exactly like one that does not exist,
so the server neither lists nor confirms the existence of other workspaces.
"""

from __future__ import annotations

import json
import logging
import sys
import uuid

# Re-route all root loggers and structlog to sys.stderr so stdout remains 100% clean for JSON-RPC
logging.getLogger().handlers.clear()
logging.basicConfig(
    stream=sys.stderr,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

try:
    import structlog

    structlog.configure(
        logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
    )
except Exception:
    pass

from mcp.server.fastmcp import FastMCP  # noqa: E402
from sqlalchemy.orm import Session  # noqa: E402

from app.core.config import settings  # noqa: E402
from app.db.session import session_scope  # noqa: E402
from app.models.tenant import Tenant  # noqa: E402
from app.rag import chain as rag_chain  # noqa: E402

logger = logging.getLogger("app.mcp.server")

# Initialize the FastMCP Server instance
mcp = FastMCP(
    name="Self-RAG Knowledge Server",
    instructions=(
        "You are connected to a multi-tenant Self-RAG Knowledge Server. "
        "Use 'list_workspaces' to discover available tenant workspaces. "
        "Use 'self_rag_query' to ask grounded questions with anti-hallucination verification. "
        "Use 'semantic_search_chunks' for raw semantic vector retrieval."
    ),
)


def _allowed_tenants_query(db: Session):
    """Active, non-deleted tenants, narrowed to ``MCP_ALLOWED_TENANT_IDS`` when set."""
    query = db.query(Tenant).filter(Tenant.deleted_at.is_(None), Tenant.is_active.is_(True))
    if settings.MCP_ALLOWED_TENANT_IDS:
        query = query.filter(Tenant.id.in_(settings.MCP_ALLOWED_TENANT_IDS))
    return query


def _resolve_tenant(db: Session, tenant_id_str: str | None) -> Tenant | None:
    """Resolve a tenant by UUID string or fallback to the first allowed tenant."""
    if tenant_id_str:
        try:
            tenant_uuid = uuid.UUID(tenant_id_str)
        except (ValueError, TypeError):
            return None
        return _allowed_tenants_query(db).filter(Tenant.id == tenant_uuid).first()

    # Fallback to the first allowed workspace if none was passed
    return _allowed_tenants_query(db).order_by(Tenant.created_at.asc()).first()


@mcp.tool(structured_output=False)
def list_workspaces() -> str:
    """List available tenant workspaces (names, UUIDs, slugs) that can be queried."""
    try:
        with session_scope() as db:
            tenants = _allowed_tenants_query(db).order_by(Tenant.created_at.asc()).all()
            workspaces = [
                {
                    "id": str(t.id),
                    "name": t.name,
                    "slug": t.slug,
                    "plan": t.plan.value if hasattr(t.plan, "value") else str(t.plan),
                    "document_limit": t.document_limit,
                }
                for t in tenants
            ]
            return json.dumps(
                {
                    "total_workspaces": len(workspaces),
                    "workspaces": workspaces,
                    "guidance": (
                        "Pass one of these workspace IDs as `tenant_id` to "
                        "`self_rag_query` or `semantic_search_chunks`."
                    ),
                },
                indent=2,
            )
    except Exception as exc:
        logger.error("list_workspaces.failed", exc_info=True)
        return json.dumps(
            {
                "error": "DatabaseUnavailable",
                "message": (
                    "Could not connect to database to list workspaces. "
                    "Ensure PostgreSQL is running (e.g., 'docker compose up -d db'). "
                    f"Details: {exc}"
                ),
            },
            indent=2,
        )


@mcp.tool(structured_output=False)
def semantic_search_chunks(
    query: str, tenant_id: str = "", top_k: int = 5
) -> str:
    """Retrieve raw document chunks matching a query via semantic vector search in pgvector.

    Args:
        query: The natural language search query.
        tenant_id: Optional UUID of the workspace. If omitted, uses the first active workspace.
        top_k: Number of chunks to retrieve (default: 5).
    """
    try:
        with session_scope() as db:
            target_tenant = tenant_id if tenant_id else None
            tenant = _resolve_tenant(db, target_tenant)
            if not tenant:
                return json.dumps(
                    {
                        "error": "TenantNotFound",
                        "message": (
                            f"No active tenant found matching ID '{tenant_id}'. "
                            "Call 'list_workspaces' to see available IDs."
                        ),
                    },
                    indent=2,
                )

            hits = rag_chain.semantic_search(
                db, tenant_id=tenant.id, query=query, top_k=top_k
            )
            results = [
                {
                    "chunk_id": str(c.chunk_id),
                    "document_id": str(c.document_id),
                    "document_title": c.document_title,
                    "ordinal": c.ordinal,
                    "score": round(c.score, 4),
                    "content": c.content,
                }
                for c in hits
            ]
            return json.dumps(
                {
                    "workspace_id": str(tenant.id),
                    "workspace_name": tenant.name,
                    "query": query,
                    "total_hits": len(results),
                    "chunks": results,
                },
                indent=2,
            )
    except Exception as exc:
        logger.error("semantic_search_chunks.failed", exc_info=True)
        return json.dumps(
            {
                "error": "RetrievalError",
                "message": f"Error performing semantic retrieval: {exc}",
            },
            indent=2,
        )


@mcp.tool(structured_output=False)
def self_rag_query(
    question: str,
    tenant_id: str = "",
    allow_web_search: bool = True,
    top_k: int = 5,
) -> str:
    """Answer questions using the full Self-RAG LangGraph pipeline.

    Features query contextualization, multi-tenant vector retrieval,
    document relevance grading, anti-hallucination consistency checks,
    strict self-correction, and web search fallback.

    Args:
        question: The user's question to answer.
        tenant_id: Optional UUID of the workspace. If omitted, uses the first active workspace.
        allow_web_search: Whether to fall back to web search if internal docs lack the answer.
        top_k: Number of document chunks to consider.
    """
    try:
        with session_scope() as db:
            target_tenant = tenant_id if tenant_id else None
            tenant = _resolve_tenant(db, target_tenant)
            if not tenant:
                return json.dumps(
                    {
                        "error": "TenantNotFound",
                        "message": (
                            f"No active tenant found matching ID '{tenant_id}'. "
                            "Call 'list_workspaces' to see available IDs."
                        ),
                    },
                    indent=2,
                )

            result = rag_chain.answer_question(
                db,
                tenant=tenant,
                question=question,
                top_k=top_k,
                allow_web_search=allow_web_search,
            )

            citations = [
                {
                    "document_title": c.get("document_title"),
                    "excerpt": c.get("excerpt"),
                    "url": c.get("url"),
                    "score": c.get("score"),
                    "source_type": c.get("source_type"),
                }
                for c in result.citations
            ]

            return json.dumps(
                {
                    "workspace_id": str(tenant.id),
                    "workspace_name": tenant.name,
                    "question": question,
                    "answer": result.answer,
                    "used_context": result.used_context,
                    "source_type": result.source_type,
                    "latency_ms": result.latency_ms,
                    "citations": citations,
                },
                indent=2,
            )
    except Exception as exc:
        logger.error("self_rag_query.failed", exc_info=True)
        return json.dumps(
            {
                "error": "PipelineError",
                "message": f"Error executing Self-RAG answering pipeline: {exc}",
            },
            indent=2,
        )


def main():
    """Run the FastMCP server over stdio transport."""
    logger.info("Starting FastMCP server '%s' on stdio transport...", mcp.name)
    if settings.MCP_ALLOWED_TENANT_IDS:
        logger.info(
            "Tenant allowlist active: %d workspace(s) exposed.",
            len(settings.MCP_ALLOWED_TENANT_IDS),
        )
    else:
        logger.warning(
            "MCP_ALLOWED_TENANT_IDS is unset - every active workspace is exposed. "
            "Set it before connecting anything other than a local client."
        )
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
