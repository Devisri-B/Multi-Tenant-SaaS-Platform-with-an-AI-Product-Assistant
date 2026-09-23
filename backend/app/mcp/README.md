# Self-RAG Model Context Protocol (MCP) Server & Autonomous Agent

A production-grade implementation of the **Model Context Protocol (MCP)** exposing an adaptive **Self-RAG (Retrieval-Augmented Generation)** knowledge base as standard AI tools, accompanied by an autonomous **Agent** acting as an MCP Client over standard I/O (`stdio`).

---

## Architecture Overview

```mermaid
%%{init: {"flowchart": {"nodeSpacing": 35, "rankSpacing": 45, "curve": "basis"}, "themeVariables": {"fontSize": "15px"}}}%%
flowchart TB
    C["AI client<br/>Claude Desktop / agent.py"]
    C <-->|"JSON-RPC<br/>over stdio"| S["FastMCP server"]
    S --> G{{"Tenant<br/>allowed?"}}
    G -- "no" --> E(["TenantNotFound"])
    G -- "yes" --> T["3 tools<br/>list_workspaces<br/>semantic_search_chunks<br/>self_rag_query"]
    T --> P["Self-RAG pipeline"]
    T --> DB[("Postgres + pgvector")]
    P --> DB

    classDef io fill:#1e3a8a,stroke:#1e3a8a,color:#ffffff
    classDef step fill:#ffffff,stroke:#475569,color:#0f172a,stroke-width:1.5px
    classDef guard fill:#fde68a,stroke:#b45309,color:#451a03,stroke-width:2px
    classDef bad fill:#fee2e2,stroke:#b91c1c,color:#450a0a,stroke-width:2px
    classDef data fill:#dcfce7,stroke:#15803d,color:#052e16,stroke-width:2px
    class C io
    class S,T,P step
    class G guard
    class E bad
    class DB data
```

## Exposed MCP Tools

The FastMCP server exposes three distinct tools:

| Tool | Parameters | Description |
| :--- | :--- | :--- |
| `list_workspaces` | *(none)* | Discovers all active tenant workspaces (UUIDs, names, slugs, plans) so the agent does not need hardcoded IDs. |
| `semantic_search_chunks` | `query: str`, `tenant_id?: str`, `top_k?: int` | Performs direct semantic vector retrieval via pgvector without LLM answer generation. Returns chunk excerpts, titles, and cosine similarity scores. |
| `self_rag_query` | `question: str`, `tenant_id?: str`, `allow_web_search?: bool`, `top_k?: int` | Runs the full LangGraph Self-RAG workflow: query reformulation, document grading, DeBERTa-v3 NLI anti-hallucination verification, strict self-correction, and web search fallback. |

---

## Quickstart Guide

### 1. Requirements
Ensure dependencies are installed:
```bash
cd backend
pip install -r requirements.txt
```

### 2. Run the Autonomous Agent (MCP Client)
The agent will automatically spawn the FastMCP server as a subprocess over `stdio`, discover available tools, and execute a reasoning loop:

```bash
# Offline demonstration mode (verifies MCP handshake & tool execution without external API keys):
python -m app.mcp.agent --offline

# Custom query in offline mode:
python -m app.mcp.agent --question "What is the refund policy?" --offline

# Full LLM mode: Claude tool-calling loop (AsyncAnthropic) over MCP, loops until a final answer:
export ANTHROPIC_API_KEY="sk-ant-..."
python -m app.mcp.agent --question "Explain the document upload limits."
```

### 3. Run Automated Tests
```bash
PYTHONPATH=. pytest tests/test_mcp.py tests/test_mcp_agent.py -v
```

---

## Connect to Claude Desktop or Cursor

You can connect this MCP server directly to **Claude Desktop** or **Cursor** to query your private workspace documents straight from your IDE or chat window.

### Configuration for Claude Desktop
Add this to your `claude_desktop_config.json` (located at `~/Library/Application Support/Claude/claude_desktop_config.json` on macOS):

```json
{
  "mcpServers": {
    "self-rag-knowledge-server": {
      "command": "python",
      "args": [
        "-m",
        "app.mcp.server"
      ],
      "cwd": "/path/to/SAAS/backend",
      "env": {
        "DATABASE_URL": "postgresql+psycopg://postgres:postgres@localhost:5432/saas_db",
        "MCP_ALLOWED_TENANT_IDS": "<workspace-uuid>,<workspace-uuid>"
      }
    }
  }
}
```

When you restart Claude Desktop or Cursor, `list_workspaces`, `semantic_search_chunks`, and `self_rag_query` will be available as active tools.

---

## Tenant Isolation

The REST API scopes every query by the caller's JWT and membership. The MCP server has no caller identity (stdio has no auth layer), so isolation is enforced by configuration instead:

| Setting | Behaviour |
| :--- | :--- |
| `MCP_ALLOWED_TENANT_IDS` unset | Every active workspace is exposed. Acceptable only for a local stdio server; the server logs a warning at startup. |
| `MCP_ALLOWED_TENANT_IDS=<uuid>,<uuid>` | `list_workspaces` returns only those workspaces; the no-`tenant_id` fallback picks the first allowed one; an explicit `tenant_id` outside the list returns `TenantNotFound` **before** any retrieval or LLM call, and the response does not reveal whether that tenant exists. |

All three tools resolve their workspace through the same `_allowed_tenants_query` helper (`server.py`), so a new tool cannot accidentally bypass the allowlist. Underneath that, `TenantScopedRepository` and the pgvector query still apply `WHERE tenant_id = :id`, so the allowlist is an outer gate rather than the only one.
