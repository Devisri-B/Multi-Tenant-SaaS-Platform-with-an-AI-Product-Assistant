# Self-RAG Model Context Protocol (MCP) Server & Autonomous Agent

A production-grade implementation of the **Model Context Protocol (MCP)** exposing an adaptive **Self-RAG (Retrieval-Augmented Generation)** knowledge base as standard AI tools, accompanied by an autonomous **Agent** acting as an MCP Client.

---

## Architecture Overview

```mermaid
flowchart LR
    subgraph Host ["Agent / Host Layer"]
        Agent["Autonomous Agent<br/>(app/mcp/agent.py)"]
        Claude["Claude Desktop / Cursor"]
    end

    subgraph MCP ["Model Context Protocol (JSON-RPC)"]
        Stdio["Standard I/O (stdio) Transport"]
        Discovery["Dynamic Tool Discovery<br/>(list_tools)"]
    end

    subgraph Server ["FastMCP Server (app/mcp/server.py)"]
        T1["list_workspaces"]
        T2["semantic_search_chunks"]
        T3["self_rag_query"]
    end

    subgraph Pipeline ["Self-RAG Pipeline (app/rag/)"]
        Graph["LangGraph Workflow"]
        PGVector["pgvector Store"]
        Graders["Hallucination & Relevance Graders"]
        Web["DuckDuckGo Web Search Fallback"]
    end

    Agent <--> Stdio
    Claude <--> Stdio
    Stdio <--> Server
    Server --> T1 & T2 & T3
    T2 --> PGVector
    T3 --> Graph
    Graph --> PGVector & Graders & Web
```

---

## Exposed MCP Tools

The FastMCP server exposes three distinct tools:

| Tool | Parameters | Description |
| :--- | :--- | :--- |
| `list_workspaces` | *(none)* | Discovers all active tenant workspaces (UUIDs, names, slugs, plans) so the agent does not need hardcoded IDs. |
| `semantic_search_chunks` | `query: str`, `tenant_id?: str`, `top_k?: int` | Performs direct semantic vector retrieval via pgvector without LLM answer generation. Returns chunk excerpts, titles, and cosine similarity scores. |
| `self_rag_query` | `question: str`, `tenant_id?: str`, `allow_web_search?: bool`, `top_k?: int` | Runs the full LangGraph Self-RAG workflow: query reformulation, document grading, anti-hallucination verification, strict self-correction, and web search fallback. |

---

## Quickstart Guide

### 1. Requirements
Ensure dependencies are installed:
```bash
cd backend
pip install -r requirements.txt
```

### 2. Run the Autonomous Agent (MCP Client)
The agent will automatically spawn the MCP server as a subprocess, discover available tools, and execute a reasoning loop:

```bash
# Offline demonstration mode (verifies MCP handshake & tool execution without external API keys):
python -m app.mcp.agent --offline

# Custom query:
python -m app.mcp.agent --question "What is the refund policy?" --offline

# Full LLM Mode (using SyncAnthropic Claude tool calling via MCP):
export ANTHROPIC_API_KEY="sk-ant-..."
python -m app.mcp.agent --question "Explain the document upload limits."
```

### 3. Run Automated Tests
```bash
pytest tests/test_mcp.py -v
```

---

## Connect to Claude Desktop or Cursor

You can connect this MCP server directly to **Claude Desktop** or **Cursor** to query your private workspace documents straight from your IDE or chat window!

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
      "cwd": "/Users/devisri/Desktop/Research/Projects/SAAS/backend",
      "env": {
        "DATABASE_URL": "postgresql+psycopg://postgres:postgres@localhost:5432/saas_db"
      }
    }
  }
}
```

When you restart Claude Desktop or Cursor, you will see a hammer icon (⚒️) with `list_workspaces`, `semantic_search_chunks`, and `self_rag_query` active and callable!
