"""Autonomous Agent operating as a Model Context Protocol (MCP) Client.

Demonstrates:
1. Connecting to the Self-RAG MCP Server over stdio.
2. Dynamic discovery of exposed tools (list_workspaces, self_rag_query, semantic_search_chunks).
3. Tool execution over the MCP JSON-RPC protocol.
4. Autonomous reasoning loop (ReAct) with tool invocation and answer synthesis.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any, Protocol

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import CallToolResult, TextContent, Tool

try:
    from langsmith import traceable
except ImportError:  # pragma: no cover
    def traceable(name: str | None = None, run_type: str | None = None, **kwargs: Any):
        def decorator(func: Any) -> Any:
            return func

        return decorator

# Determine backend base directory
BACKEND_DIR = Path(__file__).resolve().parent.parent.parent

# Same default as ``app.core.config.Settings.ANTHROPIC_CHAT_MODEL``; kept as a
# plain env lookup so the agent process does not need to load the full settings.
DEFAULT_CHAT_MODEL = "claude-haiku-4-5"

# Hard ceiling on model <-> tool round-trips so a confused model cannot loop forever.
MAX_AGENT_TURNS = 8

# Token budget guard and tool call timeout defaults
DEFAULT_MAX_TOKEN_BUDGET = int(os.getenv("MCP_AGENT_MAX_TOKEN_BUDGET", "50000"))
DEFAULT_TOOL_TIMEOUT = float(os.getenv("MCP_TOOL_TIMEOUT_SECONDS", "30.0"))


def get_server_parameters() -> StdioServerParameters:
    """Configure parameters to spawn the MCP Server subprocess."""
    return StdioServerParameters(
        command=sys.executable,
        args=["-m", "app.mcp.server"],
        cwd=str(BACKEND_DIR),
        env={**os.environ, "PYTHONPATH": str(BACKEND_DIR)},
    )


def tool_result_text(result: CallToolResult) -> str:
    """Flatten an MCP tool result into a single string for the model.

    FastMCP tools in this project always return one ``TextContent`` block, but
    the protocol allows images, audio and embedded resources too, so only the
    text blocks are collected and everything else is described by type.
    """
    parts: list[str] = []
    for block in result.content:
        text = getattr(block, "text", None)
        if text is not None:
            parts.append(text)
        else:
            parts.append(f"[non-text {block.type} content omitted]")
    return "\n".join(parts) if parts else ""


class ToolCaller(Protocol):
    """The slice of ``ClientSession`` the agent loop depends on (mockable in tests)."""

    async def call_tool(
        self, name: str, arguments: dict[str, Any] | None = None
    ) -> CallToolResult: ...


@traceable(name="mcp_agent_workflow", run_type="chain")
async def execute_mcp_query(
    question: str,
    tenant_id: str | None = None,
    use_llm: bool = True,
    *,
    max_token_budget: int = DEFAULT_MAX_TOKEN_BUDGET,
    tool_timeout: float = DEFAULT_TOOL_TIMEOUT,
) -> None:
    """Run the autonomous agent flow connected to the Self-RAG MCP server."""
    server_params = get_server_parameters()

    print("\n" + "=" * 70)
    print("Starting Self-RAG MCP Agent")
    print("=" * 70)
    print(f"Connecting to MCP Server subprocess in: {BACKEND_DIR} ...")

    async with (
        stdio_client(server_params) as (read, write),
        ClientSession(read, write) as session,
    ):
        # 1. Initialize MCP Session handshake
        init_res = await session.initialize()
        srv_info = init_res.serverInfo
        print(f"Connected! MCP Server: {srv_info.name} (v{srv_info.version})")

        # 2. Dynamic Tool Discovery
        tools_response = await session.list_tools()
        available_tools = tools_response.tools
        print(f"\nDiscovered {len(available_tools)} MCP Tools:")
        for t in available_tools:
            first_line = (t.description or "").strip().splitlines()[:1]
            print(f"   - {t.name}: {first_line[0] if first_line else '(no description)'}")

        anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")

        # 3. Autonomous Reasoning Loop
        if use_llm and anthropic_api_key:
            await _run_anthropic_agent_loop(
                session,
                available_tools,
                question,
                tenant_id,
                max_token_budget=max_token_budget,
                tool_timeout=tool_timeout,
            )
        else:
            if use_llm and not anthropic_api_key:
                print("\nNo ANTHROPIC_API_KEY detected. Running in demonstration mode.")
            await _run_deterministic_agent_loop(
                session,
                question,
                tenant_id,
                tool_timeout=tool_timeout,
            )


@traceable(name="mcp_deterministic_loop", run_type="chain")
async def _run_deterministic_agent_loop(
    session: ToolCaller,
    question: str,
    tenant_id: str | None,
    *,
    tool_timeout: float = DEFAULT_TOOL_TIMEOUT,
) -> None:
    """Deterministic agent flow for offline testing or verifying MCP protocol."""
    @traceable(name="mcp_tool_call", run_type="tool")
    async def _call_tool(name: str, args: dict[str, Any]) -> CallToolResult:
        try:
            return await asyncio.wait_for(session.call_tool(name, args), timeout=tool_timeout)
        except asyncio.TimeoutError:
            return CallToolResult(
                content=[
                    TextContent(
                        type="text",
                        text=f"Error: Tool '{name}' timed out after {tool_timeout} seconds.",
                    )
                ],
                isError=True,
            )

    print("\n" + "-" * 70)
    print("Agent Step 1: Discovering active workspaces via 'list_workspaces'...")
    print("-" * 70)

    workspaces_res = await _call_tool("list_workspaces", {})
    workspaces_text = tool_result_text(workspaces_res)
    try:
        workspaces_data = json.loads(workspaces_text)
        workspaces = workspaces_data.get("workspaces", [])
        if workspaces:
            selected_tenant = tenant_id or workspaces[0]["id"]
            selected_name = next(
                (w["name"] for w in workspaces if w["id"] == selected_tenant),
                workspaces[0]["name"],
            )
            print(f"Found {len(workspaces)} workspace(s). Selected: '{selected_name}'")
        else:
            selected_tenant = tenant_id
            print("No workspaces found in database. Proceeding with default tenant.")
    except Exception:
        selected_tenant = tenant_id

    print("\n" + "-" * 70)
    print("Agent Step 2: Invoking MCP tool 'self_rag_query'...")
    print(f"   Question: {question}")
    print(f"   Target Workspace: {selected_tenant or 'auto-resolve'}")
    print("-" * 70)

    tool_args: dict[str, Any] = {"question": question, "allow_web_search": True}
    if selected_tenant:
        tool_args["tenant_id"] = selected_tenant

    result = await _call_tool("self_rag_query", tool_args)
    content_text = tool_result_text(result)

    try:
        data = json.loads(content_text)
        if "error" in data:
            print("\n" + "=" * 70)
            print(f"MCP Tool Notice [{data.get('error')}]:")
            print("=" * 70)
            print(f"Message: {data.get('message')}")
            return

        print("\n" + "=" * 70)
        print("Final Agent Response (Self-RAG MCP)")
        print("=" * 70)
        print(f"Workspace   : {data.get('workspace_name')} ({data.get('workspace_id')})")
        print(f"Source Type : {data.get('source_type')} (used_context={data.get('used_context')})")
        print(f"Latency     : {data.get('latency_ms')} ms")
        print("\nAnswer:\n")
        print(data.get("answer"))

        citations = data.get("citations", [])
        if citations:
            print("\nCitations:")
            for i, c in enumerate(citations, 1):
                title = c.get("document_title") or c.get("url") or "Document"
                excerpt = (c.get("excerpt") or "").replace("\n", " ")
                print(f" [{i}] {title}: {excerpt[:120]}...")
    except Exception:
        print("Raw MCP Output:")
        print(content_text)


def mcp_tools_to_anthropic(tools: list[Tool]) -> list[dict[str, Any]]:
    """Translate MCP tool descriptors into Anthropic ``tools`` entries.

    Both sides speak JSON Schema for parameters, so the translation is a rename:
    MCP ``inputSchema`` -> Anthropic ``input_schema``.
    """
    return [
        {
            "name": t.name,
            "description": t.description or "",
            "input_schema": t.inputSchema,
        }
        for t in tools
    ]


@traceable(name="mcp_react_agent_loop", run_type="chain")
async def run_agent_loop(
    client: Any,
    session: ToolCaller,
    tools: list[dict[str, Any]],
    question: str,
    *,
    system_prompt: str,
    model: str,
    max_turns: int = MAX_AGENT_TURNS,
    max_token_budget: int = DEFAULT_MAX_TOKEN_BUDGET,
    tool_timeout: float = DEFAULT_TOOL_TIMEOUT,
    on_tool_call: Any = None,
) -> str:
    """ReAct loop: let Claude call MCP tools until it produces a final answer.

    Every request carries the full ``tools`` list - the API rejects a history
    that contains ``tool_use``/``tool_result`` blocks without it - and the loop
    only exits on a non-``tool_use`` stop reason, so the model is free to chain
    ``list_workspaces`` -> ``self_rag_query`` (or several searches) in one run.

    ``client`` is an ``anthropic.AsyncAnthropic``; typed as ``Any`` so tests can
    substitute a scripted fake without importing the SDK's param types.
    """
    @traceable(name="mcp_tool_call", run_type="tool")
    async def _call_tool(name: str, args: dict[str, Any]) -> CallToolResult:
        try:
            return await asyncio.wait_for(session.call_tool(name, args), timeout=tool_timeout)
        except asyncio.TimeoutError:
            return CallToolResult(
                content=[
                    TextContent(
                        type="text",
                        text=f"Error: Tool '{name}' timed out after {tool_timeout} seconds.",
                    )
                ],
                isError=True,
            )

    messages: list[dict[str, Any]] = [{"role": "user", "content": question}]
    response = None
    cumulative_tokens = 0

    for _turn in range(max_turns):
        response = await client.messages.create(
            model=model,
            max_tokens=4096,
            system=system_prompt,
            messages=messages,
            tools=tools,
        )

        # Track cumulative token consumption across turns
        usage = getattr(response, "usage", None)
        if usage:
            in_toks = getattr(usage, "input_tokens", 0) or 0
            out_toks = getattr(usage, "output_tokens", 0) or 0
            cumulative_tokens += in_toks + out_toks
            if max_token_budget > 0 and cumulative_tokens > max_token_budget:
                return (
                    f"Agent stopped: Token budget exceeded ({cumulative_tokens} "
                    f"tokens used > {max_token_budget} limit)."
                )

        if response.stop_reason != "tool_use":
            break

        tool_uses = [b for b in response.content if b.type == "tool_use"]
        # The assistant turn must be echoed back verbatim, tool_use blocks included.
        messages.append({"role": "assistant", "content": response.content})

        async def _execute_single_tool(tu: Any) -> dict[str, Any]:
            fn_args = dict(tu.input or {})
            if on_tool_call is not None:
                on_tool_call(tu.name, fn_args)

            mcp_res = await _call_tool(tu.name, fn_args)
            block: dict[str, Any] = {
                "type": "tool_result",
                "tool_use_id": tu.id,
                "content": tool_result_text(mcp_res),
            }
            if mcp_res.isError:
                block["is_error"] = True
            return block

        # Dispatch tool calls concurrently using asyncio.gather with per-tool timeouts
        raw_results = await asyncio.gather(
            *[_execute_single_tool(tu) for tu in tool_uses],
            return_exceptions=True,
        )

        tool_results: list[dict[str, Any]] = []
        for tu, res in zip(tool_uses, raw_results, strict=True):
            if isinstance(res, Exception):
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": tu.id,
                        "content": f"Error: Tool execution failed: {res}",
                        "is_error": True,
                    }
                )
            else:
                tool_results.append(res)

        # All results for one assistant turn go back in a single user message.
        messages.append({"role": "user", "content": tool_results})
    else:
        return (
            f"Agent stopped after {max_turns} tool-calling turns without a final answer."
        )

    assert response is not None
    return "".join(b.text for b in response.content if b.type == "text")


@traceable(name="mcp_anthropic_agent_loop", run_type="chain")
async def _run_anthropic_agent_loop(
    session: ClientSession,
    available_tools: list[Tool],
    question: str,
    tenant_id: str | None,
    *,
    max_token_budget: int = DEFAULT_MAX_TOKEN_BUDGET,
    tool_timeout: float = DEFAULT_TOOL_TIMEOUT,
) -> None:
    """Full ReAct loop powered by Anthropic Claude tool calling over MCP."""
    import anthropic

    # AsyncAnthropic keeps the event loop free so the MCP stdio reader task can
    # keep servicing the session while a model request is in flight.
    client = anthropic.AsyncAnthropic()
    model = os.getenv("ANTHROPIC_CHAT_MODEL", DEFAULT_CHAT_MODEL)

    system_prompt = (
        "You are an executive research agent following the ReAct (Reason + Act) pattern.\n"
        "You have access to a Self-RAG Knowledge Base via Model Context Protocol (MCP) tools.\n\n"
        "For each step of your investigation:\n"
        "1. Thought: Briefly explain your reasoning and what specific information you need next.\n"
        "2. Action: Invoke the appropriate MCP tool.\n"
        "3. Observation: Analyze the returned tool results and decide if further "
        "retrieval is needed.\n\n"
        "Always query the MCP tools to verify facts before answering. Cite sources appropriately."
    )
    if tenant_id:
        system_prompt += f"\nThe current tenant UUID is {tenant_id}."

    print("\n" + "-" * 70)
    print(f"Agent Reasoning: Claude ({model}) tool-calling loop over MCP...")
    print("-" * 70)

    def _log_tool_call(name: str, args: dict[str, Any]) -> None:
        print(f"Agent invoking MCP tool: '{name}' with args: {args}")

    try:
        final_text = await run_agent_loop(
            client,
            session,
            mcp_tools_to_anthropic(available_tools),
            question,
            system_prompt=system_prompt,
            model=model,
            max_token_budget=max_token_budget,
            tool_timeout=tool_timeout,
            on_tool_call=_log_tool_call,
        )
    except anthropic.APIError as exc:
        print(f"Anthropic API error: {exc}. Falling back to deterministic mode.")
        await _run_deterministic_agent_loop(
            session,
            question,
            tenant_id,
            tool_timeout=tool_timeout,
        )
        return

    print("\n" + "=" * 70)
    print("Final Agent Response (Anthropic Claude)")
    print("=" * 70)
    print(final_text)


def main():
    parser = argparse.ArgumentParser(description="Self-RAG MCP Autonomous Agent")
    parser.add_argument(
        "--question",
        "-q",
        type=str,
        default="What are the key policies and features documented in this workspace?",
        help="The question or goal for the agent.",
    )
    parser.add_argument(
        "--tenant-id",
        "-t",
        type=str,
        default=None,
        help="Optional tenant UUID.",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Force deterministic mode without external LLM API calls.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_MAX_TOKEN_BUDGET,
        help=f"Cumulative token budget ceiling (default: {DEFAULT_MAX_TOKEN_BUDGET}).",
    )
    parser.add_argument(
        "--tool-timeout",
        type=float,
        default=DEFAULT_TOOL_TIMEOUT,
        help=f"Per-tool execution timeout in seconds (default: {DEFAULT_TOOL_TIMEOUT}s).",
    )
    args = parser.parse_args()

    asyncio.run(
        execute_mcp_query(
            question=args.question,
            tenant_id=args.tenant_id,
            use_llm=not args.offline,
            max_token_budget=args.max_tokens,
            tool_timeout=args.tool_timeout,
        )
    )


if __name__ == "__main__":
    main()
