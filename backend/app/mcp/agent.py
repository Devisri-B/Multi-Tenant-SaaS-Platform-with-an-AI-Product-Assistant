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
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# Determine backend base directory
BACKEND_DIR = Path(__file__).resolve().parent.parent.parent


def get_server_parameters() -> StdioServerParameters:
    """Configure parameters to spawn the MCP Server subprocess."""
    return StdioServerParameters(
        command=sys.executable,
        args=["-m", "app.mcp.server"],
        cwd=str(BACKEND_DIR),
        env={**os.environ, "PYTHONPATH": str(BACKEND_DIR)},
    )


async def execute_mcp_query(
    question: str,
    tenant_id: str | None = None,
    use_llm: bool = True,
) -> None:
    """Run the autonomous agent flow connected to the Self-RAG MCP server."""
    server_params = get_server_parameters()

    print("\n" + "=" * 70)
    print("🤖  STARTING SELF-RAG MCP AGENT")
    print("=" * 70)
    print(f"Connecting to MCP Server subprocess in: {BACKEND_DIR} ...")

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # 1. Initialize MCP Session handshake
            init_res = await session.initialize()
            print(f" Connected! MCP Server Name: {init_res.serverInfo.name} (v{init_res.serverInfo.version})")

            # 2. Dynamic Tool Discovery
            tools_response = await session.list_tools()
            available_tools = tools_response.tools
            print(f"\n Discovered {len(available_tools)} MCP Tools:")
            for t in available_tools:
                print(f"   • {t.name}: {t.description.strip().splitlines()[0]}")

            anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
            openai_api_key = os.getenv("OPENAI_API_KEY")

            # 3. Autonomous Reasoning Loop
            if use_llm and anthropic_api_key:
                await _run_anthropic_agent_loop(session, available_tools, question, tenant_id, anthropic_api_key)
            elif use_llm and openai_api_key:
                await _run_openai_agent_loop(session, available_tools, question, tenant_id)
            else:
                if use_llm and not anthropic_api_key and not openai_api_key:
                    print("\n⚠️  Neither ANTHROPIC_API_KEY nor OPENAI_API_KEY detected. Running in deterministic Agent Demonstration Mode.")
                await _run_deterministic_agent_loop(session, question, tenant_id)


async def _run_deterministic_agent_loop(
    session: ClientSession,
    question: str,
    tenant_id: str | None,
) -> None:
    """Deterministic agent flow for offline testing or verifying MCP protocol."""
    print("\n" + "-" * 70)
    print("🧠 AGENT STEP 1: Discovering active workspaces via MCP tool 'list_workspaces'...")
    print("-" * 70)

    workspaces_res = await session.call_tool("list_workspaces", {})
    workspaces_text = workspaces_res.content[0].text
    try:
        workspaces_data = json.loads(workspaces_text)
        workspaces = workspaces_data.get("workspaces", [])
        if workspaces:
            selected_tenant = tenant_id or workspaces[0]["id"]
            selected_name = next(
                (w["name"] for w in workspaces if w["id"] == selected_tenant),
                workspaces[0]["name"],
            )
            print(f" Found {len(workspaces)} workspace(s). Selected: '{selected_name}' ({selected_tenant})")
        else:
            selected_tenant = tenant_id
            print(" No workspaces found in database yet. Proceeding with default tenant.")
    except Exception:
        selected_tenant = tenant_id

    print("\n" + "-" * 70)
    print("🧠 AGENT STEP 2: Invoking MCP tool 'self_rag_query'...")
    print(f"   Question: {question}")
    print(f"   Target Workspace: {selected_tenant or 'auto-resolve'}")
    print("-" * 70)

    tool_args: dict[str, Any] = {"question": question, "allow_web_search": True}
    if selected_tenant:
        tool_args["tenant_id"] = selected_tenant

    result = await session.call_tool("self_rag_query", tool_args)
    content_text = result.content[0].text

    try:
        data = json.loads(content_text)
        if "error" in data:
            print("\n" + "=" * 70)
            print(f"⚠️  MCP TOOL NOTICE [{data.get('error')}]")
            print("=" * 70)
            print(f"Message: {data.get('message')}")
            return

        print("\n" + "=" * 70)
        print("🎯 FINAL AGENT SYNTHESIS (Grounded via Self-RAG MCP)")
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


async def _run_anthropic_agent_loop(
    session: ClientSession,
    available_tools: list[Any],
    question: str,
    tenant_id: str | None,
    api_key: str,
) -> None:
    """Full ReAct loop powered by SyncAnthropic (Claude) tool calling over MCP."""
    try:
        from anthropic import Anthropic as SyncAnthropic

        client = SyncAnthropic(api_key=api_key)

        # Convert MCP tools to Anthropic tool definitions
        anthropic_tools = [
            {
                "name": t.name,
                "description": t.description,
                "input_schema": t.inputSchema,
            }
            for t in available_tools
        ]

        system_prompt = (
            "You are an executive research agent. You have access to a Self-RAG Knowledge Base "
            "via Model Context Protocol (MCP) tools. Always query the MCP tools to verify facts "
            "before answering. Cite sources appropriately."
        )
        if tenant_id:
            system_prompt += f" The current tenant UUID is {tenant_id}."

        messages: list[dict[str, Any]] = [{"role": "user", "content": question}]

        print("\n" + "-" * 70)
        print("🧠 AGENT REASONING: Planning tool invocation with SyncAnthropic (Claude)...")
        print("-" * 70)

        response = client.messages.create(
            model=os.getenv("ANTHROPIC_CHAT_MODEL", "claude-3-5-haiku-20241022"),
            max_tokens=1024,
            system=system_prompt,
            messages=messages,
            tools=anthropic_tools,
        )

        tool_uses = [c for c in response.content if c.type == "tool_use"]
        if tool_uses:
            messages.append({"role": "assistant", "content": response.content})
            tool_results = []
            for tool_use in tool_uses:
                fn_name = tool_use.name
                fn_args = tool_use.input or {}
                print(f" Agent invoking MCP tool: '{fn_name}' with args: {fn_args}")

                mcp_res = await session.call_tool(fn_name, fn_args)
                tool_output = mcp_res.content[0].text
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": tool_use.id,
                        "content": tool_output,
                    }
                )

            messages.append({"role": "user", "content": tool_results})

            final_res = client.messages.create(
                model=os.getenv("ANTHROPIC_CHAT_MODEL", "claude-3-5-haiku-20241022"),
                max_tokens=1024,
                system=system_prompt,
                messages=messages,
            )
            final_text = "".join(b.text for b in final_res.content if hasattr(b, "text"))
            print("\n" + "=" * 70)
            print("🎯 FINAL AGENT SYNTHESIS (Anthropic Claude)")
            print("=" * 70)
            print(final_text)
        else:
            final_text = "".join(b.text for b in response.content if hasattr(b, "text"))
            print("\n" + "=" * 70)
            print("🎯 AGENT RESPONSE (Direct Claude)")
            print("=" * 70)
            print(final_text)

    except Exception as exc:
        print(f"⚠️  Anthropic execution error: {exc}. Falling back to deterministic mode.")
        await _run_deterministic_agent_loop(session, question, tenant_id)


async def _run_openai_agent_loop(
    session: ClientSession,
    available_tools: list[Any],
    question: str,
    tenant_id: str | None,
) -> None:
    """Full ReAct loop powered by OpenAI tool calling over MCP."""
    try:
        from openai import OpenAI

        client = OpenAI()

        # Convert MCP tools to OpenAI function calling format
        openai_tools = [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.inputSchema,
                },
            }
            for t in available_tools
        ]

        system_prompt = (
            "You are an executive research agent. You have access to a Self-RAG Knowledge Base "
            "via Model Context Protocol (MCP) tools. Always query the MCP tools to verify facts "
            "before answering. Cite sources appropriately."
        )
        if tenant_id:
            system_prompt += f" The current tenant UUID is {tenant_id}."

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ]

        print("\n" + "-" * 70)
        print("🧠 AGENT REASONING: Planning tool invocation with LLM...")
        print("-" * 70)

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            tools=openai_tools,
            tool_choice="auto",
        )

        msg = response.choices[0].message
        if msg.tool_calls:
            messages.append(msg)
            for tool_call in msg.tool_calls:
                fn_name = tool_call.function.name
                fn_args = json.loads(tool_call.function.arguments)
                print(f" Agent invoking MCP tool: '{fn_name}' with args: {fn_args}")

                mcp_res = await session.call_tool(fn_name, fn_args)
                tool_output = mcp_res.content[0].text

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": tool_output,
                    }
                )

            # Synthesize final answer from tool outputs
            final_res = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
            )
            final_answer = final_res.choices[0].message.content
            print("\n" + "=" * 70)
            print("🎯 FINAL AGENT SYNTHESIS")
            print("=" * 70)
            print(final_answer)
        else:
            print("\n" + "=" * 70)
            print("🎯 AGENT RESPONSE (Direct)")
            print("=" * 70)
            print(msg.content)

    except Exception as exc:
        print(f"⚠️  LLM execution error: {exc}. Falling back to deterministic mode.")
        await _run_deterministic_agent_loop(session, question, tenant_id)


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
    args = parser.parse_args()

    asyncio.run(
        execute_mcp_query(
            question=args.question,
            tenant_id=args.tenant_id,
            use_llm=not args.offline,
        )
    )


if __name__ == "__main__":
    main()
