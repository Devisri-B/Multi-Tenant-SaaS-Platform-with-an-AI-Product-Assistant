"""Tests for the MCP client agent's Claude tool-calling loop.

The Anthropic client and the MCP session are both scripted fakes, so these
tests pin down the protocol shape of the loop (what gets sent on each turn)
without any network access.
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from typing import Any

import pytest
from mcp.types import CallToolResult, ImageContent, TextContent, Tool

from app.mcp import agent


def _text(text: str) -> Any:
    return SimpleNamespace(type="text", text=text)


def _tool_use(tool_id: str, name: str, **args: Any) -> Any:
    return SimpleNamespace(type="tool_use", id=tool_id, name=name, input=args)


def _response(stop_reason: str, *content: Any, usage: Any = None) -> Any:
    return SimpleNamespace(stop_reason=stop_reason, content=list(content), usage=usage)


class FakeMessages:
    """Replays a scripted list of responses and records every request."""

    def __init__(self, responses: list[Any]) -> None:
        self._responses = list(responses)
        self.requests: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> Any:
        self.requests.append(kwargs)
        return self._responses.pop(0)


class FakeAnthropic:
    def __init__(self, responses: list[Any]) -> None:
        self.messages = FakeMessages(responses)


class FakeSession:
    """Stands in for ``ClientSession``: returns canned JSON per tool name."""

    def __init__(self, results: dict[str, Any], *, error_tools: set[str] | None = None):
        self._results = results
        self._error_tools = error_tools or set()
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def call_tool(self, name: str, arguments: dict[str, Any] | None = None):
        self.calls.append((name, arguments or {}))
        return CallToolResult(
            content=[TextContent(type="text", text=json.dumps(self._results[name]))],
            isError=name in self._error_tools,
        )


TOOLS = [{"name": "list_workspaces", "description": "", "input_schema": {"type": "object"}}]


@pytest.mark.asyncio
async def test_loop_chains_multiple_tool_calls_and_keeps_tools_on_every_request():
    """list_workspaces -> self_rag_query -> final text, tools sent each turn."""
    client = FakeAnthropic(
        [
            _response("tool_use", _text("Let me look up workspaces."),
                      _tool_use("tu_1", "list_workspaces")),
            _response("tool_use", _tool_use("tu_2", "self_rag_query",
                                            question="refund policy", tenant_id="abc")),
            _response("end_turn", _text("Refunds take 10 business days.")),
        ]
    )
    session = FakeSession(
        {
            "list_workspaces": {"workspaces": [{"id": "abc", "name": "Acme"}]},
            "self_rag_query": {"answer": "Refunds take 10 business days."},
        }
    )
    seen: list[str] = []

    answer = await agent.run_agent_loop(
        client, session, TOOLS, "What is the refund policy?",
        system_prompt="sys", model="claude-haiku-4-5",
        on_tool_call=lambda name, _args: seen.append(name),
    )

    assert answer == "Refunds take 10 business days."
    assert seen == ["list_workspaces", "self_rag_query"]
    assert [c[0] for c in session.calls] == ["list_workspaces", "self_rag_query"]
    assert session.calls[1][1] == {"question": "refund policy", "tenant_id": "abc"}

    # Three API calls, each carrying the tool definitions.
    assert len(client.messages.requests) == 3
    assert all(req["tools"] == TOOLS for req in client.messages.requests)

    # The final request replays the whole transcript in the right shape.
    final_messages = client.messages.requests[-1]["messages"]
    assert [m["role"] for m in final_messages] == [
        "user", "assistant", "user", "assistant", "user"
    ]
    first_results = final_messages[2]["content"]
    assert first_results[0]["type"] == "tool_result"
    assert first_results[0]["tool_use_id"] == "tu_1"
    assert "Acme" in first_results[0]["content"]
    assert "is_error" not in first_results[0]


@pytest.mark.asyncio
async def test_loop_returns_text_directly_when_no_tool_needed():
    client = FakeAnthropic([_response("end_turn", _text("Hello!"))])
    session = FakeSession({})

    answer = await agent.run_agent_loop(
        client, session, TOOLS, "hi", system_prompt="sys", model="m"
    )

    assert answer == "Hello!"
    assert session.calls == []
    assert len(client.messages.requests) == 1


@pytest.mark.asyncio
async def test_loop_flags_mcp_errors_as_is_error_tool_results():
    client = FakeAnthropic(
        [
            _response("tool_use", _tool_use("tu_1", "self_rag_query", question="q")),
            _response("end_turn", _text("The workspace could not be found.")),
        ]
    )
    session = FakeSession(
        {"self_rag_query": {"error": "TenantNotFound"}}, error_tools={"self_rag_query"}
    )

    await agent.run_agent_loop(
        client, session, TOOLS, "q", system_prompt="sys", model="m"
    )

    tool_result = client.messages.requests[-1]["messages"][2]["content"][0]
    assert tool_result["is_error"] is True
    assert "TenantNotFound" in tool_result["content"]


@pytest.mark.asyncio
async def test_loop_stops_at_max_turns():
    looping = _response("tool_use", _tool_use("tu", "list_workspaces"))
    client = FakeAnthropic([looping] * 3)
    session = FakeSession({"list_workspaces": {"workspaces": []}})

    answer = await agent.run_agent_loop(
        client, session, TOOLS, "q", system_prompt="sys", model="m", max_turns=3
    )

    assert "stopped after 3" in answer
    assert len(client.messages.requests) == 3


def test_mcp_tools_to_anthropic_renames_schema_key():
    tool = Tool(
        name="semantic_search_chunks",
        description="Search chunks.",
        inputSchema={"type": "object", "properties": {"query": {"type": "string"}}},
    )
    (converted,) = agent.mcp_tools_to_anthropic([tool])
    assert converted == {
        "name": "semantic_search_chunks",
        "description": "Search chunks.",
        "input_schema": {"type": "object", "properties": {"query": {"type": "string"}}},
    }


def test_tool_result_text_skips_non_text_blocks():
    result = CallToolResult(
        content=[
            TextContent(type="text", text="hello"),
            ImageContent(type="image", data="aGk=", mimeType="image/png"),
        ]
    )
    assert agent.tool_result_text(result) == "hello\n[non-text image content omitted]"


@pytest.mark.asyncio
async def test_loop_stops_when_token_budget_exceeded():
    looping = _response(
        "tool_use",
        _tool_use("tu", "list_workspaces"),
        usage=SimpleNamespace(input_tokens=2000, output_tokens=1000),
    )
    client = FakeAnthropic([looping] * 5)
    session = FakeSession({"list_workspaces": {"workspaces": []}})

    answer = await agent.run_agent_loop(
        client,
        session,
        TOOLS,
        "q",
        system_prompt="sys",
        model="m",
        max_token_budget=5000,
    )

    assert "Token budget exceeded" in answer
    assert "6000 tokens used > 5000 limit" in answer
    # Exceeded on turn 2 (turn 1 = 3000, turn 2 = 6000)
    assert len(client.messages.requests) == 2


@pytest.mark.asyncio
async def test_loop_handles_tool_timeout_gracefully():
    class SlowSession:
        def __init__(self, delay: float):
            self.delay = delay
            self.calls: list[str] = []

        async def call_tool(self, name: str, arguments: dict[str, Any] | None = None):
            self.calls.append(name)
            await asyncio.sleep(self.delay)
            return CallToolResult(
                content=[TextContent(type="text", text="ok")],
                isError=False,
            )

    client = FakeAnthropic(
        [
            _response("tool_use", _tool_use("tu_1", "slow_tool")),
            _response("end_turn", _text("Tool timed out, answered with fallback.")),
        ]
    )
    slow_session = SlowSession(delay=0.2)

    answer = await agent.run_agent_loop(
        client,
        slow_session,
        TOOLS,
        "query",
        system_prompt="sys",
        model="m",
        tool_timeout=0.05,
    )

    assert answer == "Tool timed out, answered with fallback."
    assert slow_session.calls == ["slow_tool"]
    # Verify the tool result returned to Claude was flagged as error and contains timeout message
    tool_result = client.messages.requests[1]["messages"][2]["content"][0]
    assert tool_result["is_error"] is True
    assert "timed out after 0.05 seconds" in tool_result["content"]


@pytest.mark.asyncio
async def test_concurrent_tool_execution():
    client = FakeAnthropic(
        [
            _response(
                "tool_use",
                _tool_use("tu_1", "tool_a"),
                _tool_use("tu_2", "tool_b"),
            ),
            _response("end_turn", _text("Both tools executed.")),
        ]
    )
    session = FakeSession(
        {
            "tool_a": {"status": "a_done"},
            "tool_b": {"status": "b_done"},
        }
    )

    answer = await agent.run_agent_loop(
        client,
        session,
        TOOLS,
        "run both",
        system_prompt="sys",
        model="m",
    )

    assert answer == "Both tools executed."
    assert {c[0] for c in session.calls} == {"tool_a", "tool_b"}
    # Verify that the user message sent back contains both tool results
    user_results = client.messages.requests[1]["messages"][2]["content"]
    assert len(user_results) == 2
    tool_use_ids = {r["tool_use_id"] for r in user_results}
    assert tool_use_ids == {"tu_1", "tu_2"}


@pytest.mark.asyncio
async def test_concurrent_tool_execution_with_exception():
    class CrashingSession:
        async def call_tool(self, name: str, arguments: dict[str, Any] | None = None):
            if name == "crash_tool":
                raise RuntimeError("Hardware failure on tool")
            return CallToolResult(
                content=[TextContent(type="text", text="ok")],
                isError=False,
            )

    client = FakeAnthropic(
        [
            _response(
                "tool_use",
                _tool_use("tu_1", "good_tool"),
                _tool_use("tu_2", "crash_tool"),
            ),
            _response("end_turn", _text("Handled partial failure.")),
        ]
    )
    session = CrashingSession()

    answer = await agent.run_agent_loop(
        client,
        session,
        TOOLS,
        "run both",
        system_prompt="sys",
        model="m",
    )

    assert answer == "Handled partial failure."
    user_results = client.messages.requests[1]["messages"][2]["content"]
    assert len(user_results) == 2

    # Verify good tool succeeded
    assert user_results[0]["tool_use_id"] == "tu_1"
    assert user_results[0].get("is_error") is not True
    assert "ok" in user_results[0]["content"]

    # Verify crash_tool was safely captured via return_exceptions=True
    assert user_results[1]["tool_use_id"] == "tu_2"
    assert user_results[1]["is_error"] is True
    assert "Hardware failure on tool" in user_results[1]["content"]
