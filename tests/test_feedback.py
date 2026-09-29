"""The feedback tool, against the MCP mock server.

Assertions about the wire read the mock's request log rather than a handler's view of
the call: the handler only sees arguments after zod has parsed them, which strips
unknown keys and hides a key sent as null.
"""

from __future__ import annotations

import logging
from typing import Any

import httpx
import pytest

from stackone_ai.tools import McpToolDefinition, StackOneMcpTool
from stackone_ai.toolset import StackOneToolSet
from stackone_ai.types import SUBMIT_FEEDBACK_TOOL_NAME


def _reset_requests(base_url: str) -> None:
    httpx.delete(f"{base_url}/__requests").raise_for_status()


def _requests(base_url: str) -> list[dict[str, Any]]:
    response = httpx.get(f"{base_url}/__requests")
    response.raise_for_status()
    return response.json()


def _tool_calls(base_url: str, name: str) -> list[dict[str, Any]]:
    return [r for r in _requests(base_url) if r["path"] == "/mcp" and r.get("name") == name]


def _rpc_requests(base_url: str) -> list[dict[str, Any]]:
    return [r for r in _requests(base_url) if r["path"] == "/actions/rpc"]


class TestFeedbackToolIsAnMcpTool:
    """Contract 1: built as an MCP tool in every mode, never as an RPC tool."""

    @pytest.mark.parametrize("mode", [None, "individual", "search_execute"])
    def test_built_as_an_mcp_tool_in_every_mode(self, mcp_mock_server: str, mode):
        toolset = StackOneToolSet(api_key="test-key", base_url=mcp_mock_server, tool_mode=mode)
        tool = toolset.fetch_tools(account_ids=["acc1"]).get_tool(SUBMIT_FEEDBACK_TOOL_NAME)
        assert isinstance(tool, StackOneMcpTool)

    @pytest.mark.parametrize("mode", [None, "search_execute"])
    def test_calling_it_sends_tools_call_and_no_rpc_request(self, mcp_mock_server: str, mode):
        """In individual mode it used to be built as an RPC tool and sent to /actions/rpc."""
        toolset = StackOneToolSet(api_key="test-key", base_url=mcp_mock_server, tool_mode=mode)
        tool = toolset.fetch_tools(account_ids=["acc1"]).get_tool(SUBMIT_FEEDBACK_TOOL_NAME)
        assert tool is not None

        _reset_requests(mcp_mock_server)
        result = tool.execute({"rating": "positive", "tool_names": ["acc1_tool_1"]})

        assert result["message"] == "Feedback recorded"
        assert len(_tool_calls(mcp_mock_server, SUBMIT_FEEDBACK_TOOL_NAME)) == 1
        assert _rpc_requests(mcp_mock_server) == []


class TestFeedbackToolIsListedOnce:
    """Contract 2: one tool however many accounts list it, and no duplicate warning."""

    @pytest.mark.parametrize("mode", [None, "search_execute"])
    def test_multi_account_listing_yields_exactly_one(self, mcp_mock_server: str, mode, caplog):
        toolset = StackOneToolSet(api_key="test-key", base_url=mcp_mock_server, tool_mode=mode)
        with caplog.at_level(logging.WARNING, logger="stackone.tools"):
            tools = toolset.fetch_tools(account_ids=["acc1", "acc2", "acc3"])

        feedback = [t for t in tools if t.name == SUBMIT_FEEDBACK_TOOL_NAME]
        assert len(feedback) == 1
        assert "more than one account" not in caplog.text

    def test_the_first_listing_is_kept(self, monkeypatch):
        monkeypatch.setattr(
            "stackone_ai.toolset.fetch_mcp_tools",
            lambda _e, _h, **_k: [
                McpToolDefinition(name=SUBMIT_FEEDBACK_TOOL_NAME, description="", input_schema={})
            ],
        )
        toolset = StackOneToolSet(api_key="test-key")
        tool = toolset.fetch_tools(account_ids=["acc2", "acc1"]).get_tool(SUBMIT_FEEDBACK_TOOL_NAME)
        assert tool is not None
        # Listings are merged in sorted account order, so "first" is stable across calls.
        assert tool.get_account_id() == "acc1"

    def test_real_duplicates_still_warn(self, monkeypatch, caplog):
        """Only the global tool is deduped; a genuine clash must still be reported."""
        monkeypatch.setattr(
            "stackone_ai.toolset.fetch_mcp_tools",
            lambda _e, _h, **_k: [
                McpToolDefinition(name="linear_list_issues", description="", input_schema={})
            ],
        )
        toolset = StackOneToolSet(api_key="test-key")
        with caplog.at_level(logging.WARNING, logger="stackone.tools"):
            tools = toolset.fetch_tools(account_ids=["acc1", "acc2"])
        assert len(tools) == 2
        assert "more than one account" in caplog.text


class TestFeedbackToolAbsent:
    """Contract 3: never invented client-side."""

    @pytest.mark.parametrize("mode", [None, "search_execute"])
    def test_absent_when_the_server_does_not_serve_it(self, mcp_mock_server_without_feedback: str, mode):
        base_url = mcp_mock_server_without_feedback
        toolset = StackOneToolSet(api_key="test-key", base_url=base_url, tool_mode=mode)
        tools = toolset.fetch_tools(account_ids=["acc1", "acc2"])
        assert len(tools) > 0
        assert tools.get_tool(SUBMIT_FEEDBACK_TOOL_NAME) is None
