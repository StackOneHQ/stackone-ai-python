"""The feedback tool and session_id linking, against the MCP mock server.

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
from stackone_ai.types import SUBMIT_FEEDBACK_TOOL_NAME, ToolsetConfigError, ToolsetLoadError

# Must match MOCK_SEARCH_SESSION_ID in tests/mocks/mcp-server.ts.
MOCK_SESSION_ID = "mock-session-1"


def _reset_requests(base_url: str) -> None:
    httpx.delete(f"{base_url}/__requests").raise_for_status()


def _requests(base_url: str) -> list[dict[str, Any]]:
    response = httpx.get(f"{base_url}/__requests")
    response.raise_for_status()
    return response.json()


def _tool_calls(base_url: str, name: str) -> list[dict[str, Any]]:
    return [r for r in _requests(base_url) if r["path"] == "/mcp" and r.get("name") == name]


def _execute_calls(base_url: str) -> list[dict[str, Any]]:
    return [
        r
        for r in _requests(base_url)
        if r["path"] == "/mcp" and str(r.get("name", "")).endswith("_execute_action")
    ]


class TestFeedbackToolIsAnMcpTool:
    """Contract 1: built as an MCP tool in every mode."""

    @pytest.mark.parametrize("mode", [None, "individual", "search_execute"])
    def test_built_as_an_mcp_tool_in_every_mode(self, mcp_mock_server: str, mode):
        toolset = StackOneToolSet(api_key="test-key", base_url=mcp_mock_server, tool_mode=mode)
        tool = toolset.fetch_tools(account_ids=["acc1"]).get_tool(SUBMIT_FEEDBACK_TOOL_NAME)
        assert isinstance(tool, StackOneMcpTool)

    @pytest.mark.parametrize("mode", [None, "search_execute"])
    def test_calling_it_sends_one_tools_call(self, mcp_mock_server: str, mode):
        toolset = StackOneToolSet(api_key="test-key", base_url=mcp_mock_server, tool_mode=mode)
        tool = toolset.fetch_tools(account_ids=["acc1"]).get_tool(SUBMIT_FEEDBACK_TOOL_NAME)
        assert tool is not None

        _reset_requests(mcp_mock_server)
        result = tool.execute({"rating": "positive", "tool_names": ["acc1_tool_1"]})

        assert result["message"] == "Feedback recorded"
        assert len(_tool_calls(mcp_mock_server, SUBMIT_FEEDBACK_TOOL_NAME)) == 1


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

    def test_submit_feedback_raises_saying_not_enabled(self, mcp_mock_server_without_feedback: str):
        toolset = StackOneToolSet(api_key="test-key", base_url=mcp_mock_server_without_feedback)
        _reset_requests(mcp_mock_server_without_feedback)

        with pytest.raises(ToolsetLoadError, match="feedback is not enabled for this project"):
            toolset.submit_feedback("negative", ["mock_list_items"])
        assert _tool_calls(mcp_mock_server_without_feedback, SUBMIT_FEEDBACK_TOOL_NAME) == []

    def test_search_and_execute_are_unaffected(self, mcp_mock_server_without_feedback: str):
        toolset = StackOneToolSet(api_key="test-key", base_url=mcp_mock_server_without_feedback)
        hit = toolset.search("list items")[0]
        result = toolset.execute(hit["action_id"], session_id=hit["session_id"])
        assert result["result"]["data"] == {"nodes": []}


class TestSearchSessionId:
    """Contract 4: each hit carries the session_id of the search that produced it."""

    def test_every_hit_carries_the_search_session_id(self, mcp_mock_server: str):
        toolset = StackOneToolSet(api_key="test-key", base_url=mcp_mock_server)
        hits = toolset.search("list items")
        assert hits
        assert all(hit["session_id"] == MOCK_SESSION_ID for hit in hits)

    def test_each_hit_keeps_its_own_search_session_across_connectors(self, monkeypatch):
        per_connector = {
            "a_acc1_search_actions": {"session_id": "s-a", "actions": [{"action_id": "a_x"}]},
            "b_acc2_search_actions": {"session_id": "s-b", "actions": [{"action_id": "b_x"}]},
        }
        monkeypatch.setattr(StackOneMcpTool, "execute", lambda self, _args: per_connector[self.name])
        monkeypatch.setattr(
            "stackone_ai.toolset.fetch_mcp_tools",
            lambda _e, headers, **_k: [
                McpToolDefinition(name=name, description="", input_schema={})
                for name in per_connector
                if headers["x-account-id"] in name
            ],
        )
        toolset = StackOneToolSet(api_key="test-key")
        toolset.set_accounts(["acc1", "acc2"])
        assert {h["action_id"]: h["session_id"] for h in toolset.search("x")} == {"a_x": "s-a", "b_x": "s-b"}

    def test_the_field_is_omitted_when_the_server_returned_none(self, monkeypatch):
        monkeypatch.setattr(
            StackOneMcpTool, "execute", lambda self, _args: {"actions": [{"action_id": "a_x"}]}
        )
        monkeypatch.setattr(
            "stackone_ai.toolset.fetch_mcp_tools",
            lambda _e, _h, **_k: [
                McpToolDefinition(name="a_acc1_search_actions", description="", input_schema={})
            ],
        )
        toolset = StackOneToolSet(api_key="test-key", account_id="acc1")
        assert toolset.search("x") == [{"action_id": "a_x"}]


class TestExecuteSessionId:
    """Contract 5: forwarded top-level when given, never when not, action_id still last."""

    def test_forwarded_when_given(self, mcp_mock_server: str):
        toolset = StackOneToolSet(api_key="test-key", base_url=mcp_mock_server)
        hit = toolset.search("list items")[0]

        _reset_requests(mcp_mock_server)
        toolset.execute(hit["action_id"], hit["example_request"], session_id=hit["session_id"])

        [call] = _execute_calls(mcp_mock_server)
        assert call["arguments"] == {
            "query": {"page_size": 25},
            "session_id": MOCK_SESSION_ID,
            "action_id": "mock_list_items",
        }

    def test_not_forwarded_when_not_given(self, mcp_mock_server: str):
        toolset = StackOneToolSet(api_key="test-key", base_url=mcp_mock_server)
        _reset_requests(mcp_mock_server)
        toolset.execute("mock_list_items")

        [call] = _execute_calls(mcp_mock_server)
        assert call["arguments"] == {"action_id": "mock_list_items"}

    def test_pinning_and_header_sanitising_survive_alongside_it(self, mcp_mock_server: str):
        toolset = StackOneToolSet(api_key="test-key", base_url=mcp_mock_server)
        _reset_requests(mcp_mock_server)
        toolset.execute(
            "mock_list_items",
            {
                "action_id": "mock_delete_everything",
                "headers": {"x-account-id": "victim"},
                "session_id": "model-invented",
            },
            session_id=MOCK_SESSION_ID,
        )

        [call] = _execute_calls(mcp_mock_server)
        assert call["arguments"]["action_id"] == "mock_list_items"
        assert call["arguments"]["headers"] == {}
        assert call["arguments"]["session_id"] == MOCK_SESSION_ID

    @pytest.mark.parametrize("bad", ["", 42])
    def test_rejects_a_non_string_or_empty_session_id(self, bad):
        toolset = StackOneToolSet(api_key="test-key", account_id="acc1")
        with pytest.raises(ToolsetConfigError, match="session_id"):
            toolset.execute("mock_list_items", session_id=bad)


class TestSubmitFeedback:
    """Contract 6: the toolset method."""

    @pytest.mark.parametrize("mode", [None, "search_execute"])
    def test_sends_only_the_given_keys_over_tools_call(self, mcp_mock_server: str, mode):
        toolset = StackOneToolSet(api_key="test-key", base_url=mcp_mock_server, tool_mode=mode)
        _reset_requests(mcp_mock_server)

        result = toolset.submit_feedback("negative", ["mock_list_items"], session_id=MOCK_SESSION_ID)

        [call] = _tool_calls(mcp_mock_server, SUBMIT_FEEDBACK_TOOL_NAME)
        # Exact equality: no feedback/category keys at all, rather than keys set to null.
        assert call["arguments"] == {
            "rating": "negative",
            "tool_names": ["mock_list_items"],
            "session_id": MOCK_SESSION_ID,
            "source": "model",
        }
        assert all(value is not None for value in call["arguments"].values())
        assert result["session_id"] == MOCK_SESSION_ID

    def test_sends_every_field_in_contract_order(self, mcp_mock_server: str):
        toolset = StackOneToolSet(api_key="test-key", base_url=mcp_mock_server)
        _reset_requests(mcp_mock_server)

        toolset.submit_feedback(
            "positive",
            ["mock_list_items"],
            feedback="Found it first try",
            category="search",
            session_id=MOCK_SESSION_ID,
            source="user",
        )

        [call] = _tool_calls(mcp_mock_server, SUBMIT_FEEDBACK_TOOL_NAME)
        assert list(call["arguments"].items()) == [
            ("rating", "positive"),
            ("tool_names", ["mock_list_items"]),
            ("feedback", "Found it first try"),
            ("category", "search"),
            ("session_id", MOCK_SESSION_ID),
            ("source", "user"),
        ]

    def test_session_id_is_omitted_when_not_given(self, mcp_mock_server: str):
        toolset = StackOneToolSet(api_key="test-key", base_url=mcp_mock_server)
        _reset_requests(mcp_mock_server)
        toolset.submit_feedback("neutral", ["mock_list_items"])

        [call] = _tool_calls(mcp_mock_server, SUBMIT_FEEDBACK_TOOL_NAME)
        assert "session_id" not in call["arguments"]

    def test_runs_once_across_many_accounts(self, mcp_mock_server: str):
        toolset = StackOneToolSet(api_key="test-key", base_url=mcp_mock_server)
        _reset_requests(mcp_mock_server)
        toolset.submit_feedback("positive", ["acc1_tool_1"], account_ids=["acc1", "acc2"])

        calls = _tool_calls(mcp_mock_server, SUBMIT_FEEDBACK_TOOL_NAME)
        assert [c["accountId"] for c in calls] == ["acc1"]

    def test_rejects_a_bare_string_of_tool_names(self):
        toolset = StackOneToolSet(api_key="test-key", account_id="acc1")
        with pytest.raises(ToolsetConfigError, match="Did you mean"):
            toolset.submit_feedback("positive", "mock_list_items")  # ty: ignore[invalid-argument-type]
