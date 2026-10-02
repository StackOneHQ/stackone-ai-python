"""Tests for StackOneToolSet MCP functionality using real MCP mock server."""

from __future__ import annotations

import asyncio
import threading
import time
from collections.abc import Iterator
from contextlib import asynccontextmanager, contextmanager
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from stackone_ai.tools import McpToolDefinition, StackOneMcpTool, fetch_mcp_tools
from stackone_ai.toolset import StackOneToolSet
from stackone_ai.types import (
    SUBMIT_FEEDBACK_TOOL_NAME,
    StackOneAPIError,
    ToolParameters,
    ToolsetConfigError,
    ToolsetError,
    ToolsetLoadError,
)

# Must match MOCK_DOWNLOAD_LINK in tests/mocks/mcp-server.ts.
MOCK_DOWNLOAD_LINK = {
    "download_url": "https://downloads.example.com/f/abc123?sig=xyz",
    "expires_at": "2026-01-01T00:00:00.000Z",
    "file": {"name": "report.pdf", "content_type": "application/pdf", "content_length": 1024},
}


def _reset_requests(base_url: str) -> None:
    httpx.delete(f"{base_url}/__requests").raise_for_status()


def _tool_calls(base_url: str, name: str) -> list[dict[str, Any]]:
    """The tools/call requests the mock received for ``name``, as they were on the wire."""
    response = httpx.get(f"{base_url}/__requests")
    response.raise_for_status()
    return [r for r in response.json() if r["method"] == "tools/call" and r["name"] == name]


class TestAccountFiltering:
    """Test account filtering functionality with real MCP server."""

    def test_set_accounts_chaining(self, mcp_mock_server: str):
        """Test that setAccounts() returns self for chaining"""
        toolset = StackOneToolSet(api_key="test-key", base_url=mcp_mock_server)
        result = toolset.set_accounts(["acc1", "acc2"])
        assert result is toolset

    def test_fetch_tools_without_account_filtering(self, mcp_mock_server: str):
        """Test fetching tools without account filtering"""
        toolset = StackOneToolSet(api_key="test-key", base_url=mcp_mock_server)
        tools = toolset.fetch_tools()
        # Plus the mock's global feedback tool, served alongside every account's catalog.
        assert len(tools) == 3
        tool_names = [t.name for t in tools.to_list()]
        assert "default_tool_1" in tool_names
        assert "default_tool_2" in tool_names

    def test_fetch_tools_with_account_ids(self, mcp_mock_server: str):
        """Test fetching tools with specific account IDs"""
        toolset = StackOneToolSet(api_key="test-key", base_url=mcp_mock_server)
        tools = toolset.fetch_tools(account_ids=["acc1"])
        assert len(tools) == 3
        tool_names = [t.name for t in tools.to_list()]
        assert "acc1_tool_1" in tool_names
        assert "acc1_tool_2" in tool_names

    def test_fetch_tools_uses_set_accounts(self, mcp_mock_server: str):
        """Test that fetch_tools uses set_accounts when no accountIds provided"""
        toolset = StackOneToolSet(api_key="test-key", base_url=mcp_mock_server)
        toolset.set_accounts(["acc1", "acc2"])
        tools = toolset.fetch_tools()
        # acc1 has 2 tools, acc2 has 2 tools, plus one feedback tool however many accounts list it
        assert len(tools) == 5
        tool_names = [t.name for t in tools.to_list()]
        assert "acc1_tool_1" in tool_names
        assert "acc1_tool_2" in tool_names
        assert "acc2_tool_1" in tool_names
        assert "acc2_tool_2" in tool_names

    def test_fetch_tools_overrides_set_accounts(self, mcp_mock_server: str):
        """Test that accountIds parameter overrides set_accounts"""
        toolset = StackOneToolSet(api_key="test-key", base_url=mcp_mock_server)
        toolset.set_accounts(["acc1", "acc2"])
        tools = toolset.fetch_tools(account_ids=["acc3"])
        # Should fetch tools only for acc3 (ignoring acc1, acc2), plus the feedback tool
        assert len(tools) == 2
        tool_names = [t.name for t in tools.to_list()]
        assert "acc3_tool_1" in tool_names
        # Verify set_accounts state is preserved
        assert toolset._account_ids == ["acc1", "acc2"]

    def test_fetch_tools_multiple_account_ids(self, mcp_mock_server: str):
        """Test fetching tools for multiple account IDs"""
        toolset = StackOneToolSet(api_key="test-key", base_url=mcp_mock_server)
        tools = toolset.fetch_tools(account_ids=["acc1", "acc2", "acc3"])
        # acc1: 2 tools, acc2: 2 tools, acc3: 1 tool, feedback: 1 = 6 total
        assert len(tools) == 6

    def test_fetch_tools_preserves_account_context(self, mcp_mock_server: str):
        """Test that tools preserve their account context"""
        toolset = StackOneToolSet(api_key="test-key", base_url=mcp_mock_server)
        tools = toolset.fetch_tools(account_ids=["acc1"])

        tool = tools.get_tool("acc1_tool_1")
        assert tool is not None
        assert tool.get_account_id() == "acc1"


class TestProviderAndActionFiltering:
    """Test provider and action filtering functionality with real MCP server."""

    def test_filter_by_providers(self, mcp_mock_server: str):
        """Test filtering tools by providers"""
        toolset = StackOneToolSet(api_key="test-key", base_url=mcp_mock_server)
        tools = toolset.fetch_tools(account_ids=["mixed"], providers=["hibob", "bamboohr"])
        assert len(tools) == 4
        tool_names = [t.name for t in tools.to_list()]
        assert "hibob_list_employees" in tool_names
        assert "hibob_create_employees" in tool_names
        assert "bamboohr_list_employees" in tool_names
        assert "bamboohr_get_employee" in tool_names
        assert "workday_list_employees" not in tool_names

    def test_filter_by_actions_exact_match(self, mcp_mock_server: str):
        """Test filtering tools by exact action names"""
        toolset = StackOneToolSet(api_key="test-key", base_url=mcp_mock_server)
        tools = toolset.fetch_tools(
            account_ids=["mixed"], actions=["hibob_list_employees", "hibob_create_employees"]
        )
        assert len(tools) == 2
        tool_names = [t.name for t in tools.to_list()]
        assert "hibob_list_employees" in tool_names
        assert "hibob_create_employees" in tool_names

    def test_filter_by_actions_glob_pattern(self, mcp_mock_server: str):
        """Test filtering tools by glob patterns"""
        toolset = StackOneToolSet(api_key="test-key", base_url=mcp_mock_server)
        tools = toolset.fetch_tools(account_ids=["mixed"], actions=["*_list_employees"])
        assert len(tools) == 3
        tool_names = [t.name for t in tools.to_list()]
        assert "hibob_list_employees" in tool_names
        assert "bamboohr_list_employees" in tool_names
        assert "workday_list_employees" in tool_names
        assert "hibob_create_employees" not in tool_names
        assert "bamboohr_get_employee" not in tool_names

    def test_combine_account_and_action_filters(self, mcp_mock_server: str):
        """Test combining account and action filters"""
        toolset = StackOneToolSet(api_key="test-key", base_url=mcp_mock_server)
        # acc1 has acc1_tool_1, acc1_tool_2
        # acc2 has acc2_tool_1, acc2_tool_2
        tools = toolset.fetch_tools(account_ids=["acc1", "acc2"], actions=["*_tool_1"])
        assert len(tools) == 2
        tool_names = [t.name for t in tools.to_list()]
        assert "acc1_tool_1" in tool_names
        assert "acc2_tool_1" in tool_names
        assert "acc1_tool_2" not in tool_names
        assert "acc2_tool_2" not in tool_names

    def test_combine_provider_and_action_filters(self, mcp_mock_server: str):
        """Test combining providers and actions filters"""
        toolset = StackOneToolSet(api_key="test-key", base_url=mcp_mock_server)
        tools = toolset.fetch_tools(account_ids=["mixed"], providers=["hibob"], actions=["*_list_*"])
        # Should only return hibob_list_employees (matches both filters)
        assert len(tools) == 1
        tool_names = [t.name for t in tools.to_list()]
        assert "hibob_list_employees" in tool_names


class TestMcpHeaders:
    """Test that MCP headers are built correctly."""

    def test_authorization_header_is_set(self, mcp_mock_server: str):
        """Test that authorization header is properly set (server validates basic auth)"""
        toolset = StackOneToolSet(api_key="test-key", base_url=mcp_mock_server)
        # If auth fails, this would raise an error
        tools = toolset.fetch_tools()
        assert len(tools) > 0

    def test_account_id_header_is_sent(self, mcp_mock_server: str):
        """Test that x-account-id header is sent when account_id is provided"""
        toolset = StackOneToolSet(api_key="test-key", base_url=mcp_mock_server)
        # When we fetch with acc1, we should get acc1's tools, proving header was sent
        tools = toolset.fetch_tools(account_ids=["acc1"])
        tool_names = [t.name for t in tools.to_list() if t.name != SUBMIT_FEEDBACK_TOOL_NAME]
        assert tool_names
        assert all("acc1" in name for name in tool_names)


class TestToolCreation:
    """Test that tools are created correctly from MCP responses."""

    def test_tool_has_name_and_description(self, mcp_mock_server: str):
        """Test that tools have proper name and description"""
        toolset = StackOneToolSet(api_key="test-key", base_url=mcp_mock_server)
        tools = toolset.fetch_tools()
        tool = tools.get_tool("default_tool_1")
        assert tool is not None
        assert tool.name == "default_tool_1"
        assert tool.description == "Default Tool 1"

    def test_tool_has_parameters_type(self, mcp_mock_server: str):
        """Test that tools have proper parameters type from input schema"""
        toolset = StackOneToolSet(api_key="test-key", base_url=mcp_mock_server)
        tools = toolset.fetch_tools()
        tool = tools.get_tool("default_tool_1")
        assert tool is not None
        assert tool.parameters is not None
        assert tool.parameters.type == "object"


class TestRootType:
    """The root type the model sees: a served string as is, anything else "object", as in Node."""

    @pytest.mark.parametrize(
        ("served", "expected"),
        [
            ({"type": "object"}, "object"),
            ({}, "object"),
            ({"type": ["object", "null"]}, "object"),
            ({"type": None}, "object"),
            ({"type": 7}, "object"),
        ],
    )
    def test_root_type(self, monkeypatch, served: dict[str, Any], expected: str):
        monkeypatch.setattr(
            "stackone_ai.toolset.fetch_mcp_tools",
            lambda *_a, **_k: [McpToolDefinition(name="t", description="", input_schema=served)],
        )
        tool = StackOneToolSet(api_key="test-key", account_id="acc1").fetch_tools()[0]
        assert tool.parameters.type == expected
        assert tool.to_openai_function()["function"]["parameters"]["type"] == expected


class TestSchemaPropertyNormalization:
    """Test schema property normalization with monkeypatch (for precise schema control)."""

    def test_tool_properties_are_normalized(self, monkeypatch):
        """Test that tool properties are correctly extracted from input schema"""
        from stackone_ai.tools import McpToolDefinition

        def fake_fetch(_: str, headers: dict[str, str], **_kwargs: object) -> list[McpToolDefinition]:
            return [
                McpToolDefinition(
                    name="test_tool",
                    description="Test tool",
                    input_schema={
                        "type": "object",
                        "properties": {
                            "name": {"type": "string", "description": "The name"},
                            "age": {"type": "integer"},
                        },
                        "required": ["name"],
                    },
                )
            ]

        monkeypatch.setattr("stackone_ai.toolset.fetch_mcp_tools", fake_fetch)

        toolset = StackOneToolSet(api_key="test-key", account_id="acc1")
        tools = toolset.fetch_tools()
        tool = tools.get_tool("test_tool")
        assert tool is not None
        assert "name" in tool.parameters.properties
        assert "age" in tool.parameters.properties

    def test_required_fields_marked_not_nullable(self, monkeypatch):
        """Test that required fields are marked as not nullable"""
        from stackone_ai.tools import McpToolDefinition

        def fake_fetch(_: str, headers: dict[str, str], **_kwargs: object) -> list[McpToolDefinition]:
            return [
                McpToolDefinition(
                    name="test_tool",
                    description="Test tool",
                    input_schema={
                        "type": "object",
                        "properties": {"id": {"type": "string"}},
                        "required": ["id"],
                    },
                )
            ]

        monkeypatch.setattr("stackone_ai.toolset.fetch_mcp_tools", fake_fetch)

        toolset = StackOneToolSet(api_key="test-key", account_id="acc1")
        tools = toolset.fetch_tools()
        tool = tools.get_tool("test_tool")
        assert tool is not None
        assert tool.parameters.properties["id"].get("nullable") is False

    def test_optional_fields_marked_nullable(self, monkeypatch):
        """Test that optional fields are marked as nullable"""
        from stackone_ai.tools import McpToolDefinition

        def fake_fetch(_: str, headers: dict[str, str], **_kwargs: object) -> list[McpToolDefinition]:
            return [
                McpToolDefinition(
                    name="test_tool",
                    description="Test tool",
                    input_schema={
                        "type": "object",
                        "properties": {"optional_field": {"type": "string"}},
                    },
                )
            ]

        monkeypatch.setattr("stackone_ai.toolset.fetch_mcp_tools", fake_fetch)

        toolset = StackOneToolSet(api_key="test-key", account_id="acc1")
        tools = toolset.fetch_tools()
        tool = tools.get_tool("test_tool")
        assert tool is not None
        assert tool.parameters.properties["optional_field"].get("nullable") is True


# Served `required` -> the `required` every adapter emits (absent means the key is omitted).
# Mirrors the Node SDK's toJsonSchema() and toolParametersFromInputSchema().
_ABSENT = object()
_SERVED_REQUIRED_CASES = [
    pytest.param(["b", "a"], ["b", "a"], id="served-order-not-property-order"),
    pytest.param(["a", "b"], ["a", "b"], id="already-in-property-order"),
    pytest.param(["c", "a", "b"], ["c", "a", "b"], id="three-reversed"),
    pytest.param(["ghost"], ["ghost"], id="undeclared-name-kept"),
    pytest.param(["a", "a"], ["a", "a"], id="duplicates-kept"),
    pytest.param(["b", 1, "a"], ["b", "a"], id="non-string-entry-dropped"),
    pytest.param([], None, id="empty"),
    pytest.param(_ABSENT, None, id="absent"),
    pytest.param(None, None, id="null"),
    pytest.param("a string", None, id="string"),
    pytest.param([1, 2], None, id="only-non-strings"),
]


class TestServedRequiredOrder:
    """`required` is the served list, verbatim and in the served order, on every adapter.

    It was rebuilt from the per-property `nullable` markers, which walked the properties
    and so re-sorted it into property order: a model was shown a list the server never
    sent, and the Node SDK's schema differed from this one for the same tool.
    """

    @staticmethod
    def _tool(monkeypatch, served_required: object):
        schema: dict[str, object] = {
            "type": "object",
            "properties": {"a": {"type": "string"}, "b": {"type": "string"}, "c": {"type": "string"}},
        }
        if served_required is not _ABSENT:
            schema["required"] = served_required
        monkeypatch.setattr(
            "stackone_ai.toolset.fetch_mcp_tools",
            lambda _e, _h, **_k: [McpToolDefinition(name="t", description="", input_schema=schema)],
        )
        tool = StackOneToolSet(api_key="k", account_id="acc1").fetch_tools().get_tool("t")
        assert tool is not None
        return tool

    @pytest.mark.parametrize(("served", "expected"), _SERVED_REQUIRED_CASES)
    def test_openai(self, monkeypatch, served, expected):
        parameters = self._tool(monkeypatch, served).to_openai_function()["function"]["parameters"]
        assert parameters.get("required") == expected
        if expected is None:
            assert "required" not in parameters

    @pytest.mark.parametrize(("served", "expected"), _SERVED_REQUIRED_CASES)
    def test_langchain(self, monkeypatch, served, expected):
        schema = self._tool(monkeypatch, served).to_langchain().args_schema
        assert schema.get("required") == expected
        if expected is None:
            assert "required" not in schema

    @pytest.mark.parametrize(("served", "expected"), _SERVED_REQUIRED_CASES)
    def test_pydantic_ai(self, monkeypatch, served, expected):
        pytest.importorskip("pydantic_ai")
        schema = self._tool(monkeypatch, served).to_pydantic_ai_tool().function_schema.json_schema
        assert schema.get("required") == expected
        if expected is None:
            assert "required" not in schema

    def test_parameters_still_carry_the_served_list_and_markers(self, monkeypatch):
        """The ADK plugin reads `tool.parameters` directly, so it must not change shape."""
        tool = self._tool(monkeypatch, ["b", "a"])
        dumped = tool.parameters.model_dump()
        assert dumped["required"] == ["b", "a"]
        assert {name: prop["nullable"] for name, prop in dumped["properties"].items()} == {
            "a": False,
            "b": False,
            "c": True,
        }

    def test_non_string_entry_does_not_mark_a_property_required(self, monkeypatch):
        """The marker agrees with the emitted list: `1` is not the property named "1"."""
        schema = {"type": "object", "properties": {"1": {"type": "string"}}, "required": [1]}
        monkeypatch.setattr(
            "stackone_ai.toolset.fetch_mcp_tools",
            lambda _e, _h, **_k: [McpToolDefinition(name="t", description="", input_schema=schema)],
        )
        tool = StackOneToolSet(api_key="k", account_id="acc1").fetch_tools().get_tool("t")
        assert tool is not None
        assert tool.parameters.properties["1"]["nullable"] is True


class TestPerActionToolsExecuteOverToolsCall:
    """Every per-action tool runs over MCP tools/call on the endpoint and account that listed it.

    Assertions about the wire read the mock's request log, not a handler's view of the call.
    """

    def test_execute_returns_the_result_as_the_server_wrote_it(self, mcp_mock_server: str):
        toolset = StackOneToolSet(api_key="test-key", base_url=mcp_mock_server)
        tool = toolset.fetch_tools(account_ids=["your-bamboohr-account-id"]).get_tool(
            "bamboohr_list_employees"
        )
        assert tool is not None

        assert tool.execute() == {
            "isError": False,
            "result": {"data": [{"id": "1", "name": "Employee 1"}, {"id": "2", "name": "Employee 2"}]},
        }

    def test_one_tools_call_on_the_listing_endpoint_and_no_param_style(self, mcp_mock_server: str):
        toolset = StackOneToolSet(api_key="test-key", base_url=mcp_mock_server)
        tool = toolset.fetch_tools(account_ids=["your-bamboohr-account-id"]).get_tool("bamboohr_get_employee")
        assert tool is not None

        _reset_requests(mcp_mock_server)
        result = tool.execute({"id": "emp-123"})

        assert result == {"isError": False, "result": {"data": {"id": "emp-123", "name": "Test Employee"}}}
        [call] = _tool_calls(mcp_mock_server, "bamboohr_get_employee")
        assert call["path"] == "/mcp"
        assert call["search"] == ""

    @pytest.mark.parametrize(
        "arguments",
        [
            pytest.param({"fields": "a,b"}, id="flat"),
            pytest.param({"path_id": "1", "query_limit": 5, "body_name": "x"}, id="prefix-lookalikes"),
            pytest.param({"path": {"id": "1"}, "query": {"limit": 5}, "body": {"n": [1, None]}}, id="nested"),
            pytest.param({"query": "not-an-object", "header_x": "y", "unicode": "é😀"}, id="odd-shapes"),
            pytest.param({}, id="empty"),
        ],
    )
    def test_arguments_are_sent_verbatim(self, mcp_mock_server: str, arguments: dict[str, Any]):
        """No envelope splitting and no prefix routing: the server maps them itself."""
        toolset = StackOneToolSet(api_key="test-key", base_url=mcp_mock_server)
        tool = toolset.fetch_tools(account_ids=["mixed"]).get_tool("hibob_list_employees")
        assert tool is not None

        _reset_requests(mcp_mock_server)
        result = tool.execute(arguments)

        [call] = _tool_calls(mcp_mock_server, "hibob_list_employees")
        assert call["arguments"] == arguments
        assert result == {
            "isError": False,
            "result": {"data": {"action": "hibob_list_employees", "received": arguments}},
        }

    def test_a_call_does_not_relist_the_catalog(self, mcp_mock_server: str):
        # The mcp client's call_tool lists tools first, to validate structuredContent against
        # an output schema, and every call opens a fresh session — so each call relisted the
        # whole catalog. The Node SDK sends only the tools/call.
        toolset = StackOneToolSet(api_key="test-key", base_url=mcp_mock_server)
        tool = toolset.fetch_tools(account_ids=["acc1"]).get_tool("acc1_tool_1")
        assert tool is not None
        _reset_requests(mcp_mock_server)

        tool.execute({"param": "x"})

        response = httpx.get(f"{mcp_mock_server}/__requests")
        assert [r["method"] for r in response.json()] == ["tools/call"]

    def test_x_account_id_is_the_listing_accounts(self, mcp_mock_server: str):
        toolset = StackOneToolSet(api_key="test-key", base_url=mcp_mock_server)
        tools = toolset.fetch_tools(account_ids=["acc1", "acc2"])

        _reset_requests(mcp_mock_server)
        for name in ("acc1_tool_1", "acc2_tool_1"):
            tool = tools.get_tool(name)
            assert tool is not None
            tool.execute({"fields": "id"})

        assert [c["accountId"] for c in _tool_calls(mcp_mock_server, "acc1_tool_1")] == ["acc1"]
        assert [c["accountId"] for c in _tool_calls(mcp_mock_server, "acc2_tool_1")] == ["acc2"]

    def test_a_download_link_is_returned_unchanged(self, mcp_mock_server: str):
        """A file action over tools/call returns a link; the SDK neither follows nor reshapes it."""
        toolset = StackOneToolSet(api_key="test-key", base_url=mcp_mock_server)
        tool = toolset.fetch_tools(account_ids=["files"]).get_tool("files_download_file")
        assert tool is not None

        assert tool.execute({"id": "f1"}) == {"isError": False, "result": MOCK_DOWNLOAD_LINK}

    def test_a_download_with_no_link_raises_with_status_501(self, mcp_mock_server: str):
        toolset = StackOneToolSet(api_key="test-key", base_url=mcp_mock_server)
        tool = toolset.fetch_tools(account_ids=["files"]).get_tool("files_download_unavailable")
        assert tool is not None

        with pytest.raises(StackOneAPIError) as excinfo:
            tool.execute({"id": "f1"})
        assert excinfo.value.status_code == 501
        assert excinfo.value.response_body["isError"] is True

    def test_defender_metadata_is_kept_beside_the_result(self, mcp_mock_server: str):
        toolset = StackOneToolSet(api_key="test-key", base_url=mcp_mock_server)
        tool = toolset.fetch_tools(account_ids=["files"]).get_tool("files_list_defended")
        assert tool is not None

        assert tool.execute() == {
            "isError": False,
            "result": {"data": []},
            "defenderMetadata": {"scanned": True, "flagged": 0},
        }

    def test_a_non_object_result_is_returned_as_served(self, mcp_mock_server: str):
        toolset = StackOneToolSet(api_key="test-key", base_url=mcp_mock_server)
        tool = toolset.fetch_tools(account_ids=["files"]).get_tool("files_count")
        assert tool is not None

        assert tool.execute() == {"isError": False, "result": 3}

    def test_an_action_named_like_a_meta_tool_runs_as_an_action(self, mcp_mock_server: str):
        """Outside search_execute mode, a name ending in `_execute_action` is just an action."""
        toolset = StackOneToolSet(api_key="test-key", base_url=mcp_mock_server)
        tool = toolset.fetch_tools(account_ids=["lookalike"]).get_tool("lookalike_execute_action")
        assert tool is not None

        assert tool.execute({"action_id": "other"}) == {
            "isError": False,
            "result": {"data": {"action": "lookalike_execute_action", "received": {"action_id": "other"}}},
        }


class TestAccountIdFallback:
    """Test account ID fallback to instance account_id."""

    def test_uses_instance_account_id_when_no_other_provided(self, monkeypatch):
        """Test that fetch_tools uses instance account_id when no account_ids provided."""
        sample_tool = McpToolDefinition(
            name="test_tool",
            description="Test tool",
            input_schema={"type": "object", "properties": {}},
        )

        captured_accounts: list[str | None] = []

        def fake_fetch(_: str, headers: dict[str, str], **_kwargs: object) -> list[McpToolDefinition]:
            captured_accounts.append(headers.get("x-account-id"))
            return [sample_tool]

        monkeypatch.setattr("stackone_ai.toolset.fetch_mcp_tools", fake_fetch)

        # Create toolset with account_id in constructor
        toolset = StackOneToolSet(api_key="test_key", account_id="instance_account")
        tools = toolset.fetch_tools()  # No account_ids, no set_accounts

        # Should use the instance account_id
        assert captured_accounts == ["instance_account"]
        assert len(tools) == 1
        tool = tools.get_tool("test_tool")
        assert tool is not None
        assert tool.get_account_id() == "instance_account"


class TestToolsetErrorHandling:
    """Test error handling in fetch_tools."""

    def test_reraises_toolset_error(self, monkeypatch):
        """Test that ToolsetError is re-raised without wrapping."""
        from stackone_ai.types import ToolsetConfigError

        def fake_fetch(_: str, headers: dict[str, str], **_kwargs: object) -> list[McpToolDefinition]:
            raise ToolsetConfigError("Original config error")

        monkeypatch.setattr("stackone_ai.toolset.fetch_mcp_tools", fake_fetch)

        toolset = StackOneToolSet(api_key="test_key", account_id="acc1")
        with pytest.raises(ToolsetConfigError, match="Original config error"):
            toolset.fetch_tools()


class TestFetchMcpToolsInternal:
    """Test fetch_mcp_tools internal implementation."""

    def test_fetch_mcp_tools_single_page(self):
        """Test fetching tools with single page response."""
        # Create mock tool response
        mock_tool = MagicMock()
        mock_tool.name = "test_tool"
        mock_tool.description = "Test description"
        mock_tool.inputSchema = {"type": "object", "properties": {"id": {"type": "string"}}}

        mock_result = MagicMock()
        mock_result.tools = [mock_tool]
        mock_result.nextCursor = None

        # Create mock session
        mock_session = AsyncMock()
        mock_session.initialize = AsyncMock()
        mock_session.list_tools = AsyncMock(return_value=mock_result)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        # Create mock streamable client
        seen_clients: list[httpx.AsyncClient] = []

        @asynccontextmanager
        async def mock_streamable_client(endpoint, *, http_client, **_kwargs: object):
            seen_clients.append(http_client)
            yield (MagicMock(), MagicMock(), MagicMock())

        # Patch at the module where imports happen
        with (
            patch(
                "mcp.client.streamable_http.streamable_http_client",
                side_effect=mock_streamable_client,
            ),
            patch("mcp.client.session.ClientSession", return_value=mock_session),
            patch("mcp.types.Implementation", MagicMock()),
        ):
            result = fetch_mcp_tools(
                "https://api.example.com/mcp", {"Authorization": "Basic test"}, timeout=7
            )

            # The caller's headers and timeout reach the HTTP client, on every leg.
            [client] = seen_clients
            assert client.headers["Authorization"] == "Basic test"
            assert client.timeout == httpx.Timeout(7)
            assert client.follow_redirects is True

            assert len(result) == 1
            assert result[0].name == "test_tool"
            assert result[0].description == "Test description"
            assert result[0].input_schema == {"type": "object", "properties": {"id": {"type": "string"}}}

    def test_fetch_mcp_tools_with_pagination(self):
        """Test fetching tools with multiple pages."""
        # First page
        mock_tool1 = MagicMock()
        mock_tool1.name = "tool_1"
        mock_tool1.description = "Tool 1"
        mock_tool1.inputSchema = {}

        mock_result1 = MagicMock()
        mock_result1.tools = [mock_tool1]
        mock_result1.nextCursor = "cursor_page_2"

        # Second page
        mock_tool2 = MagicMock()
        mock_tool2.name = "tool_2"
        mock_tool2.description = "Tool 2"
        mock_tool2.inputSchema = None  # Test None inputSchema

        mock_result2 = MagicMock()
        mock_result2.tools = [mock_tool2]
        mock_result2.nextCursor = None

        # Create mock session with pagination
        mock_session = AsyncMock()
        mock_session.initialize = AsyncMock()
        mock_session.list_tools = AsyncMock(side_effect=[mock_result1, mock_result2])
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        @asynccontextmanager
        async def mock_streamable_client(endpoint, **_kwargs: object):
            yield (MagicMock(), MagicMock(), MagicMock())

        with (
            patch(
                "mcp.client.streamable_http.streamable_http_client",
                side_effect=mock_streamable_client,
            ),
            patch("mcp.client.session.ClientSession", return_value=mock_session),
            patch("mcp.types.Implementation", MagicMock()),
        ):
            result = fetch_mcp_tools("https://api.example.com/mcp", {})

            assert len(result) == 2
            assert result[0].name == "tool_1"
            assert result[1].name == "tool_2"
            assert result[1].input_schema == {}  # None should become empty dict
            assert mock_session.list_tools.call_count == 2


class TestCatalogCache:
    """Verify fetch_tools memoization on StackOneToolSet._catalog_cache."""

    def test_repeat_calls_hit_cache(self, monkeypatch):
        calls = {"count": 0}

        def fake_fetch(
            _endpoint: str, _headers: dict[str, str], **_kwargs: object
        ) -> list[McpToolDefinition]:
            calls["count"] += 1
            return [McpToolDefinition(name="t", description="d", input_schema={})]

        monkeypatch.setattr("stackone_ai.toolset.fetch_mcp_tools", fake_fetch)

        toolset = StackOneToolSet(api_key="test-key", account_id="acc1")
        toolset.fetch_tools()
        toolset.fetch_tools()
        toolset.fetch_tools()

        assert calls["count"] == 1

    def test_different_accounts_separate_cache_entries(self, monkeypatch):
        calls = {"count": 0}

        def fake_fetch(_endpoint: str, headers: dict[str, str], **_kwargs: object) -> list[McpToolDefinition]:
            calls["count"] += 1
            acc = headers.get("x-account-id", "none")
            return [McpToolDefinition(name=f"t_{acc}", description="d", input_schema={})]

        monkeypatch.setattr("stackone_ai.toolset.fetch_mcp_tools", fake_fetch)

        toolset = StackOneToolSet(api_key="test-key", account_id="acc1")
        toolset.fetch_tools(account_ids=["a"])
        toolset.fetch_tools(account_ids=["a"])
        assert calls["count"] == 1

        toolset.fetch_tools(account_ids=["b"])
        assert calls["count"] == 2

        toolset.fetch_tools(account_ids=["a"])
        assert calls["count"] == 2

    def test_filters_are_applied_to_one_cached_listing(self, monkeypatch):
        """Filtering is local, so changing a filter must not re-fetch the catalog.

        The cache holds what the server listed for an account scope; providers and
        actions narrow that list in memory.
        """
        calls = {"count": 0}

        def fake_fetch(
            _endpoint: str, _headers: dict[str, str], **_kwargs: object
        ) -> list[McpToolDefinition]:
            calls["count"] += 1
            return [
                McpToolDefinition(name="prov_list_a", description="", input_schema={}),
                McpToolDefinition(name="other_get_b", description="", input_schema={}),
            ]

        monkeypatch.setattr("stackone_ai.toolset.fetch_mcp_tools", fake_fetch)

        toolset = StackOneToolSet(api_key="test-key", account_id="acc1")
        assert len(toolset.fetch_tools()) == 2
        assert len(toolset.fetch_tools(providers=["prov"])) == 1
        assert len(toolset.fetch_tools(providers=["prov"])) == 1
        assert len(toolset.fetch_tools(actions=["*_list_*"])) == 1

        assert calls["count"] == 1

    def test_clear_catalog_cache_forces_refetch(self, monkeypatch):
        calls = {"count": 0}

        def fake_fetch(
            _endpoint: str, _headers: dict[str, str], **_kwargs: object
        ) -> list[McpToolDefinition]:
            calls["count"] += 1
            return []

        monkeypatch.setattr("stackone_ai.toolset.fetch_mcp_tools", fake_fetch)

        toolset = StackOneToolSet(api_key="test-key", account_id="acc1")
        toolset.fetch_tools()
        toolset.clear_catalog_cache()
        toolset.fetch_tools()

        assert calls["count"] == 2

    def test_account_id_ordering_does_not_affect_cache_hits(self, monkeypatch):
        calls = {"count": 0}

        def fake_fetch(
            _endpoint: str, _headers: dict[str, str], **_kwargs: object
        ) -> list[McpToolDefinition]:
            calls["count"] += 1
            return [McpToolDefinition(name="t", description="", input_schema={})]

        monkeypatch.setattr("stackone_ai.toolset.fetch_mcp_tools", fake_fetch)

        toolset = StackOneToolSet(api_key="test-key", account_id="acc1")
        toolset.fetch_tools(account_ids=["a", "b"])
        # Reordered account list should hit the cache — matches Node SDK behavior.
        toolset.fetch_tools(account_ids=["b", "a"])

        assert calls["count"] == 2  # one call per account, total = 2

    def test_set_accounts_invalidates_cache(self, monkeypatch):
        calls = {"count": 0}

        def fake_fetch(_endpoint: str, headers: dict[str, str], **_kwargs: object) -> list[McpToolDefinition]:
            calls["count"] += 1
            acc = headers.get("x-account-id", "none")
            return [McpToolDefinition(name=f"t_{acc}", description="", input_schema={})]

        monkeypatch.setattr("stackone_ai.toolset.fetch_mcp_tools", fake_fetch)

        toolset = StackOneToolSet(api_key="test-key", account_id="acc1")
        toolset.set_accounts(["a"])
        toolset.fetch_tools()
        toolset.fetch_tools()
        assert calls["count"] == 1

        toolset.set_accounts(["b"])
        toolset.fetch_tools()
        assert calls["count"] == 2


class TestParallelFetch:
    """Verify per-account fetches run concurrently."""

    def test_parallel_across_accounts_bounded_by_slowest(self, monkeypatch):
        import time

        per_call_delay = 0.15

        def slow_fetch(
            _endpoint: str, _headers: dict[str, str], **_kwargs: object
        ) -> list[McpToolDefinition]:
            time.sleep(per_call_delay)
            return [McpToolDefinition(name="t", description="", input_schema={})]

        monkeypatch.setattr("stackone_ai.toolset.fetch_mcp_tools", slow_fetch)

        toolset = StackOneToolSet(api_key="test-key", account_id="acc1")
        account_ids = [f"acc-{i}" for i in range(5)]

        start = time.perf_counter()
        toolset.fetch_tools(account_ids=account_ids)
        elapsed = time.perf_counter() - start

        # Sequential would be 5 * 0.15 = 0.75s; parallel should finish in well
        # under half that. Give generous headroom for CI jitter.
        assert elapsed < 0.45, f"expected parallel fetch, took {elapsed:.2f}s"

    def test_preserves_all_tools_regardless_of_completion_order(self, monkeypatch):
        def fake_fetch(_endpoint: str, headers: dict[str, str], **_kwargs: object) -> list[McpToolDefinition]:
            acc = headers.get("x-account-id", "none")
            return [McpToolDefinition(name=f"tool_{acc}", description="", input_schema={})]

        monkeypatch.setattr("stackone_ai.toolset.fetch_mcp_tools", fake_fetch)

        toolset = StackOneToolSet(api_key="test-key", account_id="acc1")
        tools = toolset.fetch_tools(account_ids=["a", "b", "c", "d"])
        names = {t.name for t in tools.to_list()}
        assert names == {"tool_a", "tool_b", "tool_c", "tool_d"}

    def test_healthy_accounts_survive_one_failure(self, monkeypatch, caplog):
        """One unusable account must not cost the caller every other account's tools."""

        def flaky_fetch(
            _endpoint: str, headers: dict[str, str], **_kwargs: object
        ) -> list[McpToolDefinition]:
            acc = headers.get("x-account-id", "none")
            if acc == "b":
                raise RuntimeError("boom")
            return [McpToolDefinition(name=f"tool_{acc}", description="", input_schema={})]

        monkeypatch.setattr("stackone_ai.toolset.fetch_mcp_tools", flaky_fetch)

        toolset = StackOneToolSet(api_key="test-key", account_id="acc1")
        with caplog.at_level("WARNING"):
            tools = toolset.fetch_tools(account_ids=["a", "b"])

        assert [t.name for t in tools] == ["tool_a"]
        assert "b" in caplog.text and "boom" in caplog.text

    def test_all_accounts_failing_raises(self, monkeypatch):
        """If nothing succeeded, that is a failure, not an empty catalog."""
        from stackone_ai.types import ToolsetLoadError

        def always_fails(
            _endpoint: str, headers: dict[str, str], **_kwargs: object
        ) -> list[McpToolDefinition]:
            raise RuntimeError(f"boom for {headers.get('x-account-id')}")

        monkeypatch.setattr("stackone_ai.toolset.fetch_mcp_tools", always_fails)

        toolset = StackOneToolSet(api_key="test-key", account_id="acc1")
        with pytest.raises(ToolsetLoadError, match="No account returned tools"):
            toolset.fetch_tools(account_ids=["a", "b"])


class TestMcpEndpoint:
    """No param-style pin: the endpoint is {base}/mcp, plus ?tool-mode= when a mode is set."""

    @pytest.mark.parametrize(
        ("mode", "expected"),
        [
            (None, "https://api.example.com/mcp"),
            ("individual", "https://api.example.com/mcp?tool-mode=individual"),
            ("search_execute", "https://api.example.com/mcp?tool-mode=search_execute"),
        ],
    )
    def test_endpoint(self, monkeypatch, mode, expected):
        captured: dict[str, str] = {}

        def fake_fetch(endpoint: str, headers: dict[str, str], **_kwargs: object) -> list[McpToolDefinition]:
            captured["endpoint"] = endpoint
            return []

        monkeypatch.setattr("stackone_ai.toolset.fetch_mcp_tools", fake_fetch)

        toolset = StackOneToolSet(api_key="test-key", base_url="https://api.example.com/", tool_mode=mode)
        toolset.fetch_tools(account_ids=["acc1"])

        assert captured["endpoint"] == expected


class TestAccountDiscovery:
    """An API key alone must work: /mcp requires an account, so the SDK finds one.

    These cover the defect class the old suite could not see — the mock used to
    invent an account when the header was absent, so an SDK that never sent one
    still got a full catalog.
    """

    def test_api_key_alone_discovers_an_account(self, mcp_mock_server: str):
        toolset = StackOneToolSet(api_key="test-key", base_url=mcp_mock_server)
        tools = toolset.fetch_tools()
        assert len(tools) > 0

    def test_inactive_accounts_are_skipped(self, mcp_mock_server: str):
        """The mock serves one active account and one in error; only the active one is used."""
        toolset = StackOneToolSet(api_key="test-key", base_url=mcp_mock_server)
        assert toolset._discover_account_ids() == ["default"]

    def test_a_discovery_in_flight_during_a_clear_is_not_cached(self, monkeypatch):
        """An account list fetched before clear_catalog_cache() must not outlive it."""
        toolset = StackOneToolSet(api_key="test-key")
        responses = iter([["before"], ["after"]])

        def fetch_accounts() -> list[dict[str, Any]]:
            listed = next(responses)
            if listed == ["before"]:
                # The clear lands while this GET /accounts is still in flight.
                toolset.clear_catalog_cache()
            return [{"id": account, "provider": "p", "status": "active"} for account in listed]

        monkeypatch.setattr(toolset, "fetch_accounts", fetch_accounts)

        assert toolset._discover_account_ids() == ["before"]
        assert toolset._discover_account_ids() == ["after"]
        assert toolset._discover_account_ids() == ["after"]

    def test_a_catalog_resolved_before_a_mid_discovery_clear_is_not_cached(self, monkeypatch):
        """The catalog for the pre-clear discovered account must not be cached under the new generation."""
        toolset = StackOneToolSet(api_key="test-key")
        account_responses = iter([["before"], ["after"]])

        def fetch_accounts() -> list[dict[str, Any]]:
            listed = next(account_responses)
            if listed == ["before"]:
                # The clear lands while this GET /accounts is still in flight.
                toolset.clear_catalog_cache()
            return [{"id": account, "provider": "p", "status": "active"} for account in listed]

        monkeypatch.setattr(toolset, "fetch_accounts", fetch_accounts)

        calls: list[str | None] = []

        def fake_fetch(_endpoint: str, headers: dict[str, str], **_kwargs: object) -> list[McpToolDefinition]:
            calls.append(headers.get("x-account-id"))
            return []

        monkeypatch.setattr("stackone_ai.toolset.fetch_mcp_tools", fake_fetch)

        toolset.fetch_tools()  # discovers "before", races the clear
        toolset.fetch_tools()  # discovers "after", caches it cleanly

        calls.clear()
        toolset.fetch_tools(account_ids=["before"])

        # Must refetch: the catalog for "before" was resolved against a generation
        # the clear had already moved past, so it must not have been cached.
        assert calls == ["before"]

    def test_concurrent_discovery_shares_one_in_flight_request(self, monkeypatch):
        """Two threads discovering accounts at once make one GET /accounts between them."""
        toolset = StackOneToolSet(api_key="test-key")
        call_count = 0
        started = threading.Event()
        proceed = threading.Event()

        def fetch_accounts() -> list[dict[str, Any]]:
            nonlocal call_count
            call_count += 1
            started.set()
            assert proceed.wait(timeout=5)
            return [{"id": "acc", "provider": "p", "status": "active"}]

        monkeypatch.setattr(toolset, "fetch_accounts", fetch_accounts)

        results: list[list[str]] = []

        def worker() -> None:
            results.append(toolset._discover_account_ids())

        t1 = threading.Thread(target=worker)
        t1.start()
        assert started.wait(timeout=5)

        t2 = threading.Thread(target=worker)
        t2.start()
        # t1 cannot finish before proceed is set, so this just gives the scheduler a
        # chance to get t2 into its wait on the shared future before we release t1.
        time.sleep(0.05)
        proceed.set()

        t1.join(timeout=5)
        t2.join(timeout=5)

        assert call_count == 1
        assert results == [["acc"], ["acc"]]

    def test_concurrent_discovery_failure_reaches_both_waiters_and_is_not_cached(self, monkeypatch):
        toolset = StackOneToolSet(api_key="test-key")
        call_count = 0
        started = threading.Event()
        proceed = threading.Event()
        t2_entered = threading.Event()

        def failing_fetch_accounts() -> list[dict[str, Any]]:
            nonlocal call_count
            call_count += 1
            started.set()
            assert proceed.wait(timeout=5)
            assert t2_entered.wait(timeout=5)
            raise StackOneAPIError("boom", 500, "boom")

        monkeypatch.setattr(toolset, "fetch_accounts", failing_fetch_accounts)

        errors: list[StackOneAPIError] = []

        def worker() -> None:
            try:
                toolset._discover_account_ids()
            except StackOneAPIError as exc:
                errors.append(exc)

        def waiter() -> None:
            # Signalled right before becoming a waiter, so the owner's raise is held
            # back until this thread is certain to be the second call into
            # _discover_account_ids(), rather than racing a fixed sleep.
            t2_entered.set()
            worker()

        t1 = threading.Thread(target=worker)
        t1.start()
        assert started.wait(timeout=5)

        t2 = threading.Thread(target=waiter)
        t2.start()
        proceed.set()

        t1.join(timeout=5)
        t2.join(timeout=5)

        assert call_count == 1
        assert len(errors) == 2
        assert toolset._discovered_account_ids is None

        # A later call retries rather than reusing the failure.
        monkeypatch.setattr(
            toolset, "fetch_accounts", lambda: [{"id": "acc", "provider": "p", "status": "active"}]
        )
        assert toolset._discover_account_ids() == ["acc"]

    def test_clear_during_a_shared_discovery_discards_the_result(self, monkeypatch):
        """A clear_catalog_cache() while a shared discovery is in flight is not undone
        by that discovery writing its result back after the clear."""
        toolset = StackOneToolSet(api_key="test-key")
        call_count = 0
        started = threading.Event()
        proceed = threading.Event()

        def fetch_accounts() -> list[dict[str, Any]]:
            nonlocal call_count
            call_count += 1
            started.set()
            assert proceed.wait(timeout=5)
            return [{"id": "acc", "provider": "p", "status": "active"}]

        monkeypatch.setattr(toolset, "fetch_accounts", fetch_accounts)

        result_holder: dict[str, list[str]] = {}

        def worker() -> None:
            result_holder["result"] = toolset._discover_account_ids()

        t = threading.Thread(target=worker)
        t.start()
        assert started.wait(timeout=5)

        toolset.clear_catalog_cache()
        proceed.set()
        t.join(timeout=5)

        # The in-flight call still resolves for whoever was waiting on it...
        assert result_holder["result"] == ["acc"]
        assert call_count == 1
        # ...but its result is discarded rather than cached past the clear.
        assert toolset._discovered_account_ids is None

        monkeypatch.setattr(
            toolset, "fetch_accounts", lambda: [{"id": "acc2", "provider": "p", "status": "active"}]
        )
        assert toolset._discover_account_ids() == ["acc2"]

    def test_missing_account_header_is_rejected_by_the_server(self, mcp_mock_server: str):
        """Guards the mock itself: if it stops enforcing this, these tests go hollow."""
        import httpx

        response = httpx.post(
            f"{mcp_mock_server}/mcp",
            headers={"Authorization": "Basic dGVzdC1rZXk6"},
            json={"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
            timeout=10,
        )
        assert response.status_code == 400
        assert "x-account-id" in response.text


class TestListingFailuresAreDiagnosable:
    """A failure must name the status, not hide it behind TaskGroup boilerplate."""

    def test_http_error_carries_status_and_is_not_taskgroup_noise(self, monkeypatch):
        import httpx

        from stackone_ai.tools import _describe_mcp_failure

        request = httpx.Request("POST", "https://api.example.com/mcp")
        response = httpx.Response(412, text='{"message":"re-link the account"}', request=request)
        inner = httpx.HTTPStatusError("412", request=request, response=response)
        # Mirrors reality: the MCP client raises inside a TaskGroup, so the real
        # error arrives wrapped in an ExceptionGroup.
        grouped = ExceptionGroup("unhandled errors in a TaskGroup", [inner])

        err = _describe_mcp_failure(grouped, "https://api.example.com/mcp", 60.0)

        assert "412" in str(err)
        assert "TaskGroup" not in str(err)
        assert getattr(err, "status_code", None) == 412

    def test_a_real_error_response_carries_the_servers_explanation(self, mcp_mock_server: str):
        """Through the real transport, whose stream is closed by the time the error surfaces.

        The test above hands in an already-read response, so it passed while every live
        failure said only "400 Bad Request" and dropped why, e.g. "Legacy accounts cannot
        be used with MCP".
        """
        toolset = StackOneToolSet(api_key="test-key", base_url=mcp_mock_server)
        with pytest.raises(StackOneAPIError) as excinfo:
            toolset.fetch_tools(account_ids=["legacy-account"])
        assert excinfo.value.status_code == 400
        assert "Legacy accounts cannot be used with MCP" in str(excinfo.value)
        assert "Legacy accounts cannot be used with MCP" in excinfo.value.response_body

    def test_non_http_error_reports_the_leaf_not_the_group(self):
        from stackone_ai.tools import _describe_mcp_failure

        grouped = ExceptionGroup("unhandled errors in a TaskGroup", [ConnectionError("no route")])

        err = _describe_mcp_failure(grouped, "https://api.example.com/mcp", 60.0)

        assert "no route" in str(err)
        assert "TaskGroup" not in str(err)


class TestToolModeRouting:
    """Every tool is an MCP tool, whatever the mode."""

    @pytest.mark.parametrize("mode", [None, "individual", "search_execute"])
    def test_every_mode_builds_mcp_tools(self, monkeypatch, mode):
        from stackone_ai.tools import StackOneMcpTool

        monkeypatch.setattr(
            "stackone_ai.toolset.fetch_mcp_tools",
            lambda _e, _h, **_k: [McpToolDefinition(name="t", description="", input_schema={})],
        )
        toolset = StackOneToolSet(api_key="k", account_id="acc1", tool_mode=mode)
        assert isinstance(toolset.fetch_tools().get_tool("t"), StackOneMcpTool)

    def test_search_execute_mode_is_requested_on_the_url(self, monkeypatch):
        seen: list[str] = []

        def capture(endpoint: str, _headers: dict[str, str], **_kwargs: object) -> list[McpToolDefinition]:
            seen.append(endpoint)
            return []

        monkeypatch.setattr("stackone_ai.toolset.fetch_mcp_tools", capture)
        StackOneToolSet(api_key="k", account_id="acc1", tool_mode="search_execute").fetch_tools()
        assert "tool-mode=search_execute" in seen[0]


class TestMcpCallFailuresSurface:
    """A tools/call failure arrives as a normal response with isError set."""

    def test_is_error_result_raises_rather_than_returning_a_success(self, monkeypatch):
        from stackone_ai.types import StackOneAPIError

        class _Part:
            text = '{"error":"Lambda execution failed"}'

        class _Result:
            isError = True
            content = [_Part()]

        from stackone_ai.tools import parse_tool_result

        with pytest.raises(StackOneAPIError, match="Lambda execution failed"):
            parse_tool_result(_Result(), "t")

    def test_successful_result_is_parsed(self):
        from stackone_ai.tools import parse_tool_result

        class _Part:
            text = '{"actions":[{"action_id":"linear_list_comments"}]}'

        class _Result:
            isError = False
            content = [_Part()]

        assert parse_tool_result(_Result(), "t")["actions"][0]["action_id"] == "linear_list_comments"


def _text(text: str) -> dict[str, str]:
    return {"type": "text", "text": text}


_IMAGE = {"type": "image", "data": "AAAA", "mimeType": "image/png"}


class TestToolResultParity:
    """parse_tool_result() matches the Node SDK's parseToolResult() case for case.

    A result carrying only structuredContent came back as `{}` — a success with the
    whole payload missing. Text wins when both are present, in success and in error.
    """

    @pytest.mark.parametrize(
        ("result", "expected"),
        [
            pytest.param({"content": [_text('{"a":1}')]}, {"a": 1}, id="text-object"),
            pytest.param({"content": [_text("[1,2]")]}, {"result": [1, 2]}, id="text-array"),
            pytest.param({"content": [_text("plain")]}, {"result": "plain"}, id="text-plain"),
            pytest.param({"content": [_text("NaN")]}, {"result": "NaN"}, id="text-nan-is-not-json"),
            pytest.param({"content": [_text('{"a":'), _text("1}")]}, {"a": 1}, id="text-parts-joined"),
            pytest.param(
                {"content": [], "structuredContent": {"ok": True}}, {"ok": True}, id="structured-only"
            ),
            pytest.param(
                {"content": [_text('{"a":1}')], "structuredContent": {"b": 2}},
                {"a": 1},
                id="text-wins-over-structured",
            ),
            pytest.param(
                {"content": [_text("")], "structuredContent": {"ok": True}},
                {"ok": True, "content_parts": [_text("")]},
                id="empty-text-is-not-text",
            ),
            pytest.param({"content": []}, {}, id="neither"),
            pytest.param(
                {"content": [_text('{"a":1}'), _IMAGE]},
                {"a": 1, "content_parts": [_IMAGE]},
                id="text-and-image",
            ),
            pytest.param(
                {"content": [_IMAGE], "structuredContent": {"ok": True}},
                {"ok": True, "content_parts": [_IMAGE]},
                id="structured-and-image",
            ),
        ],
    )
    def test_success(self, result: dict, expected: dict):
        self._assert_parses_to(result, expected)

    @staticmethod
    def _assert_parses_to(result: dict, expected: dict) -> None:
        from mcp.types import CallToolResult

        from stackone_ai.tools import parse_tool_result

        parsed = parse_tool_result(CallToolResult.model_validate(result), "t")
        # Non-text parts are kept as the MCP types they arrived as; compare their wire form.
        if "content_parts" in parsed:
            parsed["content_parts"] = [part.model_dump(exclude_none=True) for part in parsed["content_parts"]]
        assert parsed == expected

    @pytest.mark.parametrize(
        ("result", "message", "status", "body"),
        [
            pytest.param(
                {"content": [_text('{"error":"Lambda execution failed","status_code":502}')]},
                'Tool "t" failed: {"error":"Lambda execution failed","status_code":502}',
                502,
                {"error": "Lambda execution failed", "status_code": 502},
                id="text",
            ),
            pytest.param(
                {"content": [], "structuredContent": {"error": "boom", "status_code": 503}},
                'Tool "t" failed: {"error":"boom","status_code":503}',
                503,
                {"error": "boom", "status_code": 503},
                id="structured-only",
            ),
            pytest.param(
                {"content": [_text('{"statusCode":409}')], "structuredContent": {"status_code": 500}},
                'Tool "t" failed: {"statusCode":409}',
                409,
                {"statusCode": 409},
                id="text-wins-over-structured",
            ),
            pytest.param({"content": []}, 'Tool "t" failed: {}', 0, {}, id="neither"),
            pytest.param(
                {"content": [_text("oops")]}, 'Tool "t" failed: oops', 0, {"result": "oops"}, id="plain-text"
            ),
            pytest.param(
                {"content": [_IMAGE], "structuredContent": {"status_code": 404}},
                'Tool "t" failed: {"status_code":404}',
                404,
                {"status_code": 404},
                id="structured-and-image",
            ),
        ],
    )
    def test_is_error(self, result: dict, message: str, status: int, body: dict):
        from mcp.types import CallToolResult

        from stackone_ai.tools import parse_tool_result

        with pytest.raises(StackOneAPIError) as excinfo:
            parse_tool_result(CallToolResult.model_validate({**result, "isError": True}), "t")
        assert str(excinfo.value) == message
        assert excinfo.value.status_code == status
        assert excinfo.value.response_body == body

    @pytest.mark.parametrize(
        "payload",
        [
            pytest.param({"isError": False, "result": {"data": {"id": "1"}}}, id="wrapper"),
            pytest.param(
                {
                    "isError": False,
                    "result": {"data": []},
                    "defenderMetadata": {"flagged": 0},
                    "policyMetadata": {"policy": "p1"},
                },
                id="wrapper-with-metadata",
            ),
            pytest.param({"isError": False, "result": [1, 2]}, id="wrapper-with-a-list"),
            pytest.param(
                {"session_id": "s1", "actions": [{"action_id": "a", "similarity_score": 0.9}]},
                id="search-result",
            ),
            pytest.param({"message": "Feedback recorded", "session_id": None}, id="feedback-receipt"),
        ],
    )
    def test_a_result_is_returned_as_the_server_wrote_it(self, payload: dict):
        """UCA's { isError: false, result, ...metadata } wrapper included, as Node returns it."""
        import json

        self._assert_parses_to({"content": [_text(json.dumps(payload))]}, payload)
        self._assert_parses_to({"content": [], "structuredContent": payload}, payload)
        self._assert_parses_to(
            {"content": [_text(json.dumps(payload))], "structuredContent": payload}, payload
        )

    def test_uca_wrapper_over_the_wire(self, mcp_mock_server: str):
        """The mock answers a per-action tools/call as UCA does: wrapper as structuredContent and text."""
        from stackone_ai.tools import build_auth_header, call_mcp_tool

        headers = {"Authorization": build_auth_header("test-key"), "x-account-id": "test-account"}
        result = call_mcp_tool(f"{mcp_mock_server}/mcp", headers, "dummy_action", {"foo": "bar"})
        assert result == {
            "isError": False,
            "result": {"data": {"action": "dummy_action", "received": {"foo": "bar"}}},
        }


class TestSearchAndExecuteApi:
    """The recommended surface: search() then execute(), no account id needed."""

    def test_search_returns_actions(self, mcp_mock_server: str):
        toolset = StackOneToolSet(api_key="test-key", base_url=mcp_mock_server)
        actions = toolset.search("list items")
        assert [a["action_id"] for a in actions] == ["mock_list_items"]

    def test_execute_runs_a_searched_action(self, mcp_mock_server: str):
        toolset = StackOneToolSet(api_key="test-key", base_url=mcp_mock_server)
        action_id = toolset.search("list items")[0]["action_id"]
        assert toolset.execute(action_id)["result"]["data"] == {"nodes": []}

    def test_execute_passes_the_nested_envelope_through(self, mcp_mock_server: str):
        """execute() takes the envelope an action's example_request shows, verbatim.

        Not the flat `query_pageSize` form: that belongs to fetch_tools() tools, whose
        own served schema names the keys. One id must not mean two argument shapes.
        """
        toolset = StackOneToolSet(api_key="test-key", base_url=mcp_mock_server)
        action = toolset.search("list items")[0]
        assert action["example_request"] == {"query": {"page_size": 25}}

        result = toolset.execute(action["action_id"], action["example_request"])
        assert result["result"]["echoed_query"] == {"page_size": 25}

    def test_execute_forwards_host_headers_but_cannot_switch_tenant(self, mcp_mock_server: str):
        """The served `headers` object is open, so a host header reaches the action — but
        neither form of header argument can carry the SDK's own headers onto the wire."""
        toolset = StackOneToolSet(api_key="test-key", base_url=mcp_mock_server)
        _reset_requests(mcp_mock_server)

        toolset.execute(
            "mock_list_items",
            {
                "query": {"page_size": 25},
                "headers": {"x-custom": "kept", "x-account-id": "victim", "Authorization": "Basic stolen"},
                "headers_x-account-id": "victim",
            },
            account_ids=["default"],
        )

        [call] = _tool_calls(mcp_mock_server, "mock_default_execute_action")
        assert call["accountId"] == "default"
        assert call["arguments"] == {
            "query": {"page_size": 25},
            "headers": {"x-custom": "kept"},
            "action_id": "mock_list_items",
        }

    def test_unknown_action_raises_rather_than_returning_an_error_body(self, mcp_mock_server: str):
        from stackone_ai.types import StackOneAPIError

        toolset = StackOneToolSet(api_key="test-key", base_url=mcp_mock_server)
        with pytest.raises(StackOneAPIError, match="Unknown action"):
            toolset.execute("mock_not_a_real_action")

    def test_fetch_accounts_lists_linked_accounts(self, mcp_mock_server: str):
        toolset = StackOneToolSet(api_key="test-key", base_url=mcp_mock_server)
        accounts = toolset.fetch_accounts()
        assert {a["id"] for a in accounts} == {"default", "dead"}
        assert [a["id"] for a in accounts if a["status"] == "active"] == ["default"]


class TestServerRefusals:
    """The mock refuses what the real API refuses, so a bug here cannot stay green."""

    def test_unknown_account_is_rejected_rather_than_served_a_default_catalog(self, mcp_mock_server: str):
        """Serving a catalog for an account that does not exist hides every id bug."""
        toolset = StackOneToolSet(api_key="test-key", base_url=mcp_mock_server)
        with pytest.raises(ToolsetError):
            toolset.fetch_tools(account_ids=["no-such-account"])

    def test_execution_without_an_account_is_rejected_by_the_server(self, mcp_mock_server: str):
        """tools/call is account-scoped too — the sibling of the bug that shipped."""
        toolset = StackOneToolSet(api_key="test-key", base_url=mcp_mock_server)
        tool = toolset.fetch_tools(account_ids=["test-account"]).get_tool("dummy_action")
        assert tool is not None
        tool.set_account_id(None)

        with pytest.raises(StackOneAPIError) as excinfo:
            tool.execute({"foo": "bar"})
        assert excinfo.value.status_code == 400


class TestMockServesSchemasVerbatim:
    """The mock once listed every per-action JSON Schema as `properties: {}`.

    MCP's high-level McpServer expects a Zod shape, and given a plain JSON Schema it
    served an empty object, so no schema test against the mock ever saw a declared
    parameter. A catalog with every field missing still looked like a working one.
    """

    def test_declared_properties_arrive(self, mcp_mock_server: str):
        toolset = StackOneToolSet(api_key="test-key", base_url=mcp_mock_server)
        tool = toolset.fetch_tools(account_ids=["test-account"]).get_tool("dummy_action")
        assert tool is not None

        assert tool.to_openai_function()["function"]["parameters"] == {
            "type": "object",
            "properties": {"foo": {"type": "string", "description": "A string parameter"}},
            "required": ["foo"],
            "additionalProperties": False,
        }


class TestRecentlyFixedBehaviour:
    """Pins for fixes that could otherwise be reverted with the suite still green."""

    def test_provider_filter_matches_the_full_connector_prefix(self, monkeypatch):
        """Splitting on the first underscore read browser_linkedin as browser."""

        def fake_fetch(
            _endpoint: str, _headers: dict[str, str], **_kwargs: object
        ) -> list[McpToolDefinition]:
            return [
                McpToolDefinition(name="browser_linkedin_search_people", description="", input_schema={}),
                McpToolDefinition(name="browser_open_page", description="", input_schema={}),
            ]

        monkeypatch.setattr("stackone_ai.toolset.fetch_mcp_tools", fake_fetch)
        toolset = StackOneToolSet(api_key="test-key", account_id="acc1")
        assert [t.name for t in toolset.fetch_tools(providers=["browser_linkedin"])] == [
            "browser_linkedin_search_people"
        ]

    @pytest.mark.parametrize("top_k", [0, -1, 51, 1000, "ten", None, 1.5, True])
    def test_search_rejects_out_of_range_top_k_without_a_round_trip(self, monkeypatch, top_k):
        """The server caps top_k at 50, but only after a request per connector."""

        def explode(*_args, **_kwargs):
            raise AssertionError("search() must reject top_k before reaching the network")

        monkeypatch.setattr("stackone_ai.toolset.fetch_mcp_tools", explode)
        toolset = StackOneToolSet(api_key="test-key", account_id="acc1")
        with pytest.raises(ToolsetConfigError, match="top_k"):
            toolset.search("anything", top_k=top_k)

    def test_execute_rejects_non_dict_arguments(self, monkeypatch):
        def explode(*_args, **_kwargs):
            raise AssertionError("execute() must reject bad arguments before reaching the network")

        monkeypatch.setattr("stackone_ai.toolset.fetch_mcp_tools", explode)
        toolset = StackOneToolSet(api_key="test-key", account_id="acc1")
        with pytest.raises(ToolsetConfigError, match="JSON object"):
            toolset.execute("linear_list_issues", [1, 2, 3])

    def test_fetch_accounts_rejects_a_non_list_body(self, monkeypatch):
        """list(dict) yields the keys, which blew up much later as an AttributeError."""
        import httpx

        def fake_send(*_args, **_kwargs):
            return httpx.Response(200, json={"results": [{"id": "a"}]})

        monkeypatch.setattr(httpx.HTTPTransport, "handle_request", fake_send)
        toolset = StackOneToolSet(api_key="test-key")
        with pytest.raises(ToolsetLoadError, match="Unexpected /accounts response shape"):
            toolset.fetch_accounts()

    def test_fetch_accounts_handles_invalid_or_malformed_encoding(self, monkeypatch):
        """Non-JSON or invalid UTF-8 (e.g. b'[\xff]') must raise ToolsetLoadError, not escape."""
        import httpx

        def fake_send(*_args, **_kwargs):
            return httpx.Response(200, content=b"[\xff]")

        monkeypatch.setattr(httpx.HTTPTransport, "handle_request", fake_send)
        toolset = StackOneToolSet(api_key="test-key")
        with pytest.raises(ToolsetLoadError, match="Invalid JSON returned by"):
            toolset.fetch_accounts()


class TestCacheIsolation:
    """The cache must not be defeatable, and must not hand out shared mutable state."""

    def test_clear_during_an_in_flight_fetch_is_not_undone_by_it(self, monkeypatch):
        """A listing already being fetched must not land after the clear that cancels it.

        Otherwise the stale catalog is written back afterwards and served for the life
        of the process — exactly what clear_catalog_cache() exists to prevent.
        """
        toolset = StackOneToolSet(api_key="test-key", account_id="acc1")

        def fake_fetch(
            _endpoint: str, _headers: dict[str, str], **_kwargs: object
        ) -> list[McpToolDefinition]:
            # Clear midway through the fetch, as a concurrent caller would.
            toolset.clear_catalog_cache()
            return [McpToolDefinition(name="stale_tool", description="", input_schema={})]

        monkeypatch.setattr("stackone_ai.toolset.fetch_mcp_tools", fake_fetch)
        toolset.fetch_tools()
        assert toolset._catalog_cache == {}

    def test_nested_schema_is_not_shared_between_callers(self, monkeypatch):
        """Rebuilding tools per call copied the tool but not the schema graph under it."""

        def fake_fetch(
            _endpoint: str, _headers: dict[str, str], **_kwargs: object
        ) -> list[McpToolDefinition]:
            return [
                McpToolDefinition(
                    name="t",
                    description="",
                    input_schema={"properties": {"body_x": {"type": "object", "properties": {}}}},
                )
            ]

        monkeypatch.setattr("stackone_ai.toolset.fetch_mcp_tools", fake_fetch)
        toolset = StackOneToolSet(api_key="test-key", account_id="acc1")

        first = toolset.fetch_tools().to_list()[0]
        first.parameters.properties["body_x"]["properties"]["injected"] = {"type": "string"}

        second = toolset.fetch_tools().to_list()[0]
        assert "injected" not in second.parameters.properties["body_x"]["properties"]


class TestExecuteReturnShape:
    def test_execute_returns_the_same_shape_as_the_tool(self, mcp_mock_server: str):
        """Both return the result as the server wrote it, UCA's wrapper included."""
        toolset = StackOneToolSet(api_key="test-key", base_url=mcp_mock_server)
        assert toolset.execute("mock_list_items") == {
            "isError": False,
            "result": {"data": {"nodes": []}, "echoed_query": None},
        }


@contextmanager
def _silent_host() -> Iterator[str]:
    """The base URL of a host that accepts connections and never answers."""
    import socket

    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(8)
    port = listener.getsockname()[1]
    accepted: list[socket.socket] = []

    def accept_and_stay_silent() -> None:
        try:
            while True:
                connection, _ = listener.accept()
                accepted.append(connection)
        except OSError:
            return

    threading.Thread(target=accept_and_stay_silent, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        listener.close()
        for connection in accepted:
            connection.close()


def _within(seconds: float, fn: Any) -> BaseException | None:
    """Run ``fn`` on a thread with a join deadline, and return what it raised.

    If a timeout regresses, the test must FAIL, not hang the whole suite.
    """
    outcome: list[BaseException | None] = []

    def attempt() -> None:
        try:
            fn()
            outcome.append(None)
        except BaseException as exc:
            outcome.append(exc)

    worker = threading.Thread(target=attempt, daemon=True)
    worker.start()
    worker.join(timeout=seconds)
    assert not worker.is_alive(), "the call ignored its timeout and is still hanging"
    return outcome[0]


class TestTimeoutIsHonoured:
    def test_mcp_listing_respects_timeout_against_a_host_that_never_answers(self):
        """The MCP client's own defaults are a 300s SSE read, and timeout= was never
        passed through — so timeout=2 against a silent host hung for five minutes.
        """
        import time

        with _silent_host() as base_url:
            toolset = StackOneToolSet(api_key="k", account_id="a", base_url=base_url, timeout=1)
            started = time.monotonic()
            assert isinstance(_within(15, toolset.fetch_tools), ToolsetError)
            assert time.monotonic() - started < 10

    @pytest.mark.parametrize("in_a_running_loop", [False, True])
    def test_a_timeout_says_it_timed_out(self, in_a_running_loop: bool):
        """Not "RuntimeError: no running event loop", or "WouldBlock:" inside a loop.

        The failure walk followed __context__ to run_async's own probe for a loop, and
        reported that as the cause of every timeout.
        """
        with _silent_host() as base_url:
            toolset = StackOneToolSet(api_key="k", account_id="a", base_url=base_url, timeout=0.5)

            def fetch() -> None:
                if not in_a_running_loop:
                    toolset.fetch_tools()
                    return

                async def inside_a_loop() -> None:
                    toolset.fetch_tools()

                asyncio.run(inside_a_loop())

            error = _within(15, fetch)

        assert isinstance(error, ToolsetLoadError)
        assert str(error) == f"MCP request to {base_url}/mcp timed out after 0.5s"

    def test_a_tool_call_timeout_says_it_timed_out(self):
        with _silent_host() as base_url:
            tool = StackOneMcpTool(
                name="t",
                description="",
                parameters=ToolParameters(type="object", properties={}),
                api_key="k",
                endpoint=f"{base_url}/mcp",
                account_id="a",
                timeout=0.5,
            )
            error = _within(15, tool.execute)

        assert isinstance(error, ToolsetLoadError)
        assert str(error) == f"MCP request to {base_url}/mcp timed out after 0.5s"


class TestInterruptsPropagate:
    """Ctrl-C and SystemExit stop the caller; they are not reported as a failed MCP request.

    Caught as BaseException they came out as a ToolsetLoadError, which LangChain's
    handle_tool_error or a LangGraph ToolNode hands the model as a tool result.
    """

    @staticmethod
    def _interrupted(monkeypatch, interrupt: BaseException) -> None:
        def run_async(awaitable: Any) -> Any:
            awaitable.close()
            raise interrupt

        monkeypatch.setattr("stackone_ai.tools.run_async", run_async)

    @pytest.mark.parametrize("interrupt", [KeyboardInterrupt, SystemExit])
    def test_listing(self, monkeypatch, interrupt: type[BaseException]):
        self._interrupted(monkeypatch, interrupt())
        with pytest.raises(interrupt):
            fetch_mcp_tools("https://api.example.com/mcp", {})

    @pytest.mark.parametrize("interrupt", [KeyboardInterrupt, SystemExit])
    def test_tool_call(self, monkeypatch, interrupt: type[BaseException]):
        self._interrupted(monkeypatch, interrupt())
        tool = StackOneMcpTool(
            name="t",
            description="",
            parameters=ToolParameters(type="object", properties={}),
            api_key="k",
            endpoint="https://api.example.com/mcp",
            account_id="a",
        )
        with pytest.raises(interrupt):
            tool.execute({})

    def test_an_ordinary_failure_is_still_described(self, monkeypatch):
        self._interrupted(monkeypatch, RuntimeError("boom"))
        with pytest.raises(ToolsetLoadError, match="failed: RuntimeError: boom"):
            fetch_mcp_tools("https://api.example.com/mcp", {})


def test_every_sdk_error_is_a_stackone_error():
    """`except StackOneError` must be a real catch-all.

    The toolset errors used to be unrelated siblings, so the obvious catch-all missed
    the two errors a user is most likely to hit first.
    """
    from stackone_ai.types import StackOneError, ToolArgumentsError

    for error in (StackOneAPIError, ToolArgumentsError, ToolsetError, ToolsetConfigError, ToolsetLoadError):
        assert issubclass(error, StackOneError), error

    with pytest.raises(StackOneError):
        StackOneToolSet(api_key="k").search("x", top_k=0)


class TestExecuteRouting:
    """Pins for execute()'s routing and argument handling."""

    @staticmethod
    def _toolset_with_meta_tools(monkeypatch, names, account_ids):
        from stackone_ai.tools import StackOneMcpTool

        seen: dict[str, object] = {}

        def fake_execute(self, arguments):
            seen["tool"] = self.name
            seen["arguments"] = arguments
            return {"data": {}}

        monkeypatch.setattr(StackOneMcpTool, "execute", fake_execute)
        monkeypatch.setattr(
            "stackone_ai.toolset.fetch_mcp_tools",
            lambda _e, headers, **_k: [
                McpToolDefinition(name=name, description="", input_schema={})
                for name in names
                if name.split("_execute_action")[0].endswith(headers["x-account-id"])
            ],
        )
        toolset = StackOneToolSet(api_key="test-key")
        toolset.set_accounts(account_ids)
        return toolset, seen

    def test_a_model_supplied_action_id_cannot_override_the_pinned_one(self, monkeypatch):
        """Spreading arguments over action_id let a prompt-injected call swap the action."""
        toolset, seen = self._toolset_with_meta_tools(monkeypatch, ["linear_acc1_execute_action"], ["acc1"])
        toolset.execute("linear_list_issues", {"action_id": "linear_delete_issue"})
        assert seen["arguments"]["action_id"] == "linear_list_issues"

    def test_account_ids_containing_underscores_still_route(self, monkeypatch):
        """nanoid's alphabet includes "_"; splitting on it made the account unroutable."""
        toolset, seen = self._toolset_with_meta_tools(monkeypatch, ["linear_acc_1_execute_action"], ["acc_1"])
        toolset.execute("linear_list_issues")
        assert seen["tool"] == "linear_acc_1_execute_action"

    def test_the_longest_matching_connector_wins(self, monkeypatch):
        """With browser and browser_linkedin both linked, the first token alone misroutes."""
        toolset, seen = self._toolset_with_meta_tools(
            monkeypatch,
            ["browser_acc1_execute_action", "browser_linkedin_acc2_execute_action"],
            ["acc1", "acc2"],
        )
        toolset.execute("browser_linkedin_search_people")
        assert seen["tool"] == "browser_linkedin_acc2_execute_action"


class TestSearchRanking:
    def test_results_are_ranked_across_connectors_and_tolerate_bad_scores(self, monkeypatch):
        """Concatenation left results grouped by connector; a non-numeric score crashed."""
        from stackone_ai.tools import StackOneMcpTool

        per_connector = {
            "a_acc1_search_actions": [
                {"action_id": "a_low", "similarity_score": 0.2},
                {"action_id": "a_bad", "similarity_score": "0.99"},
            ],
            "b_acc2_search_actions": [{"action_id": "b_high", "similarity_score": 0.9}],
        }

        monkeypatch.setattr(
            StackOneMcpTool, "execute", lambda self, _args: {"actions": per_connector[self.name]}
        )
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
        assert [a["action_id"] for a in toolset.search("x")] == ["b_high", "a_low", "a_bad"]


class TestAdapterErrorsReachTheModel:
    """A rejected call must reach the agent with the server's reason, not kill the run."""

    @staticmethod
    def _failing_tool():
        from stackone_ai.tools import StackOneMcpTool
        from stackone_ai.types import ToolParameters

        tool = StackOneMcpTool(
            name="linear_get_issue",
            description="",
            parameters=ToolParameters(type="object", properties={"path_id": {"type": "string"}}),
            api_key="k",
            endpoint="https://api.example.com/mcp",
            account_id="acc1",
        )

        def reject(_self, _arguments=None):
            raise StackOneAPIError("400 Bad Request", 400, {"message": "path.id is missing"})

        return tool, reject

    def test_langchain_raises_tool_exception_carrying_the_response_body(self, monkeypatch):
        from langchain_core.tools import ToolException

        from stackone_ai.tools import StackOneMcpTool

        tool, reject = self._failing_tool()
        monkeypatch.setattr(StackOneMcpTool, "execute", reject)
        with pytest.raises(ToolException, match="path.id is missing") as excinfo:
            tool.to_langchain()._run(path_id="x")
        assert excinfo.value.status_code == 400

    def test_langchain_does_not_forward_unsupplied_optionals_as_null(self, monkeypatch):
        """The API reads an explicit null as "required field missing", so this 400'd every call."""
        from stackone_ai.tools import StackOneMcpTool

        seen: dict[str, object] = {}
        tool, _ = self._failing_tool()
        monkeypatch.setattr(StackOneMcpTool, "execute", lambda _self, args=None: seen.update(args=args) or {})
        tool.to_langchain()._run(path_id="x", body_after=None)
        assert seen["args"] == {"path_id": "x"}

    def test_pydantic_ai_raises_model_retry(self, monkeypatch):
        pytest.importorskip("pydantic_ai")
        from pydantic_ai.exceptions import ModelRetry

        from stackone_ai.tools import StackOneMcpTool

        tool, reject = self._failing_tool()
        monkeypatch.setattr(StackOneMcpTool, "execute", reject)
        with pytest.raises(ModelRetry, match="path.id is missing"):
            tool.to_pydantic_ai_tool().function(path_id="x")
