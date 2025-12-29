"""Tests for StackOneToolSet MCP functionality using real MCP mock server."""

from __future__ import annotations

from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from stackone_ai.toolset import StackOneToolSet, _fetch_mcp_tools, _McpToolDefinition


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
        assert len(tools) == 2
        tool_names = [t.name for t in tools.to_list()]
        assert "default_tool_1" in tool_names
        assert "default_tool_2" in tool_names

    def test_fetch_tools_with_account_ids(self, mcp_mock_server: str):
        """Test fetching tools with specific account IDs"""
        toolset = StackOneToolSet(api_key="test-key", base_url=mcp_mock_server)
        tools = toolset.fetch_tools(account_ids=["acc1"])
        assert len(tools) == 2
        tool_names = [t.name for t in tools.to_list()]
        assert "acc1_tool_1" in tool_names
        assert "acc1_tool_2" in tool_names

    def test_fetch_tools_uses_set_accounts(self, mcp_mock_server: str):
        """Test that fetch_tools uses set_accounts when no accountIds provided"""
        toolset = StackOneToolSet(api_key="test-key", base_url=mcp_mock_server)
        toolset.set_accounts(["acc1", "acc2"])
        tools = toolset.fetch_tools()
        # acc1 has 2 tools, acc2 has 2 tools, total should be 4
        assert len(tools) == 4
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
        # Should fetch tools only for acc3 (ignoring acc1, acc2)
        assert len(tools) == 1
        tool_names = [t.name for t in tools.to_list()]
        assert "acc3_tool_1" in tool_names
        # Verify set_accounts state is preserved
        assert toolset._account_ids == ["acc1", "acc2"]

    def test_fetch_tools_multiple_account_ids(self, mcp_mock_server: str):
        """Test fetching tools for multiple account IDs"""
        toolset = StackOneToolSet(api_key="test-key", base_url=mcp_mock_server)
        tools = toolset.fetch_tools(account_ids=["acc1", "acc2", "acc3"])
        # acc1: 2 tools, acc2: 2 tools, acc3: 1 tool = 5 total
        assert len(tools) == 5

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
        tool_names = [t.name for t in tools.to_list()]
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


class TestSchemaPropertyNormalization:
    """Test schema property normalization with monkeypatch (for precise schema control)."""

    def test_tool_properties_are_normalized(self, monkeypatch):
        """Test that tool properties are correctly extracted from input schema"""
        from stackone_ai.toolset import _McpToolDefinition

        def fake_fetch(_: str, headers: dict[str, str]) -> list[_McpToolDefinition]:
            return [
                _McpToolDefinition(
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

        monkeypatch.setattr("stackone_ai.toolset._fetch_mcp_tools", fake_fetch)

        toolset = StackOneToolSet(api_key="test-key")
        tools = toolset.fetch_tools()
        tool = tools.get_tool("test_tool")
        assert tool is not None
        assert "name" in tool.parameters.properties
        assert "age" in tool.parameters.properties

    def test_required_fields_marked_not_nullable(self, monkeypatch):
        """Test that required fields are marked as not nullable"""
        from stackone_ai.toolset import _McpToolDefinition

        def fake_fetch(_: str, headers: dict[str, str]) -> list[_McpToolDefinition]:
            return [
                _McpToolDefinition(
                    name="test_tool",
                    description="Test tool",
                    input_schema={
                        "type": "object",
                        "properties": {"id": {"type": "string"}},
                        "required": ["id"],
                    },
                )
            ]

        monkeypatch.setattr("stackone_ai.toolset._fetch_mcp_tools", fake_fetch)

        toolset = StackOneToolSet(api_key="test-key")
        tools = toolset.fetch_tools()
        tool = tools.get_tool("test_tool")
        assert tool is not None
        assert tool.parameters.properties["id"].get("nullable") is False

    def test_optional_fields_marked_nullable(self, monkeypatch):
        """Test that optional fields are marked as nullable"""
        from stackone_ai.toolset import _McpToolDefinition

        def fake_fetch(_: str, headers: dict[str, str]) -> list[_McpToolDefinition]:
            return [
                _McpToolDefinition(
                    name="test_tool",
                    description="Test tool",
                    input_schema={
                        "type": "object",
                        "properties": {"optional_field": {"type": "string"}},
                    },
                )
            ]

        monkeypatch.setattr("stackone_ai.toolset._fetch_mcp_tools", fake_fetch)

        toolset = StackOneToolSet(api_key="test-key")
        tools = toolset.fetch_tools()
        tool = tools.get_tool("test_tool")
        assert tool is not None
        assert tool.parameters.properties["optional_field"].get("nullable") is True


class TestRpcToolExecution:
    """Test RPC tool execution through the MCP server."""

    def test_execute_tool_returns_response(self, mcp_mock_server: str):
        """Test executing a tool via RPC returns response"""
        toolset = StackOneToolSet(api_key="test-key", base_url=mcp_mock_server)
        tools = toolset.fetch_tools(account_ids=["your-bamboohr-account-id"])
        tool = tools.get_tool("bamboohr_list_employees")
        assert tool is not None

        result = tool.execute()
        assert result is not None
        assert "data" in result

    def test_execute_tool_with_arguments(self, mcp_mock_server: str):
        """Test executing a tool with arguments"""
        toolset = StackOneToolSet(api_key="test-key", base_url=mcp_mock_server)
        tools = toolset.fetch_tools(account_ids=["your-bamboohr-account-id"])
        tool = tools.get_tool("bamboohr_get_employee")
        assert tool is not None

        result = tool.execute({"id": "emp-123"})
        assert result is not None
        assert result.get("data", {}).get("id") == "emp-123"

    def test_execute_tool_sends_account_id_header(self, mcp_mock_server: str):
        """Test that tool execution sends x-account-id header"""
        toolset = StackOneToolSet(api_key="test-key", base_url=mcp_mock_server)
        tools = toolset.fetch_tools(account_ids=["test-account"])
        tool = tools.get_tool("dummy_action")
        assert tool is not None
        assert tool.get_account_id() == "test-account"

        # Execute and verify account context is preserved
        result = tool.execute({"foo": "bar"})
        assert result is not None


class TestAccountIdFallback:
    """Test account ID fallback to instance account_id."""

    def test_uses_instance_account_id_when_no_other_provided(self, monkeypatch):
        """Test that fetch_tools uses instance account_id when no account_ids provided."""
        sample_tool = _McpToolDefinition(
            name="test_tool",
            description="Test tool",
            input_schema={"type": "object", "properties": {}},
        )

        captured_accounts: list[str | None] = []

        def fake_fetch(_: str, headers: dict[str, str]) -> list[_McpToolDefinition]:
            captured_accounts.append(headers.get("x-account-id"))
            return [sample_tool]

        monkeypatch.setattr("stackone_ai.toolset._fetch_mcp_tools", fake_fetch)

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
        from stackone_ai.toolset import ToolsetConfigError

        def fake_fetch(_: str, headers: dict[str, str]) -> list[_McpToolDefinition]:
            raise ToolsetConfigError("Original config error")

        monkeypatch.setattr("stackone_ai.toolset._fetch_mcp_tools", fake_fetch)

        toolset = StackOneToolSet(api_key="test_key")
        with pytest.raises(ToolsetConfigError, match="Original config error"):
            toolset.fetch_tools()


class TestFetchMcpToolsInternal:
    """Test _fetch_mcp_tools internal implementation."""

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
        @asynccontextmanager
        async def mock_streamable_client(endpoint, headers):
            yield (MagicMock(), MagicMock(), MagicMock())

        # Patch at the module where imports happen
        with (
            patch(
                "mcp.client.streamable_http.streamablehttp_client",
                side_effect=mock_streamable_client,
            ),
            patch("mcp.client.session.ClientSession", return_value=mock_session),
            patch("mcp.types.Implementation", MagicMock()),
        ):
            result = _fetch_mcp_tools("https://api.example.com/mcp", {"Authorization": "Basic test"})

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
        async def mock_streamable_client(endpoint, headers):
            yield (MagicMock(), MagicMock(), MagicMock())

        with (
            patch(
                "mcp.client.streamable_http.streamablehttp_client",
                side_effect=mock_streamable_client,
            ),
            patch("mcp.client.session.ClientSession", return_value=mock_session),
            patch("mcp.types.Implementation", MagicMock()),
        ):
            result = _fetch_mcp_tools("https://api.example.com/mcp", {})

            assert len(result) == 2
            assert result[0].name == "tool_1"
            assert result[1].name == "tool_2"
            assert result[1].input_schema == {}  # None should become empty dict
            assert mock_session.list_tools.call_count == 2
