from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from stackone_ai.toolset import StackOneToolSet, _fetch_mcp_tools, _McpToolDefinition


@pytest.fixture
def mock_mcp_catalog(monkeypatch):
    """Mock MCP fetch calls with per-account catalogs."""

    def make_tool(name: str, description: str) -> _McpToolDefinition:
        return _McpToolDefinition(
            name=name,
            description=description,
            input_schema={
                "type": "object",
                "properties": {
                    "id": {"type": "string", "description": "Record identifier"},
                },
            },
        )

    catalog: dict[str | None, list[_McpToolDefinition]] = {
        None: [
            make_tool("default_tool_1", "Default Tool 1"),
            make_tool("default_tool_2", "Default Tool 2"),
        ],
        "acc1": [
            make_tool("acc1_tool_1", "Account 1 Tool 1"),
            make_tool("acc1_tool_2", "Account 1 Tool 2"),
        ],
        "acc2": [
            make_tool("acc2_tool_1", "Account 2 Tool 1"),
            make_tool("acc2_tool_2", "Account 2 Tool 2"),
        ],
        "acc3": [
            make_tool("acc3_tool_1", "Account 3 Tool 1"),
        ],
    }

    def fake_fetch(_: str, headers: dict[str, str]) -> list[_McpToolDefinition]:
        account = headers.get("x-account-id")
        return catalog.get(account, catalog[None])

    monkeypatch.setattr("stackone_ai.toolset._fetch_mcp_tools", fake_fetch)
    return catalog


class TestAccountFiltering:
    """Test account filtering functionality"""

    def test_set_accounts_chaining(self, mock_mcp_catalog):
        """Test that setAccounts() returns self for chaining"""
        toolset = StackOneToolSet(api_key="test_key")
        result = toolset.set_accounts(["acc1", "acc2"])
        assert result is toolset

    def test_fetch_tools_without_account_filtering(self, mock_mcp_catalog):
        """Test fetching tools without account filtering"""
        toolset = StackOneToolSet(api_key="test_key")
        tools = toolset.fetch_tools()
        assert len(tools) == 2
        tool_names = [t.name for t in tools.to_list()]
        assert "default_tool_1" in tool_names
        assert "default_tool_2" in tool_names

    def test_fetch_tools_with_account_ids(self, mock_mcp_catalog):
        """Test fetching tools with specific account IDs"""
        toolset = StackOneToolSet(api_key="test_key")
        tools = toolset.fetch_tools(account_ids=["acc1"])
        assert len(tools) == 2
        tool_names = [t.name for t in tools.to_list()]
        assert "acc1_tool_1" in tool_names
        assert "acc1_tool_2" in tool_names

    def test_fetch_tools_uses_set_accounts(self, mock_mcp_catalog):
        """Test that fetch_tools uses set_accounts when no accountIds provided"""
        toolset = StackOneToolSet(api_key="test_key")
        toolset.set_accounts(["acc1", "acc2"])
        tools = toolset.fetch_tools()
        # acc1 has 2 tools, acc2 has 2 tools, total should be 4
        assert len(tools) == 4
        tool_names = [t.name for t in tools.to_list()]
        assert "acc1_tool_1" in tool_names
        assert "acc1_tool_2" in tool_names
        assert "acc2_tool_1" in tool_names
        assert "acc2_tool_2" in tool_names

    def test_fetch_tools_overrides_set_accounts(self, mock_mcp_catalog):
        """Test that accountIds parameter overrides set_accounts"""
        toolset = StackOneToolSet(api_key="test_key")
        toolset.set_accounts(["acc1", "acc2"])
        tools = toolset.fetch_tools(account_ids=["acc3"])
        # Should fetch tools only for acc3 (ignoring acc1, acc2)
        assert len(tools) == 1
        tool_names = [t.name for t in tools.to_list()]
        assert "acc3_tool_1" in tool_names
        # Verify set_accounts state is preserved
        assert toolset._account_ids == ["acc1", "acc2"]

    def test_fetch_tools_multiple_account_ids(self, mock_mcp_catalog):
        """Test fetching tools for multiple account IDs"""
        toolset = StackOneToolSet(api_key="test_key")
        tools = toolset.fetch_tools(account_ids=["acc1", "acc2", "acc3"])
        # acc1: 2 tools, acc2: 2 tools, acc3: 1 tool = 5 total
        assert len(tools) == 5

    def test_fetch_tools_preserves_account_context(self, monkeypatch):
        """Test that tools preserve their account context"""
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

        toolset = StackOneToolSet(api_key="test_key")
        tools = toolset.fetch_tools(account_ids=["specific-account"])

        tool = tools.get_tool("test_tool")
        assert tool is not None
        assert tool.get_account_id() == "specific-account"
        assert captured_accounts == ["specific-account"]


class TestProviderAndActionFiltering:
    """Test provider and action filtering functionality"""

    @pytest.fixture
    def mixed_tools_catalog(self, monkeypatch):
        """Mock catalog with mixed provider tools"""

        def make_tool(name: str, description: str) -> _McpToolDefinition:
            return _McpToolDefinition(
                name=name,
                description=description,
                input_schema={
                    "type": "object",
                    "properties": {"fields": {"type": "string"}},
                },
            )

        catalog: dict[str | None, list[_McpToolDefinition]] = {
            None: [
                make_tool("hibob_list_employees", "HiBob List Employees"),
                make_tool("hibob_create_employees", "HiBob Create Employees"),
                make_tool("bamboohr_list_employees", "BambooHR List Employees"),
                make_tool("bamboohr_get_employee", "BambooHR Get Employee"),
                make_tool("workday_list_employees", "Workday List Employees"),
            ],
        }

        def fake_fetch(_: str, headers: dict[str, str]) -> list[_McpToolDefinition]:
            account = headers.get("x-account-id")
            return catalog.get(account, catalog[None])

        monkeypatch.setattr("stackone_ai.toolset._fetch_mcp_tools", fake_fetch)
        return catalog

    def test_filter_by_providers(self, mixed_tools_catalog):
        """Test filtering tools by providers"""
        toolset = StackOneToolSet(api_key="test_key")
        tools = toolset.fetch_tools(providers=["hibob", "bamboohr"])
        assert len(tools) == 4
        tool_names = [t.name for t in tools.to_list()]
        assert "hibob_list_employees" in tool_names
        assert "hibob_create_employees" in tool_names
        assert "bamboohr_list_employees" in tool_names
        assert "bamboohr_get_employee" in tool_names
        assert "workday_list_employees" not in tool_names

    def test_filter_by_actions_exact_match(self, mixed_tools_catalog):
        """Test filtering tools by exact action names"""
        toolset = StackOneToolSet(api_key="test_key")
        tools = toolset.fetch_tools(actions=["hibob_list_employees", "hibob_create_employees"])
        assert len(tools) == 2
        tool_names = [t.name for t in tools.to_list()]
        assert "hibob_list_employees" in tool_names
        assert "hibob_create_employees" in tool_names

    def test_filter_by_actions_glob_pattern(self, mixed_tools_catalog):
        """Test filtering tools by glob patterns"""
        toolset = StackOneToolSet(api_key="test_key")
        tools = toolset.fetch_tools(actions=["*_list_employees"])
        assert len(tools) == 3
        tool_names = [t.name for t in tools.to_list()]
        assert "hibob_list_employees" in tool_names
        assert "bamboohr_list_employees" in tool_names
        assert "workday_list_employees" in tool_names
        assert "hibob_create_employees" not in tool_names
        assert "bamboohr_get_employee" not in tool_names

    def test_combine_account_and_action_filters(self, monkeypatch):
        """Test combining account and action filters"""

        def make_tool(name: str, description: str) -> _McpToolDefinition:
            return _McpToolDefinition(
                name=name,
                description=description,
                input_schema={
                    "type": "object",
                    "properties": {"fields": {"type": "string"}},
                },
            )

        catalog: dict[str | None, list[_McpToolDefinition]] = {
            "acc1": [
                make_tool("hibob_list_employees", "HiBob List Employees"),
                make_tool("hibob_create_employees", "HiBob Create Employees"),
            ],
            "acc2": [
                make_tool("bamboohr_list_employees", "BambooHR List Employees"),
                make_tool("bamboohr_get_employee", "BambooHR Get Employee"),
            ],
        }

        def fake_fetch(_: str, headers: dict[str, str]) -> list[_McpToolDefinition]:
            account = headers.get("x-account-id")
            return catalog.get(account, [])

        monkeypatch.setattr("stackone_ai.toolset._fetch_mcp_tools", fake_fetch)

        toolset = StackOneToolSet(api_key="test_key")
        tools = toolset.fetch_tools(account_ids=["acc1", "acc2"], actions=["*_list_employees"])
        assert len(tools) == 2
        tool_names = [t.name for t in tools.to_list()]
        assert "hibob_list_employees" in tool_names
        assert "bamboohr_list_employees" in tool_names
        assert "hibob_create_employees" not in tool_names
        assert "bamboohr_get_employee" not in tool_names

    def test_combine_all_filters(self, monkeypatch):
        """Test combining accountIds, providers, and actions filters"""

        def make_tool(name: str, description: str) -> _McpToolDefinition:
            return _McpToolDefinition(
                name=name,
                description=description,
                input_schema={
                    "type": "object",
                    "properties": {"fields": {"type": "string"}},
                },
            )

        catalog: dict[str | None, list[_McpToolDefinition]] = {
            "acc1": [
                make_tool("hibob_list_employees", "HiBob List Employees"),
                make_tool("hibob_create_employees", "HiBob Create Employees"),
                make_tool("workday_list_employees", "Workday List Employees"),
            ],
        }

        def fake_fetch(_: str, headers: dict[str, str]) -> list[_McpToolDefinition]:
            account = headers.get("x-account-id")
            return catalog.get(account, [])

        monkeypatch.setattr("stackone_ai.toolset._fetch_mcp_tools", fake_fetch)

        toolset = StackOneToolSet(api_key="test_key")
        tools = toolset.fetch_tools(account_ids=["acc1"], providers=["hibob"], actions=["*_list_*"])
        # Should only return hibob_list_employees (matches all filters)
        assert len(tools) == 1
        tool_names = [t.name for t in tools.to_list()]
        assert "hibob_list_employees" in tool_names


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
