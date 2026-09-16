"""Tests for StackOneToolSet MCP functionality using real MCP mock server."""

from __future__ import annotations

from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from stackone_ai.tools import McpToolDefinition, fetch_mcp_tools
from stackone_ai.toolset import StackOneToolSet


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
        from stackone_ai.tools import McpToolDefinition

        def fake_fetch(_: str, headers: dict[str, str]) -> list[McpToolDefinition]:
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

        def fake_fetch(_: str, headers: dict[str, str]) -> list[McpToolDefinition]:
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

        def fake_fetch(_: str, headers: dict[str, str]) -> list[McpToolDefinition]:
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
        sample_tool = McpToolDefinition(
            name="test_tool",
            description="Test tool",
            input_schema={"type": "object", "properties": {}},
        )

        captured_accounts: list[str | None] = []

        def fake_fetch(_: str, headers: dict[str, str]) -> list[McpToolDefinition]:
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

        def fake_fetch(_: str, headers: dict[str, str]) -> list[McpToolDefinition]:
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
            result = fetch_mcp_tools("https://api.example.com/mcp", {"Authorization": "Basic test"})

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

        def fake_fetch(_endpoint: str, _headers: dict[str, str]) -> list[McpToolDefinition]:
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

        def fake_fetch(_endpoint: str, headers: dict[str, str]) -> list[McpToolDefinition]:
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

    def test_provider_and_action_filters_participate_in_cache_key(self, monkeypatch):
        calls = {"count": 0}

        def fake_fetch(_endpoint: str, _headers: dict[str, str]) -> list[McpToolDefinition]:
            calls["count"] += 1
            return [
                McpToolDefinition(name="prov_list_a", description="", input_schema={}),
                McpToolDefinition(name="other_get_b", description="", input_schema={}),
            ]

        monkeypatch.setattr("stackone_ai.toolset.fetch_mcp_tools", fake_fetch)

        toolset = StackOneToolSet(api_key="test-key", account_id="acc1")
        toolset.fetch_tools()
        toolset.fetch_tools(providers=["prov"])
        toolset.fetch_tools(providers=["prov"])
        toolset.fetch_tools(actions=["*_list_*"])

        assert calls["count"] == 3

    def test_clear_catalog_cache_forces_refetch(self, monkeypatch):
        calls = {"count": 0}

        def fake_fetch(_endpoint: str, _headers: dict[str, str]) -> list[McpToolDefinition]:
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

        def fake_fetch(_endpoint: str, _headers: dict[str, str]) -> list[McpToolDefinition]:
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

        def fake_fetch(_endpoint: str, headers: dict[str, str]) -> list[McpToolDefinition]:
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

        def slow_fetch(_endpoint: str, _headers: dict[str, str]) -> list[McpToolDefinition]:
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
        def fake_fetch(_endpoint: str, headers: dict[str, str]) -> list[McpToolDefinition]:
            acc = headers.get("x-account-id", "none")
            return [McpToolDefinition(name=f"tool_{acc}", description="", input_schema={})]

        monkeypatch.setattr("stackone_ai.toolset.fetch_mcp_tools", fake_fetch)

        toolset = StackOneToolSet(api_key="test-key", account_id="acc1")
        tools = toolset.fetch_tools(account_ids=["a", "b", "c", "d"])
        names = {t.name for t in tools.to_list()}
        assert names == {"tool_a", "tool_b", "tool_c", "tool_d"}

    def test_healthy_accounts_survive_one_failure(self, monkeypatch, caplog):
        """One unusable account must not cost the caller every other account's tools."""

        def flaky_fetch(_endpoint: str, headers: dict[str, str]) -> list[McpToolDefinition]:
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

        def always_fails(_endpoint: str, headers: dict[str, str]) -> list[McpToolDefinition]:
            raise RuntimeError(f"boom for {headers.get('x-account-id')}")

        monkeypatch.setattr("stackone_ai.toolset.fetch_mcp_tools", always_fails)

        toolset = StackOneToolSet(api_key="test-key", account_id="acc1")
        with pytest.raises(ToolsetLoadError, match="No account returned tools"):
            toolset.fetch_tools(account_ids=["a", "b"])


class TestMcpParamStylePinning:
    """The /mcp listing URL must pin param-style so the schema matches the RPC unwrap."""

    def test_fetch_tools_pins_flat_prefixed_param_style(self, monkeypatch):
        captured: dict[str, str] = {}

        def fake_fetch(endpoint: str, headers: dict[str, str]) -> list[McpToolDefinition]:
            captured["endpoint"] = endpoint
            return []

        monkeypatch.setattr("stackone_ai.toolset.fetch_mcp_tools", fake_fetch)

        toolset = StackOneToolSet(api_key="test-key", base_url="https://api.example.com")
        toolset.fetch_tools(account_ids=["acc1"])

        assert captured["endpoint"] == "https://api.example.com/mcp?param-style=flat_prefixed"


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

        err = _describe_mcp_failure(grouped, "https://api.example.com/mcp")

        assert "412" in str(err)
        assert "TaskGroup" not in str(err)
        assert getattr(err, "status_code", None) == 412

    def test_non_http_error_reports_the_leaf_not_the_group(self):
        from stackone_ai.tools import _describe_mcp_failure

        grouped = ExceptionGroup("unhandled errors in a TaskGroup", [ConnectionError("no route")])

        err = _describe_mcp_failure(grouped, "https://api.example.com/mcp")

        assert "no route" in str(err)
        assert "TaskGroup" not in str(err)


class TestToolModeRouting:
    """search_execute tools are MCP-native: they must not be sent to /actions/rpc."""

    def test_default_mode_builds_rpc_tools(self, monkeypatch):
        from stackone_ai.tools import StackOneRpcTool

        monkeypatch.setattr(
            "stackone_ai.toolset.fetch_mcp_tools",
            lambda _e, _h: [McpToolDefinition(name="t", description="", input_schema={})],
        )
        toolset = StackOneToolSet(api_key="k", account_id="acc1")
        assert isinstance(toolset.fetch_tools().get_tool("t"), StackOneRpcTool)

    def test_search_execute_mode_builds_mcp_tools(self, monkeypatch):
        from stackone_ai.tools import StackOneMcpTool

        monkeypatch.setattr(
            "stackone_ai.toolset.fetch_mcp_tools",
            lambda _e, _h: [McpToolDefinition(name="t", description="", input_schema={})],
        )
        toolset = StackOneToolSet(api_key="k", account_id="acc1", tool_mode="search_execute")
        assert isinstance(toolset.fetch_tools().get_tool("t"), StackOneMcpTool)

    def test_search_execute_mode_is_requested_on_the_url(self, monkeypatch):
        seen: list[str] = []

        def capture(endpoint: str, _headers: dict[str, str]) -> list[McpToolDefinition]:
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


class TestSearchAndExecuteApi:
    """The recommended surface: search() then execute(), no account id needed."""

    def test_search_returns_actions(self, mcp_mock_server: str):
        toolset = StackOneToolSet(api_key="test-key", base_url=mcp_mock_server)
        actions = toolset.search("list items")
        assert [a["action_id"] for a in actions] == ["mock_list_items"]

    def test_execute_runs_a_searched_action(self, mcp_mock_server: str):
        toolset = StackOneToolSet(api_key="test-key", base_url=mcp_mock_server)
        action_id = toolset.search("list items")[0]["action_id"]
        assert toolset.execute(action_id)["data"] == {"nodes": []}

    def test_execute_passes_the_nested_envelope_through(self, mcp_mock_server: str):
        """execute() takes the envelope an action's example_request shows, verbatim.

        Not the flat `query_pageSize` form: that belongs to fetch_tools() tools, whose
        own served schema names the keys. One id must not mean two argument shapes.
        """
        toolset = StackOneToolSet(api_key="test-key", base_url=mcp_mock_server)
        action = toolset.search("list items")[0]
        assert action["example_request"] == {"query": {"page_size": 25}}

        result = toolset.execute(action["action_id"], action["example_request"])
        assert result["echoed_query"] == {"page_size": 25}

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
