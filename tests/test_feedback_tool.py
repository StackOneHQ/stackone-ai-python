"""Tests for the global feedback tool being inherited from the MCP catalog.

The StackOne MCP server exposes ``submit_feedback`` on every account, so the SDK no longer defines
its own feedback tool — it inherits it from the catalog. These tests patch the MCP fetch so they do
not depend on the vendored node mock server (which only gains ``submit_feedback`` once its submodule
is bumped to the matching release).
"""

from stackone_ai import StackOneToolSet
from stackone_ai import toolset as toolset_module
from stackone_ai.toolset import _McpToolDefinition


def _fake_catalog(endpoint: str, headers: dict[str, str]) -> list[_McpToolDefinition]:
    """A connector tool plus the global submit_feedback tool, mirroring the real MCP catalog."""
    return [
        _McpToolDefinition(
            name="hibob_list_employees",
            description="List employees",
            input_schema={"type": "object", "properties": {}},
        ),
        _McpToolDefinition(
            name="submit_feedback",
            description="Submit feedback",
            input_schema={"type": "object", "properties": {}},
        ),
    ]


class TestFeedbackInheritedFromMcp:
    def test_included_by_default(self, monkeypatch):
        monkeypatch.setattr(toolset_module, "_fetch_mcp_tools", _fake_catalog)
        toolset = StackOneToolSet(api_key="test-key", account_id="acc1")

        tool_names = [tool.name for tool in toolset.fetch_tools().to_list()]

        assert "submit_feedback" in tool_names

    def test_excluded_when_feedback_disabled(self, monkeypatch):
        monkeypatch.setattr(toolset_module, "_fetch_mcp_tools", _fake_catalog)
        toolset = StackOneToolSet(api_key="test-key", account_id="acc1")

        tool_names = [tool.name for tool in toolset.fetch_tools(feedback=False).to_list()]

        assert "submit_feedback" not in tool_names
        assert "hibob_list_employees" in tool_names

    def test_survives_provider_filter(self, monkeypatch):
        monkeypatch.setattr(toolset_module, "_fetch_mcp_tools", _fake_catalog)
        toolset = StackOneToolSet(api_key="test-key", account_id="acc1")

        tool_names = [tool.name for tool in toolset.fetch_tools(providers=["hibob"]).to_list()]

        assert "submit_feedback" in tool_names
        assert "hibob_list_employees" in tool_names

    def test_single_instance_across_accounts(self, monkeypatch):
        monkeypatch.setattr(toolset_module, "_fetch_mcp_tools", _fake_catalog)
        toolset = StackOneToolSet(api_key="test-key")

        tool_names = [tool.name for tool in toolset.fetch_tools(account_ids=["acc1", "acc2"]).to_list()]

        assert tool_names.count("submit_feedback") == 1


class TestFeedbackInSearchAndExecute:
    """search-and-execute agents inherit submit_feedback alongside the meta tools."""

    def test_build_tools_includes_feedback_by_default(self, monkeypatch):
        monkeypatch.setattr(toolset_module, "_fetch_mcp_tools", _fake_catalog)
        toolset = StackOneToolSet(api_key="test-key", account_id="acc1", search={"method": "local"})

        tools = toolset._build_tools(account_ids=["acc1"])
        tool_names = [tool.name for tool in tools.to_list()]

        # the two meta tools + the inherited feedback tool
        assert len(tool_names) == 3
        assert "submit_feedback" in tool_names

    def test_build_tools_excludes_feedback_when_disabled(self, monkeypatch):
        monkeypatch.setattr(toolset_module, "_fetch_mcp_tools", _fake_catalog)
        toolset = StackOneToolSet(api_key="test-key", account_id="acc1", search={"method": "local"})

        tools = toolset._build_tools(account_ids=["acc1"], feedback=False)
        tool_names = [tool.name for tool in tools.to_list()]

        assert "submit_feedback" not in tool_names
        assert len(tool_names) == 2
