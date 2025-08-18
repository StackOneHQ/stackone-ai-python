"""Tests for meta tools functionality"""

import pytest
import responses

from stackone_ai import StackOneTool, Tools
from stackone_ai.meta_tools import (
    ToolIndex,
    create_meta_execute_tool,
    create_meta_search_tools,
)
from stackone_ai.models import ExecuteConfig, ToolParameters


@pytest.fixture
def sample_tools():
    """Create sample tools for testing"""
    tools = []

    # Create HRIS tools
    for action in ["create", "list", "update", "delete"]:
        for entity in ["employee", "department", "timeoff"]:
            tool_name = f"hris_{action}_{entity}"
            execute_config = ExecuteConfig(
                name=tool_name,
                method="POST" if action in ["create", "update"] else "GET",
                url=f"https://api.example.com/hris/{entity}",
                headers={},
            )

            parameters = ToolParameters(
                type="object",
                properties={
                    "id": {"type": "string", "description": "Entity ID"},
                    "data": {"type": "object", "description": "Entity data"},
                },
            )

            tool = StackOneTool(
                description=f"{action.capitalize()} {entity} in HRIS system",
                parameters=parameters,
                _execute_config=execute_config,
                _api_key="test_key",
            )
            tools.append(tool)

    # Create ATS tools
    for action in ["create", "list", "search"]:
        for entity in ["candidate", "job", "application"]:
            tool_name = f"ats_{action}_{entity}"
            execute_config = ExecuteConfig(
                name=tool_name,
                method="POST" if action == "create" else "GET",
                url=f"https://api.example.com/ats/{entity}",
                headers={},
            )

            parameters = ToolParameters(
                type="object",
                properties={
                    "query": {"type": "string", "description": "Search query"},
                    "filters": {"type": "object", "description": "Filter criteria"},
                },
            )

            tool = StackOneTool(
                description=f"{action.capitalize()} {entity} in ATS system",
                parameters=parameters,
                _execute_config=execute_config,
                _api_key="test_key",
            )
            tools.append(tool)

    return tools


@pytest.fixture
def tools_collection(sample_tools):
    """Create a Tools collection from sample tools"""
    return Tools(sample_tools)


class TestToolIndex:
    """Test the BM25 tool search index"""

    def test_index_creation(self, sample_tools):
        """Test creating a tool index"""
        index = ToolIndex(sample_tools)
        assert len(index.tools) == len(sample_tools)
        assert len(index.tool_map) == len(sample_tools)

    def test_search_basic(self, sample_tools):
        """Test basic search functionality"""
        index = ToolIndex(sample_tools)

        # Search for employee-related tools
        results = index.search("employee", limit=5)

        assert len(results) > 0
        # Check that at least one result contains "employee"
        assert any("employee" in r.name for r in results)

    def test_search_with_action(self, sample_tools):
        """Test searching with action keywords"""
        index = ToolIndex(sample_tools)

        # Search for create operations
        results = index.search("create new", limit=5)

        assert len(results) > 0
        # Most results should contain "create"
        create_tools = [r for r in results if "create" in r.name]
        assert len(create_tools) > 0

    def test_search_with_min_score(self, sample_tools):
        """Test filtering by minimum score"""
        index = ToolIndex(sample_tools)

        # Search with a high min_score
        results = index.search("employee", limit=10, min_score=0.5)

        # All results should have score >= 0.5
        assert all(r.score >= 0.5 for r in results)

    def test_search_limit(self, sample_tools):
        """Test limiting search results"""
        index = ToolIndex(sample_tools)

        # Search with limit
        results = index.search("", limit=3)

        assert len(results) <= 3


class TestMetaSearchTool:
    """Test the meta_search_tools functionality"""

    def test_filter_tool_creation(self, sample_tools):
        """Test creating the filter tool"""
        index = ToolIndex(sample_tools)
        filter_tool = create_meta_search_tools(index)

        assert filter_tool.name == "meta_search_tools"
        assert "natural language query" in filter_tool.description.lower()

    def test_filter_tool_execute(self, sample_tools):
        """Test executing the filter tool"""
        index = ToolIndex(sample_tools)
        filter_tool = create_meta_search_tools(index)

        # Execute with a query
        result = filter_tool.execute(
            {
                "query": "manage employees",
                "limit": 3,
                "minScore": 0.0,
            }
        )

        assert "tools" in result
        assert isinstance(result["tools"], list)
        assert len(result["tools"]) <= 3

        # Check tool structure
        if result["tools"]:
            tool = result["tools"][0]
            assert "name" in tool
            assert "description" in tool
            assert "score" in tool

    def test_filter_tool_call(self, sample_tools):
        """Test calling the filter tool with call method"""
        index = ToolIndex(sample_tools)
        filter_tool = create_meta_search_tools(index)

        # Call with kwargs
        result = filter_tool.call(query="candidate", limit=2)

        assert "tools" in result
        assert len(result["tools"]) <= 2


class TestMetaExecuteTool:
    """Test the meta_execute_tool functionality"""

    def test_execute_tool_creation(self, tools_collection):
        """Test creating the execute tool"""
        execute_tool = create_meta_execute_tool(tools_collection)

        assert execute_tool.name == "meta_execute_tool"
        assert "executes a tool" in execute_tool.description.lower()

    def test_execute_tool_missing_name(self, tools_collection):
        """Test execute tool with missing tool name"""
        execute_tool = create_meta_execute_tool(tools_collection)

        with pytest.raises(ValueError, match="toolName is required"):
            execute_tool.execute({"params": {}})

    def test_execute_tool_invalid_name(self, tools_collection):
        """Test execute tool with invalid tool name"""
        execute_tool = create_meta_execute_tool(tools_collection)

        with pytest.raises(ValueError, match="Tool 'invalid_tool' not found"):
            execute_tool.execute(
                {
                    "toolName": "invalid_tool",
                    "params": {},
                }
            )

    def test_execute_tool_call(self, tools_collection):
        """Test calling the execute tool with call method"""
        execute_tool = create_meta_execute_tool(tools_collection)

        # Mock the actual tool execution by patching the requests
        with responses.RequestsMock() as rsps:
            rsps.add(
                responses.GET,
                "https://api.example.com/hris/employee",
                json={"success": True, "employees": []},
                status=200,
            )

            # Call the meta execute tool
            result = execute_tool.call(toolName="hris_list_employee", params={"limit": 10})

            assert result is not None
            assert "success" in result or "employees" in result


class TestToolsMetaTools:
    """Test the meta_tools method on Tools collection"""

    def test_meta_tools_creation(self, tools_collection):
        """Test creating meta tools from a Tools collection"""
        meta_tools = tools_collection.meta_tools()

        assert isinstance(meta_tools, Tools)
        assert len(meta_tools) == 2

        # Check tool names
        tool_names = [tool.name for tool in meta_tools.tools]
        assert "meta_search_tools" in tool_names
        assert "meta_execute_tool" in tool_names

    def test_meta_tools_functionality(self, tools_collection):
        """Test that meta tools work correctly"""
        meta_tools = tools_collection.meta_tools()

        # Get the filter tool
        filter_tool = meta_tools.get_tool("meta_search_tools")
        assert filter_tool is not None

        # Search for tools
        result = filter_tool.execute(
            {
                "query": "create employee",
                "limit": 1,
            }
        )

        assert "tools" in result
        assert len(result["tools"]) > 0

        # The top result should be related to creating employees
        top_tool = result["tools"][0]
        assert "employee" in top_tool["name"].lower() or "create" in top_tool["name"].lower()
