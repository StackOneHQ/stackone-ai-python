"""Tests for meta tools functionality"""

import httpx
import pytest
import respx
from hypothesis import given, settings
from hypothesis import strategies as st

from stackone_ai import StackOneTool, Tools
from stackone_ai.meta_tools import (
    ToolIndex,
    create_meta_execute_tool,
    create_meta_search_tools,
)
from stackone_ai.models import ExecuteConfig, ToolParameters

# Hypothesis strategies for PBT
# Score threshold strategy
score_threshold_strategy = st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)

# Hybrid alpha strategy (can be outside [0, 1] to test clamping)
hybrid_alpha_strategy = st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False)

# Limit strategy
limit_strategy = st.integers(min_value=1, max_value=100)


def _create_sample_tools() -> list[StackOneTool]:
    """Helper function to create sample tools (for use in PBT tests)."""
    tools = []

    # Create HiBob tools
    for action in ["create", "list", "update", "delete"]:
        for entity in ["employee", "department", "timeoff"]:
            tool_name = f"hibob_{action}_{entity}"
            execute_config = ExecuteConfig(
                name=tool_name,
                method="POST" if action in ["create", "update"] else "GET",
                url=f"https://api.example.com/hibob/{entity}",
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
                description=f"{action.capitalize()} {entity} in HiBob system",
                parameters=parameters,
                _execute_config=execute_config,
                _api_key="test_key",
            )
            tools.append(tool)

    # Create BambooHR tools
    for action in ["create", "list", "search"]:
        for entity in ["candidate", "job", "application"]:
            tool_name = f"bamboohr_{action}_{entity}"
            execute_config = ExecuteConfig(
                name=tool_name,
                method="POST" if action == "create" else "GET",
                url=f"https://api.example.com/bamboohr/{entity}",
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
                description=f"{action.capitalize()} {entity} in BambooHR system",
                parameters=parameters,
                _execute_config=execute_config,
                _api_key="test_key",
            )
            tools.append(tool)

    return tools


@pytest.fixture
def sample_tools():
    """Create sample tools for testing"""
    tools = []

    # Create HiBob tools
    for action in ["create", "list", "update", "delete"]:
        for entity in ["employee", "department", "timeoff"]:
            tool_name = f"hibob_{action}_{entity}"
            execute_config = ExecuteConfig(
                name=tool_name,
                method="POST" if action in ["create", "update"] else "GET",
                url=f"https://api.example.com/hibob/{entity}",
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
                description=f"{action.capitalize()} {entity} in HiBob system",
                parameters=parameters,
                _execute_config=execute_config,
                _api_key="test_key",
            )
            tools.append(tool)

    # Create BambooHR tools
    for action in ["create", "list", "search"]:
        for entity in ["candidate", "job", "application"]:
            tool_name = f"bamboohr_{action}_{entity}"
            execute_config = ExecuteConfig(
                name=tool_name,
                method="POST" if action == "create" else "GET",
                url=f"https://api.example.com/bamboohr/{entity}",
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
                description=f"{action.capitalize()} {entity} in BambooHR system",
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

    @given(min_score=score_threshold_strategy, limit=limit_strategy)
    @settings(max_examples=50)
    def test_search_with_min_score_pbt(self, min_score: float, limit: int):
        """PBT: Test that min_score filtering always works correctly."""
        # Create tools inside test to avoid fixture issues with Hypothesis
        tools = _create_sample_tools()
        index = ToolIndex(tools)

        results = index.search("employee", limit=limit, min_score=min_score)

        # All results must meet the score threshold
        for r in results:
            assert r.score >= min_score, f"Score {r.score} < min_score {min_score}"

        # Result count should not exceed limit
        assert len(results) <= limit

    @given(limit=limit_strategy)
    @settings(max_examples=50)
    def test_search_limit_pbt(self, limit: int):
        """PBT: Test that limit is always respected."""
        # Create tools inside test to avoid fixture issues with Hypothesis
        tools = _create_sample_tools()
        index = ToolIndex(tools)

        results = index.search("employee", limit=limit)

        assert len(results) <= limit
        assert len(results) <= len(tools)


class TestMetaSearchTool:
    """Test the meta_search_tools functionality"""

    def test_filter_tool_creation(self, sample_tools):
        """Test creating the filter tool"""
        index = ToolIndex(sample_tools)
        filter_tool = create_meta_search_tools(index)

        assert filter_tool.name == "meta_search_tools"
        assert "natural language query" in filter_tool.description.lower()

    def test_filter_tool_execute_with_json_string(self, sample_tools):
        """Test executing the filter tool with JSON string input."""
        import json

        index = ToolIndex(sample_tools)
        filter_tool = create_meta_search_tools(index)

        # Execute with JSON string
        json_input = json.dumps({"query": "employee", "limit": 2, "minScore": 0.0})
        result = filter_tool.execute(json_input)

        assert "tools" in result
        assert isinstance(result["tools"], list)
        assert len(result["tools"]) <= 2

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

    def test_execute_tool_with_json_string(self, tools_collection):
        """Test execute tool with JSON string input."""
        import json

        execute_tool = create_meta_execute_tool(tools_collection)

        # Execute with JSON string - should raise ValueError for invalid tool
        json_input = json.dumps({"toolName": "nonexistent_tool", "params": {}})
        with pytest.raises(ValueError, match="Tool 'nonexistent_tool' not found"):
            execute_tool.execute(json_input)

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

    @respx.mock
    def test_execute_tool_call(self, tools_collection):
        """Test calling the execute tool with call method"""
        execute_tool = create_meta_execute_tool(tools_collection)

        # Mock the actual tool execution
        route = respx.get("https://api.example.com/hibob/employee").mock(
            return_value=httpx.Response(200, json={"success": True, "employees": []})
        )

        # Call the meta execute tool
        result = execute_tool.call(toolName="hibob_list_employee", params={"limit": 10})

        assert result == {"success": True, "employees": []}
        assert route.called
        assert route.calls[0].response.status_code == 200


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


class TestHybridSearch:
    """Test hybrid search functionality"""

    def test_hybrid_alpha_parameter(self, sample_tools):
        """Test that hybrid_alpha parameter is properly set"""
        # Create index with custom alpha
        index = ToolIndex(sample_tools, hybrid_alpha=0.5)
        assert index.hybrid_alpha == 0.5

        # Test boundary values
        index_min = ToolIndex(sample_tools, hybrid_alpha=-0.1)
        assert index_min.hybrid_alpha == 0.0

        index_max = ToolIndex(sample_tools, hybrid_alpha=1.5)
        assert index_max.hybrid_alpha == 1.0

    @given(alpha=hybrid_alpha_strategy)
    @settings(max_examples=100)
    def test_hybrid_alpha_clamping_pbt(self, alpha: float):
        """PBT: Test that hybrid_alpha is always clamped to [0.0, 1.0]."""
        # Create tools inside test to avoid fixture issues with Hypothesis
        tools = _create_sample_tools()
        index = ToolIndex(tools, hybrid_alpha=alpha)

        # Should always be clamped to [0.0, 1.0]
        assert 0.0 <= index.hybrid_alpha <= 1.0

        # Verify clamping logic
        if alpha < 0.0:
            assert index.hybrid_alpha == 0.0
        elif alpha > 1.0:
            assert index.hybrid_alpha == 1.0
        else:
            assert index.hybrid_alpha == alpha

    def test_hybrid_search_returns_results(self, sample_tools):
        """Test that hybrid search returns meaningful results"""
        index = ToolIndex(sample_tools, hybrid_alpha=0.2)
        # Use more specific query to ensure we get employee tools
        results = index.search("employee hibob", limit=10)

        assert len(results) > 0
        # Should find HiBob employee tools - check in broader result set
        result_names = [r.name for r in results]
        # At least one result should contain "employee" or "hibob"
        assert any("employee" in name or "hibob" in name for name in result_names), (
            f"Expected 'employee' or 'hibob' in results: {result_names}"
        )

    def test_hybrid_search_with_different_alphas(self, sample_tools):
        """Test that different alpha values affect ranking"""
        # Create indexes with different alphas
        index_bm25_heavy = ToolIndex(sample_tools, hybrid_alpha=0.9)  # More BM25
        index_tfidf_heavy = ToolIndex(sample_tools, hybrid_alpha=0.1)  # More TF-IDF
        index_balanced = ToolIndex(sample_tools, hybrid_alpha=0.5)  # Balanced

        query = "create new employee record"

        # Get results from each - use larger limit for better coverage
        results_bm25 = index_bm25_heavy.search(query, limit=10)
        results_tfidf = index_tfidf_heavy.search(query, limit=10)
        results_balanced = index_balanced.search(query, limit=10)

        # All should return results
        assert len(results_bm25) > 0
        assert len(results_tfidf) > 0
        assert len(results_balanced) > 0

        # All should have "employee" and "create" tools in results
        assert any("employee" in r.name and "create" in r.name for r in results_bm25), (
            f"BM25 results: {[r.name for r in results_bm25]}"
        )
        assert any("employee" in r.name and "create" in r.name for r in results_tfidf), (
            f"TF-IDF results: {[r.name for r in results_tfidf]}"
        )
        assert any("employee" in r.name and "create" in r.name for r in results_balanced), (
            f"Balanced results: {[r.name for r in results_balanced]}"
        )

    def test_meta_tools_with_custom_alpha(self, sample_tools):
        """Test that meta_tools() accepts hybrid_alpha parameter"""
        tools_collection = Tools(sample_tools)

        # Create meta tools with custom alpha
        meta_tools = tools_collection.meta_tools(hybrid_alpha=0.3)

        filter_tool = meta_tools.get_tool("meta_search_tools")
        assert filter_tool is not None

        # Check that description mentions the alpha value
        assert "alpha=0.3" in filter_tool.description

        # Test it works
        result = filter_tool.execute({"query": "list employees", "limit": 3})
        assert "tools" in result
        assert len(result["tools"]) > 0
