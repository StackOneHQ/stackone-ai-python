"""Tests for semantic search client and integration."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import httpx
import pytest

from stackone_ai.semantic_search import (
    SemanticSearchClient,
    SemanticSearchError,
    SemanticSearchResponse,
    SemanticSearchResult,
)


class TestSemanticSearchResult:
    """Tests for SemanticSearchResult model."""

    def test_create_result(self) -> None:
        """Test creating a search result."""
        result = SemanticSearchResult(
            action_name="bamboohr_create_employee",
            connector_key="bamboohr",
            similarity_score=0.92,
            label="Create Employee",
            description="Creates a new employee in BambooHR",
        )

        assert result.action_name == "bamboohr_create_employee"
        assert result.connector_key == "bamboohr"
        assert result.similarity_score == 0.92
        assert result.label == "Create Employee"
        assert result.description == "Creates a new employee in BambooHR"


class TestSemanticSearchResponse:
    """Tests for SemanticSearchResponse model."""

    def test_create_response(self) -> None:
        """Test creating a search response."""
        results = [
            SemanticSearchResult(
                action_name="bamboohr_create_employee",
                connector_key="bamboohr",
                similarity_score=0.92,
                label="Create Employee",
                description="Creates a new employee",
            ),
            SemanticSearchResult(
                action_name="hibob_create_employee",
                connector_key="hibob",
                similarity_score=0.85,
                label="Create Employee",
                description="Creates a new employee",
            ),
        ]
        response = SemanticSearchResponse(
            results=results,
            total_count=2,
            query="create employee",
        )

        assert len(response.results) == 2
        assert response.total_count == 2
        assert response.query == "create employee"


class TestSemanticSearchClient:
    """Tests for SemanticSearchClient."""

    def test_init(self) -> None:
        """Test client initialization."""
        client = SemanticSearchClient(api_key="test-key")

        assert client.api_key == "test-key"
        assert client.base_url == "https://api.stackone.com"
        assert client.timeout == 30.0

    def test_init_custom_base_url(self) -> None:
        """Test client initialization with custom base URL."""
        client = SemanticSearchClient(
            api_key="test-key",
            base_url="https://custom.api.com/",
        )

        assert client.base_url == "https://custom.api.com"  # Trailing slash stripped

    def test_build_auth_header(self) -> None:
        """Test building the authorization header."""
        client = SemanticSearchClient(api_key="test-key")
        header = client._build_auth_header()

        # test-key: encoded in base64 = dGVzdC1rZXk6
        assert header == "Basic dGVzdC1rZXk6"

    @patch("httpx.post")
    def test_search_success(self, mock_post: MagicMock) -> None:
        """Test successful search request."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "results": [
                {
                    "action_name": "bamboohr_create_employee",
                    "connector_key": "bamboohr",
                    "similarity_score": 0.92,
                    "label": "Create Employee",
                    "description": "Creates a new employee",
                }
            ],
            "total_count": 1,
            "query": "create employee",
        }
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        client = SemanticSearchClient(api_key="test-key")
        response = client.search("create employee", top_k=5)

        assert len(response.results) == 1
        assert response.results[0].action_name == "bamboohr_create_employee"
        assert response.total_count == 1
        assert response.query == "create employee"

        # Verify request was made correctly
        mock_post.assert_called_once()
        call_kwargs = mock_post.call_args
        assert call_kwargs.kwargs["json"] == {"query": "create employee", "top_k": 5}
        assert "Authorization" in call_kwargs.kwargs["headers"]

    @patch("httpx.post")
    def test_search_with_connector(self, mock_post: MagicMock) -> None:
        """Test search with connector filter."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "results": [],
            "total_count": 0,
            "query": "create employee",
        }
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        client = SemanticSearchClient(api_key="test-key")
        client.search("create employee", connector="bamboohr", top_k=10)

        call_kwargs = mock_post.call_args
        assert call_kwargs.kwargs["json"] == {
            "query": "create employee",
            "connector": "bamboohr",
            "top_k": 10,
        }

    @patch("httpx.post")
    def test_search_http_error(self, mock_post: MagicMock) -> None:
        """Test search with HTTP error."""
        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.text = "Unauthorized"
        mock_post.return_value = mock_response
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Unauthorized",
            request=MagicMock(),
            response=mock_response,
        )

        client = SemanticSearchClient(api_key="invalid-key")

        with pytest.raises(SemanticSearchError) as exc_info:
            client.search("create employee")

        assert "API error: 401" in str(exc_info.value)

    @patch("httpx.post")
    def test_search_request_error(self, mock_post: MagicMock) -> None:
        """Test search with request error."""
        mock_post.side_effect = httpx.RequestError("Connection failed")

        client = SemanticSearchClient(api_key="test-key")

        with pytest.raises(SemanticSearchError) as exc_info:
            client.search("create employee")

        assert "Request failed" in str(exc_info.value)

    @patch("httpx.post")
    def test_search_action_names(self, mock_post: MagicMock) -> None:
        """Test search_action_names convenience method."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "results": [
                {
                    "action_name": "bamboohr_create_employee",
                    "connector_key": "bamboohr",
                    "similarity_score": 0.92,
                    "label": "Create Employee",
                    "description": "Creates a new employee",
                },
                {
                    "action_name": "hibob_create_employee",
                    "connector_key": "hibob",
                    "similarity_score": 0.45,
                    "label": "Create Employee",
                    "description": "Creates a new employee",
                },
            ],
            "total_count": 2,
            "query": "create employee",
        }
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        client = SemanticSearchClient(api_key="test-key")

        # Without min_score filter
        names = client.search_action_names("create employee")
        assert len(names) == 2
        assert "bamboohr_create_employee" in names
        assert "hibob_create_employee" in names

        # With min_score filter
        names = client.search_action_names("create employee", min_score=0.5)
        assert len(names) == 1
        assert "bamboohr_create_employee" in names


class TestSemanticSearchIntegration:
    """Integration tests for semantic search with toolset."""

    def test_toolset_semantic_client_lazy_init(self) -> None:
        """Test that semantic_client is lazily initialized."""
        from stackone_ai import StackOneToolSet

        toolset = StackOneToolSet(api_key="test-key")

        # Access semantic_client
        client = toolset.semantic_client
        assert isinstance(client, SemanticSearchClient)
        assert client.api_key == "test-key"

        # Same instance on second access
        assert toolset.semantic_client is client

    @patch.object(SemanticSearchClient, "search")
    @patch("stackone_ai.toolset._fetch_mcp_tools")
    def test_toolset_search_tools(
        self,
        mock_fetch: MagicMock,
        mock_search: MagicMock,
    ) -> None:
        """Test toolset.search_tools() method with connector filtering."""
        from stackone_ai import StackOneToolSet
        from stackone_ai.toolset import _McpToolDefinition

        # Mock semantic search to return results (including some for unavailable connectors)
        mock_search.return_value = SemanticSearchResponse(
            results=[
                SemanticSearchResult(
                    action_name="bamboohr_create_employee",
                    connector_key="bamboohr",
                    similarity_score=0.95,
                    label="Create Employee",
                    description="Creates a new employee",
                ),
                SemanticSearchResult(
                    action_name="workday_create_worker",
                    connector_key="workday",  # User doesn't have this connector
                    similarity_score=0.90,
                    label="Create Worker",
                    description="Creates a new worker",
                ),
                SemanticSearchResult(
                    action_name="hibob_create_employee",
                    connector_key="hibob",
                    similarity_score=0.85,
                    label="Create Employee",
                    description="Creates a new employee",
                ),
            ],
            total_count=3,
            query="create employee",
        )

        # Mock MCP fetch to return only bamboohr and hibob tools (user's linked accounts)
        mock_fetch.return_value = [
            _McpToolDefinition(
                name="bamboohr_create_employee",
                description="Creates a new employee",
                input_schema={"type": "object", "properties": {}},
            ),
            _McpToolDefinition(
                name="hibob_create_employee",
                description="Creates a new employee",
                input_schema={"type": "object", "properties": {}},
            ),
            _McpToolDefinition(
                name="bamboohr_list_employees",
                description="Lists employees",
                input_schema={"type": "object", "properties": {}},
            ),
        ]

        toolset = StackOneToolSet(api_key="test-key")
        tools = toolset.search_tools("create employee", top_k=5)

        # Should only return tools for available connectors (bamboohr, hibob)
        # workday_create_worker should be filtered out
        assert len(tools) == 2
        tool_names = [t.name for t in tools]
        assert "bamboohr_create_employee" in tool_names
        assert "hibob_create_employee" in tool_names
        assert "workday_create_worker" not in tool_names  # Filtered out - connector not available

        # Results should be sorted by semantic score
        assert tools[0].name == "bamboohr_create_employee"  # score 0.95
        assert tools[1].name == "hibob_create_employee"  # score 0.85

    @patch.object(SemanticSearchClient, "search")
    @patch("stackone_ai.toolset._fetch_mcp_tools")
    def test_toolset_search_tools_fallback(
        self,
        mock_fetch: MagicMock,
        mock_search: MagicMock,
    ) -> None:
        """Test search_tools() fallback when semantic search fails."""
        from stackone_ai import StackOneToolSet
        from stackone_ai.toolset import _McpToolDefinition

        # Semantic search raises an error to trigger fallback
        mock_search.side_effect = SemanticSearchError("API unavailable")

        # Mock MCP fetch to return tools from multiple connectors
        mock_fetch.return_value = [
            _McpToolDefinition(
                name="bamboohr_create_employee",
                description="Creates a new employee in BambooHR",
                input_schema={"type": "object", "properties": {}},
            ),
            _McpToolDefinition(
                name="bamboohr_list_employees",
                description="Lists all employees in BambooHR",
                input_schema={"type": "object", "properties": {}},
            ),
            _McpToolDefinition(
                name="workday_create_worker",
                description="Creates a new worker in Workday",
                input_schema={"type": "object", "properties": {}},
            ),
        ]

        toolset = StackOneToolSet(api_key="test-key")
        tools = toolset.search_tools("create employee", top_k=5, fallback_to_local=True)

        # Should return results from the local BM25+TF-IDF fallback
        assert len(tools) > 0
        tool_names = [t.name for t in tools]
        # Should only include tools for available connectors (bamboohr, workday)
        for name in tool_names:
            connector = name.split("_")[0]
            assert connector in {"bamboohr", "workday"}

    @patch.object(SemanticSearchClient, "search")
    @patch("stackone_ai.toolset._fetch_mcp_tools")
    def test_toolset_search_tools_fallback_disabled(
        self,
        mock_fetch: MagicMock,
        mock_search: MagicMock,
    ) -> None:
        """Test search_tools() raises when fallback is disabled."""
        from stackone_ai import StackOneToolSet
        from stackone_ai.toolset import _McpToolDefinition

        mock_search.side_effect = SemanticSearchError("API unavailable")
        # Must provide tools so the flow reaches the semantic search call
        mock_fetch.return_value = [
            _McpToolDefinition(
                name="bamboohr_create_employee",
                description="Creates a new employee",
                input_schema={"type": "object", "properties": {}},
            ),
        ]

        toolset = StackOneToolSet(api_key="test-key")
        with pytest.raises(SemanticSearchError):
            toolset.search_tools("create employee", fallback_to_local=False)

    @patch.object(SemanticSearchClient, "search")
    @patch("stackone_ai.toolset._fetch_mcp_tools")
    def test_toolset_search_action_names(
        self,
        mock_fetch: MagicMock,
        mock_search: MagicMock,
    ) -> None:
        """Test toolset.search_action_names() method."""
        from stackone_ai import StackOneToolSet

        mock_search.return_value = SemanticSearchResponse(
            results=[
                SemanticSearchResult(
                    action_name="bamboohr_create_employee",
                    connector_key="bamboohr",
                    similarity_score=0.92,
                    label="Create Employee",
                    description="Creates a new employee",
                ),
                SemanticSearchResult(
                    action_name="hibob_create_employee",
                    connector_key="hibob",
                    similarity_score=0.45,
                    label="Create Employee",
                    description="Creates a new employee",
                ),
            ],
            total_count=2,
            query="create employee",
        )

        toolset = StackOneToolSet(api_key="test-key")
        results = toolset.search_action_names("create employee", min_score=0.5)

        # Should filter by min_score
        assert len(results) == 1
        assert results[0].action_name == "bamboohr_create_employee"

    def test_utility_tools_semantic_search(self) -> None:
        """Test utility_tools with semantic search."""
        from stackone_ai.models import StackOneTool, Tools

        # Create a mock tools collection
        tool = MagicMock(spec=StackOneTool)
        tool.name = "test_tool"
        tool.description = "Test tool"
        tools = Tools([tool])

        # Without semantic search - should use local search
        # Patch ToolIndex in utility_tools module where it's imported
        with (
            patch("stackone_ai.utility_tools.ToolIndex"),
            patch("stackone_ai.utility_tools.create_tool_search") as mock_create_search,
            patch("stackone_ai.utility_tools.create_tool_execute") as mock_create_execute,
        ):
            mock_search_tool = MagicMock(spec=StackOneTool)
            mock_search_tool.name = "tool_search"
            mock_execute_tool = MagicMock(spec=StackOneTool)
            mock_execute_tool.name = "tool_execute"
            mock_create_search.return_value = mock_search_tool
            mock_create_execute.return_value = mock_execute_tool
            utility = tools.utility_tools()
            assert len(utility) == 2  # tool_search + tool_execute

        # With semantic search - presence of semantic_client enables it
        mock_client = MagicMock(spec=SemanticSearchClient)
        with (
            patch("stackone_ai.utility_tools.create_semantic_tool_search") as mock_create,
            patch("stackone_ai.utility_tools.create_tool_execute") as mock_create_execute,
        ):
            mock_search_tool = MagicMock(spec=StackOneTool)
            mock_search_tool.name = "tool_search"
            mock_execute_tool = MagicMock(spec=StackOneTool)
            mock_execute_tool.name = "tool_execute"
            mock_create.return_value = mock_search_tool
            mock_create_execute.return_value = mock_execute_tool
            utility = tools.utility_tools(semantic_client=mock_client)
            assert len(utility) == 2
            mock_create.assert_called_once_with(mock_client)


class TestSemanticToolSearch:
    """Tests for create_semantic_tool_search utility."""

    def test_create_semantic_tool_search_type_error(self) -> None:
        """Test that invalid client raises TypeError."""
        from stackone_ai.utility_tools import create_semantic_tool_search

        with pytest.raises(TypeError) as exc_info:
            create_semantic_tool_search("not a client")  # type: ignore

        assert "SemanticSearchClient instance" in str(exc_info.value)

    @patch.object(SemanticSearchClient, "search")
    def test_semantic_tool_search_execute(self, mock_search: MagicMock) -> None:
        """Test executing semantic tool search."""
        from stackone_ai.utility_tools import create_semantic_tool_search

        mock_search.return_value = SemanticSearchResponse(
            results=[
                SemanticSearchResult(
                    action_name="bamboohr_create_employee",
                    connector_key="bamboohr",
                    similarity_score=0.92,
                    label="Create Employee",
                    description="Creates a new employee",
                ),
            ],
            total_count=1,
            query="create employee",
        )

        client = SemanticSearchClient(api_key="test-key")
        tool = create_semantic_tool_search(client)

        result = tool.execute({"query": "create employee", "limit": 5})

        assert "tools" in result
        assert len(result["tools"]) == 1
        assert result["tools"][0]["name"] == "bamboohr_create_employee"
        assert result["tools"][0]["score"] == 0.92
        assert result["tools"][0]["connector"] == "bamboohr"

    @patch.object(SemanticSearchClient, "search")
    def test_semantic_tool_search_with_min_score(self, mock_search: MagicMock) -> None:
        """Test semantic tool search with min_score filter."""
        from stackone_ai.utility_tools import create_semantic_tool_search

        mock_search.return_value = SemanticSearchResponse(
            results=[
                SemanticSearchResult(
                    action_name="high_score_action",
                    connector_key="test",
                    similarity_score=0.9,
                    label="High Score",
                    description="High scoring action",
                ),
                SemanticSearchResult(
                    action_name="low_score_action",
                    connector_key="test",
                    similarity_score=0.3,
                    label="Low Score",
                    description="Low scoring action",
                ),
            ],
            total_count=2,
            query="test",
        )

        client = SemanticSearchClient(api_key="test-key")
        tool = create_semantic_tool_search(client)

        result = tool.execute({"query": "test", "limit": 10, "minScore": 0.5})

        assert len(result["tools"]) == 1
        assert result["tools"][0]["name"] == "high_score_action"

    @patch.object(SemanticSearchClient, "search")
    def test_semantic_tool_search_with_connector(self, mock_search: MagicMock) -> None:
        """Test semantic tool search with connector filter."""
        from stackone_ai.utility_tools import create_semantic_tool_search

        mock_search.return_value = SemanticSearchResponse(
            results=[],
            total_count=0,
            query="create employee",
        )

        client = SemanticSearchClient(api_key="test-key")
        tool = create_semantic_tool_search(client)

        tool.execute({"query": "create employee", "connector": "bamboohr"})

        mock_search.assert_called_once_with(
            query="create employee",
            connector="bamboohr",
            top_k=5,  # default limit
        )

    def test_semantic_tool_search_has_correct_parameters(self) -> None:
        """Test that semantic tool has the expected parameter schema."""
        from stackone_ai.utility_tools import create_semantic_tool_search

        client = SemanticSearchClient(api_key="test-key")
        tool = create_semantic_tool_search(client)

        assert tool.name == "tool_search"
        assert "semantic" in tool.description.lower()

        props = tool.parameters.properties
        assert "query" in props
        assert "limit" in props
        assert "minScore" in props
        assert "connector" in props


class TestConnectorProperty:
    """Tests for StackOneTool.connector property."""

    def test_connector_extracts_from_name(self) -> None:
        """Test that connector is extracted from tool name."""
        from stackone_ai.models import ExecuteConfig, StackOneTool, ToolParameters

        execute_config = ExecuteConfig(
            name="bamboohr_create_employee",
            method="POST",
            url="https://api.example.com",
            headers={},
        )
        tool = StackOneTool(
            description="Creates employee",
            parameters=ToolParameters(type="object", properties={}),
            _execute_config=execute_config,
            _api_key="test-key",
        )

        assert tool.connector == "bamboohr"

    def test_connector_is_lowercase(self) -> None:
        """Test that connector is always lowercase."""
        from stackone_ai.models import ExecuteConfig, StackOneTool, ToolParameters

        execute_config = ExecuteConfig(
            name="BambooHR_Create_Employee",
            method="POST",
            url="https://api.example.com",
            headers={},
        )
        tool = StackOneTool(
            description="Creates employee",
            parameters=ToolParameters(type="object", properties={}),
            _execute_config=execute_config,
            _api_key="test-key",
        )

        assert tool.connector == "bamboohr"

    def test_connector_with_single_word_name(self) -> None:
        """Test connector extraction with single-word tool name."""
        from stackone_ai.models import ExecuteConfig, StackOneTool, ToolParameters

        execute_config = ExecuteConfig(
            name="utility",
            method="POST",
            url="https://api.example.com",
            headers={},
        )
        tool = StackOneTool(
            description="Utility tool",
            parameters=ToolParameters(type="object", properties={}),
            _execute_config=execute_config,
            _api_key="test-key",
        )

        assert tool.connector == "utility"


class TestToolsConnectorHelpers:
    """Tests for Tools.get_connectors() and filter_by_connector()."""

    def test_get_connectors(self) -> None:
        """Test getting unique connectors from tools collection."""
        from stackone_ai.models import ExecuteConfig, StackOneTool, ToolParameters, Tools

        def make_tool(name: str) -> StackOneTool:
            return StackOneTool(
                description=f"Tool {name}",
                parameters=ToolParameters(type="object", properties={}),
                _execute_config=ExecuteConfig(name=name, method="POST", url="", headers={}),
                _api_key="test-key",
            )

        tools = Tools(
            [
                make_tool("bamboohr_create_employee"),
                make_tool("bamboohr_list_employees"),
                make_tool("hibob_create_employee"),
                make_tool("slack_send_message"),
            ]
        )

        connectors = tools.get_connectors()

        assert connectors == {"bamboohr", "hibob", "slack"}

    def test_get_connectors_empty(self) -> None:
        """Test get_connectors with empty tools collection."""
        from stackone_ai.models import Tools

        tools = Tools([])
        assert tools.get_connectors() == set()

    def test_filter_by_connector(self) -> None:
        """Test filtering tools by connector."""
        from stackone_ai.models import ExecuteConfig, StackOneTool, ToolParameters, Tools

        def make_tool(name: str) -> StackOneTool:
            return StackOneTool(
                description=f"Tool {name}",
                parameters=ToolParameters(type="object", properties={}),
                _execute_config=ExecuteConfig(name=name, method="POST", url="", headers={}),
                _api_key="test-key",
            )

        tools = Tools(
            [
                make_tool("bamboohr_create_employee"),
                make_tool("bamboohr_list_employees"),
                make_tool("hibob_create_employee"),
                make_tool("slack_send_message"),
            ]
        )

        # Filter by single connector
        bamboo_tools = tools.filter_by_connector(["bamboohr"])
        assert len(bamboo_tools) == 2
        assert all(t.connector == "bamboohr" for t in bamboo_tools)

        # Filter by multiple connectors
        hr_tools = tools.filter_by_connector(["bamboohr", "hibob"])
        assert len(hr_tools) == 3
        assert all(t.connector in {"bamboohr", "hibob"} for t in hr_tools)

    def test_filter_by_connector_case_insensitive(self) -> None:
        """Test that filter_by_connector is case-insensitive."""
        from stackone_ai.models import ExecuteConfig, StackOneTool, ToolParameters, Tools

        tool = StackOneTool(
            description="Creates employee",
            parameters=ToolParameters(type="object", properties={}),
            _execute_config=ExecuteConfig(name="bamboohr_create_employee", method="POST", url="", headers={}),
            _api_key="test-key",
        )
        tools = Tools([tool])

        # Should match regardless of case
        assert len(tools.filter_by_connector(["BambooHR"])) == 1
        assert len(tools.filter_by_connector(["BAMBOOHR"])) == 1
        assert len(tools.filter_by_connector(["bamboohr"])) == 1

    def test_filter_by_connector_returns_new_tools(self) -> None:
        """Test that filter_by_connector returns a new Tools instance."""
        from stackone_ai.models import ExecuteConfig, StackOneTool, ToolParameters, Tools

        tool = StackOneTool(
            description="Creates employee",
            parameters=ToolParameters(type="object", properties={}),
            _execute_config=ExecuteConfig(name="bamboohr_create_employee", method="POST", url="", headers={}),
            _api_key="test-key",
        )
        tools = Tools([tool])

        filtered = tools.filter_by_connector(["bamboohr"])

        assert filtered is not tools
        assert isinstance(filtered, Tools)


class TestSearchActionNamesWithAvailableConnectors:
    """Tests for search_action_names with available_connectors parameter."""

    @patch.object(SemanticSearchClient, "search")
    def test_filters_by_available_connectors(self, mock_search: MagicMock) -> None:
        """Test that results are filtered by available connectors."""
        from stackone_ai import StackOneToolSet

        mock_search.return_value = SemanticSearchResponse(
            results=[
                SemanticSearchResult(
                    action_name="bamboohr_create_employee",
                    connector_key="bamboohr",
                    similarity_score=0.95,
                    label="Create Employee",
                    description="Creates employee",
                ),
                SemanticSearchResult(
                    action_name="workday_create_worker",
                    connector_key="workday",
                    similarity_score=0.90,
                    label="Create Worker",
                    description="Creates worker",
                ),
                SemanticSearchResult(
                    action_name="hibob_create_employee",
                    connector_key="hibob",
                    similarity_score=0.85,
                    label="Create Employee",
                    description="Creates employee",
                ),
            ],
            total_count=3,
            query="create employee",
        )

        toolset = StackOneToolSet(api_key="test-key")
        results = toolset.search_action_names(
            "create employee",
            available_connectors={"bamboohr", "hibob"},
            top_k=10,
        )

        # workday should be filtered out
        assert len(results) == 2
        action_names = [r.action_name for r in results]
        assert "bamboohr_create_employee" in action_names
        assert "hibob_create_employee" in action_names
        assert "workday_create_worker" not in action_names

    @patch.object(SemanticSearchClient, "search")
    def test_fetches_max_then_falls_back_per_connector(self, mock_search: MagicMock) -> None:
        """Test that API fetches max results first, then per-connector if not enough."""
        from stackone_ai import StackOneToolSet

        mock_search.return_value = SemanticSearchResponse(
            results=[],
            total_count=0,
            query="test",
        )

        toolset = StackOneToolSet(api_key="test-key")
        toolset.search_action_names(
            "test",
            available_connectors={"bamboohr"},
            top_k=5,
        )

        # First call: fetch API max (500) for broad search
        # Second call: per-connector fallback for "bamboohr" since first returned nothing
        assert mock_search.call_count == 2
        first_call = mock_search.call_args_list[0].kwargs
        assert first_call["top_k"] == 500
        assert first_call["connector"] is None
        second_call = mock_search.call_args_list[1].kwargs
        assert second_call["connector"] == "bamboohr"
        assert second_call["top_k"] == 5

    @patch.object(SemanticSearchClient, "search")
    def test_respects_top_k_after_filtering(self, mock_search: MagicMock) -> None:
        """Test that results are limited to top_k after filtering."""
        from stackone_ai import StackOneToolSet

        # Return more results than top_k
        mock_search.return_value = SemanticSearchResponse(
            results=[
                SemanticSearchResult(
                    action_name=f"bamboohr_action_{i}",
                    connector_key="bamboohr",
                    similarity_score=0.9 - i * 0.1,
                    label=f"Action {i}",
                    description=f"Action {i}",
                )
                for i in range(10)
            ],
            total_count=10,
            query="test",
        )

        toolset = StackOneToolSet(api_key="test-key")
        results = toolset.search_action_names(
            "test",
            available_connectors={"bamboohr"},
            top_k=3,
        )

        assert len(results) == 3
