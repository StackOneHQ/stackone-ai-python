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

    @patch.object(SemanticSearchClient, "search_action_names")
    @patch("stackone_ai.toolset._fetch_mcp_tools")
    def test_toolset_search_tools(
        self,
        mock_fetch: MagicMock,
        mock_search: MagicMock,
    ) -> None:
        """Test toolset.search_tools() method."""
        from stackone_ai import StackOneToolSet
        from stackone_ai.toolset import _McpToolDefinition

        # Mock semantic search to return action names
        mock_search.return_value = ["bamboohr_create_employee", "hibob_create_employee"]

        # Mock MCP fetch to return tools using actual dataclass
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

        # Should only return the 2 matching tools
        assert len(tools) == 2
        tool_names = [t.name for t in tools]
        assert "bamboohr_create_employee" in tool_names
        assert "hibob_create_employee" in tool_names
        assert "bamboohr_list_employees" not in tool_names

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
            patch("stackone_ai.utility_tools.ToolIndex") as mock_index_class,
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

        # With semantic search - requires client
        with pytest.raises(ValueError) as exc_info:
            tools.utility_tools(use_semantic_search=True)
        assert "semantic_client is required" in str(exc_info.value)

        # With semantic search and client
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
            utility = tools.utility_tools(use_semantic_search=True, semantic_client=mock_client)
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
        assert "84%" in tool.description

        props = tool.parameters.properties
        assert "query" in props
        assert "limit" in props
        assert "minScore" in props
        assert "connector" in props
