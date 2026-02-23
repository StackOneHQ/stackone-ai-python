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

        # Without min_similarity — returns all results
        names = client.search_action_names("create employee")
        assert len(names) == 2
        assert "bamboohr_create_employee" in names
        assert "hibob_create_employee" in names

        # With min_similarity — passes threshold to server
        names = client.search_action_names("create employee", min_similarity=0.5)
        assert len(names) == 2  # Mock returns same data; filtering is server-side
        # Verify min_similarity was sent in the request payload
        last_call_kwargs = mock_post.call_args
        payload = last_call_kwargs.kwargs.get("json") or last_call_kwargs[1].get("json")
        assert payload["min_similarity"] == 0.5


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

        # Mock semantic search to return versioned API names (including some for unavailable connectors)
        mock_search.return_value = SemanticSearchResponse(
            results=[
                SemanticSearchResult(
                    action_name="bamboohr_1.0.0_bamboohr_create_employee_global",
                    connector_key="bamboohr",
                    similarity_score=0.95,
                    label="Create Employee",
                    description="Creates a new employee",
                ),
                SemanticSearchResult(
                    action_name="workday_1.0.0_workday_create_worker_global",
                    connector_key="workday",  # User doesn't have this connector
                    similarity_score=0.90,
                    label="Create Worker",
                    description="Creates a new worker",
                ),
                SemanticSearchResult(
                    action_name="hibob_1.0.0_hibob_create_employee_global",
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
    def test_toolset_search_tools_fallback_respects_connector(
        self,
        mock_fetch: MagicMock,
        mock_search: MagicMock,
    ) -> None:
        """Test BM25 fallback filters to the requested connector."""
        from stackone_ai import StackOneToolSet
        from stackone_ai.toolset import _McpToolDefinition

        mock_search.side_effect = SemanticSearchError("API unavailable")

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
        tools = toolset.search_tools("create employee", connector="bamboohr", fallback_to_local=True)

        assert len(tools) > 0
        tool_names = [t.name for t in tools]
        for name in tool_names:
            assert name.split("_")[0] == "bamboohr"

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
                    action_name="bamboohr_1.0.0_bamboohr_create_employee_global",
                    connector_key="bamboohr",
                    similarity_score=0.92,
                    label="Create Employee",
                    description="Creates a new employee",
                ),
                SemanticSearchResult(
                    action_name="hibob_1.0.0_hibob_create_employee_global",
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
        results = toolset.search_action_names("create employee", min_similarity=0.5)

        # min_similarity is passed to server; mock returns both results
        # Verify results are normalized
        assert len(results) == 2
        assert results[0].action_name == "bamboohr_create_employee"
        assert results[1].action_name == "hibob_create_employee"
        # Verify min_similarity was passed to the search call
        mock_search.assert_called_with(
            query="create employee", connector=None, top_k=None, min_similarity=0.5
        )

    def test_utility_tools_semantic_search(self) -> None:
        """Test utility_tools with semantic search."""
        from stackone_ai.models import StackOneTool, Tools

        # Create a mock tools collection
        tool = MagicMock(spec=StackOneTool)
        tool.name = "test_tool"
        tool.description = "Test tool"
        tool.connector = "test"
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
            # Should pass available connectors from the tools collection
            mock_create.assert_called_once_with(mock_client, available_connectors={"test"})


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
                    action_name="bamboohr_1.0.0_bamboohr_create_employee_global",
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
        # Name should be normalized from versioned API format to MCP format
        assert result["tools"][0]["name"] == "bamboohr_create_employee"
        assert result["tools"][0]["score"] == 0.92
        assert result["tools"][0]["connector"] == "bamboohr"

    @patch.object(SemanticSearchClient, "search")
    def test_semantic_tool_search_with_min_similarity(self, mock_search: MagicMock) -> None:
        """Test semantic tool search passes min_similarity to server."""
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
            ],
            total_count=1,
            query="test",
        )

        client = SemanticSearchClient(api_key="test-key")
        tool = create_semantic_tool_search(client)

        result = tool.execute({"query": "test", "limit": 10, "minSimilarity": 0.5})

        assert len(result["tools"]) == 1
        assert result["tools"][0]["name"] == "high_score_action"
        # Verify min_similarity was passed to the search API
        mock_search.assert_called_once_with(
            query="test", connector=None, top_k=10, min_similarity=0.5
        )

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
            min_similarity=None,
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
        assert "minSimilarity" in props
        assert "connector" in props


class TestSemanticToolSearchScoping:
    """Tests for connector scoping in create_semantic_tool_search."""

    @patch.object(SemanticSearchClient, "search")
    def test_scoped_searches_each_connector_in_parallel(self, mock_search: MagicMock) -> None:
        """Test that available_connectors triggers per-connector parallel searches."""
        from stackone_ai.utility_tools import create_semantic_tool_search

        def _search_side_effect(
            query: str,
            connector: str | None = None,
            top_k: int | None = None,
            min_similarity: float | None = None,
        ) -> SemanticSearchResponse:
            if connector == "bamboohr":
                return SemanticSearchResponse(
                    results=[
                        SemanticSearchResult(
                            action_name="bamboohr_create_employee",
                            connector_key="bamboohr",
                            similarity_score=0.95,
                            label="Create Employee",
                            description="Creates employee",
                        ),
                    ],
                    total_count=1,
                    query=query,
                )
            elif connector == "hibob":
                return SemanticSearchResponse(
                    results=[
                        SemanticSearchResult(
                            action_name="hibob_create_employee",
                            connector_key="hibob",
                            similarity_score=0.85,
                            label="Create Employee",
                            description="Creates employee",
                        ),
                    ],
                    total_count=1,
                    query=query,
                )
            return SemanticSearchResponse(results=[], total_count=0, query=query)

        mock_search.side_effect = _search_side_effect

        client = SemanticSearchClient(api_key="test-key")
        tool = create_semantic_tool_search(client, available_connectors={"bamboohr", "hibob"})

        result = tool.execute({"query": "create employee", "limit": 10})

        # Should have searched each connector separately
        assert mock_search.call_count == 2
        called_connectors = {call.kwargs.get("connector") for call in mock_search.call_args_list}
        assert called_connectors == {"bamboohr", "hibob"}

        # Should return results from both connectors
        assert len(result["tools"]) == 2
        names = [t["name"] for t in result["tools"]]
        assert "bamboohr_create_employee" in names
        assert "hibob_create_employee" in names

    @patch.object(SemanticSearchClient, "search")
    def test_scoped_agent_connector_intersects_with_available(self, mock_search: MagicMock) -> None:
        """Test that agent's connector param is intersected with available_connectors."""
        from stackone_ai.utility_tools import create_semantic_tool_search

        mock_search.return_value = SemanticSearchResponse(
            results=[
                SemanticSearchResult(
                    action_name="bamboohr_create_employee",
                    connector_key="bamboohr",
                    similarity_score=0.95,
                    label="Create Employee",
                    description="Creates employee",
                ),
            ],
            total_count=1,
            query="create employee",
        )

        client = SemanticSearchClient(api_key="test-key")
        tool = create_semantic_tool_search(client, available_connectors={"bamboohr", "hibob"})

        # Agent requests connector="bamboohr" — should only search bamboohr
        tool.execute({"query": "create employee", "connector": "bamboohr"})

        assert mock_search.call_count == 1
        assert mock_search.call_args.kwargs["connector"] == "bamboohr"

    @patch.object(SemanticSearchClient, "search")
    def test_scoped_agent_connector_not_available_returns_empty(self, mock_search: MagicMock) -> None:
        """Test that requesting an unavailable connector returns empty results."""
        from stackone_ai.utility_tools import create_semantic_tool_search

        client = SemanticSearchClient(api_key="test-key")
        tool = create_semantic_tool_search(client, available_connectors={"bamboohr", "hibob"})

        # Agent requests connector="workday" — not in available_connectors
        result = tool.execute({"query": "create employee", "connector": "workday"})

        # Should not call API at all
        mock_search.assert_not_called()
        assert result["tools"] == []

    @patch.object(SemanticSearchClient, "search")
    def test_no_connectors_queries_full_catalog(self, mock_search: MagicMock) -> None:
        """Test that available_connectors=None preserves full catalog behavior (backwards compat)."""
        from stackone_ai.utility_tools import create_semantic_tool_search

        mock_search.return_value = SemanticSearchResponse(
            results=[
                SemanticSearchResult(
                    action_name="workday_create_worker",
                    connector_key="workday",
                    similarity_score=0.90,
                    label="Create Worker",
                    description="Creates worker",
                ),
            ],
            total_count=1,
            query="create employee",
        )

        client = SemanticSearchClient(api_key="test-key")
        tool = create_semantic_tool_search(client)  # No available_connectors

        result = tool.execute({"query": "create employee", "limit": 5})

        # Should make a single call without connector scoping
        mock_search.assert_called_once_with(
            query="create employee",
            connector=None,
            top_k=5,
            min_similarity=None,
        )
        assert len(result["tools"]) == 1
        assert result["tools"][0]["name"] == "workday_create_worker"


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
    """Tests for Tools.get_connectors()."""

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


class TestSearchActionNamesWithAccountIds:
    """Tests for search_action_names with account_ids parameter."""

    @patch.object(SemanticSearchClient, "search")
    @patch("stackone_ai.toolset._fetch_mcp_tools")
    def test_filters_by_account_connectors(self, mock_fetch: MagicMock, mock_search: MagicMock) -> None:
        """Test that only connectors from linked accounts are searched (per-connector parallel)."""
        from stackone_ai import StackOneToolSet
        from stackone_ai.toolset import _McpToolDefinition

        def _search_side_effect(
            query: str,
            connector: str | None = None,
            top_k: int | None = None,
            min_similarity: float | None = None,
        ) -> SemanticSearchResponse:
            if connector == "bamboohr":
                return SemanticSearchResponse(
                    results=[
                        SemanticSearchResult(
                            action_name="bamboohr_1.0.0_bamboohr_create_employee_global",
                            connector_key="bamboohr",
                            similarity_score=0.95,
                            label="Create Employee",
                            description="Creates employee",
                        ),
                    ],
                    total_count=1,
                    query=query,
                )
            elif connector == "hibob":
                return SemanticSearchResponse(
                    results=[
                        SemanticSearchResult(
                            action_name="hibob_1.0.0_hibob_create_employee_global",
                            connector_key="hibob",
                            similarity_score=0.85,
                            label="Create Employee",
                            description="Creates employee",
                        ),
                    ],
                    total_count=1,
                    query=query,
                )
            return SemanticSearchResponse(results=[], total_count=0, query=query)

        mock_search.side_effect = _search_side_effect

        # Mock MCP to return only bamboohr and hibob tools (user's linked accounts)
        mock_fetch.return_value = [
            _McpToolDefinition(
                name="bamboohr_create_employee",
                description="Creates employee",
                input_schema={"type": "object", "properties": {}},
            ),
            _McpToolDefinition(
                name="hibob_create_employee",
                description="Creates employee",
                input_schema={"type": "object", "properties": {}},
            ),
        ]

        toolset = StackOneToolSet(api_key="test-key")
        results = toolset.search_action_names(
            "create employee",
            account_ids=["acc-123"],
            top_k=10,
        )

        # Only bamboohr and hibob searched (workday never queried)
        assert len(results) == 2
        action_names = [r.action_name for r in results]
        assert "bamboohr_create_employee" in action_names
        assert "hibob_create_employee" in action_names
        # Verify only per-connector calls were made (no global call)
        assert mock_search.call_count == 2
        called_connectors = {call.kwargs.get("connector") for call in mock_search.call_args_list}
        assert called_connectors == {"bamboohr", "hibob"}

    @patch.object(SemanticSearchClient, "search")
    def test_search_action_names_returns_empty_on_failure(self, mock_search: MagicMock) -> None:
        """Test that search_action_names returns [] when semantic search fails."""
        from stackone_ai import StackOneToolSet

        mock_search.side_effect = SemanticSearchError("API unavailable")

        toolset = StackOneToolSet(api_key="test-key")
        results = toolset.search_action_names("create employee")

        assert results == []

    @patch.object(SemanticSearchClient, "search")
    @patch("stackone_ai.toolset._fetch_mcp_tools")
    def test_searches_all_connectors_in_parallel(self, mock_fetch: MagicMock, mock_search: MagicMock) -> None:
        """Test that all available connectors are searched directly (no global call + fallback)."""
        from stackone_ai import StackOneToolSet
        from stackone_ai.toolset import _McpToolDefinition

        mock_search.return_value = SemanticSearchResponse(
            results=[],
            total_count=0,
            query="test",
        )

        # Mock MCP to return tools from two connectors
        mock_fetch.return_value = [
            _McpToolDefinition(
                name="bamboohr_list_employees",
                description="Lists employees",
                input_schema={"type": "object", "properties": {}},
            ),
            _McpToolDefinition(
                name="hibob_list_employees",
                description="Lists employees",
                input_schema={"type": "object", "properties": {}},
            ),
        ]

        toolset = StackOneToolSet(api_key="test-key")
        toolset.search_action_names(
            "test",
            account_ids=["acc-123"],
            top_k=5,
        )

        # Each connector gets its own search call (parallel, not sequential fallback)
        assert mock_search.call_count == 2
        called_connectors = {call.kwargs.get("connector") for call in mock_search.call_args_list}
        assert called_connectors == {"bamboohr", "hibob"}
        # top_k is passed to each per-connector call
        for call in mock_search.call_args_list:
            assert call.kwargs["top_k"] == 5

    @patch.object(SemanticSearchClient, "search")
    @patch("stackone_ai.toolset._fetch_mcp_tools")
    def test_respects_top_k_after_filtering(self, mock_fetch: MagicMock, mock_search: MagicMock) -> None:
        """Test that results are limited to top_k after filtering."""
        from stackone_ai import StackOneToolSet
        from stackone_ai.toolset import _McpToolDefinition

        # Return more results than top_k using versioned API names
        mock_search.return_value = SemanticSearchResponse(
            results=[
                SemanticSearchResult(
                    action_name=f"bamboohr_1.0.0_bamboohr_action_{i}_global",
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

        # Mock MCP to return bamboohr tools
        mock_fetch.return_value = [
            _McpToolDefinition(
                name="bamboohr_action_0",
                description="Action 0",
                input_schema={"type": "object", "properties": {}},
            ),
        ]

        toolset = StackOneToolSet(api_key="test-key")
        results = toolset.search_action_names(
            "test",
            account_ids=["acc-123"],
            top_k=3,
        )

        # Should be limited to top_k after normalization
        assert len(results) == 3
        # Names should be normalized
        assert results[0].action_name == "bamboohr_action_0"


class TestNormalizeActionName:
    """Tests for _normalize_action_name() function."""

    def test_versioned_name_is_normalized(self) -> None:
        """Test that versioned API names are normalized to MCP format."""
        from stackone_ai.utils.normalize import _normalize_action_name

        assert (
            _normalize_action_name("calendly_1.0.0_calendly_create_scheduling_link_global")
            == "calendly_create_scheduling_link"
        )

    def test_multi_segment_version(self) -> None:
        """Test normalization with multi-segment semver."""
        from stackone_ai.utils.normalize import _normalize_action_name

        assert (
            _normalize_action_name("breathehr_1.0.1_breathehr_list_employees_global")
            == "breathehr_list_employees"
        )

    def test_already_normalized_name_unchanged(self) -> None:
        """Test that MCP-format names pass through unchanged."""
        from stackone_ai.utils.normalize import _normalize_action_name

        assert _normalize_action_name("bamboohr_create_employee") == "bamboohr_create_employee"

    def test_non_matching_name_unchanged(self) -> None:
        """Test that names that don't match the pattern pass through unchanged."""
        from stackone_ai.utils.normalize import _normalize_action_name

        assert _normalize_action_name("some_random_tool") == "some_random_tool"

    def test_empty_string(self) -> None:
        """Test empty string input."""
        from stackone_ai.utils.normalize import _normalize_action_name

        assert _normalize_action_name("") == ""

    def test_multiple_versions_normalize_to_same(self) -> None:
        """Test that different versions of the same action normalize identically."""
        from stackone_ai.utils.normalize import _normalize_action_name

        name_v1 = _normalize_action_name("breathehr_1.0.0_breathehr_list_employees_global")
        name_v2 = _normalize_action_name("breathehr_1.0.1_breathehr_list_employees_global")
        assert name_v1 == name_v2 == "breathehr_list_employees"


class TestSemanticSearchDeduplication:
    """Tests for deduplication after name normalization."""

    @patch.object(SemanticSearchClient, "search")
    @patch("stackone_ai.toolset._fetch_mcp_tools")
    def test_search_tools_deduplicates_versions(self, mock_fetch: MagicMock, mock_search: MagicMock) -> None:
        """Test that search_tools deduplicates multiple API versions of the same action."""
        from stackone_ai import StackOneToolSet
        from stackone_ai.toolset import _McpToolDefinition

        mock_search.return_value = SemanticSearchResponse(
            results=[
                SemanticSearchResult(
                    action_name="breathehr_1.0.0_breathehr_list_employees_global",
                    connector_key="breathehr",
                    similarity_score=0.95,
                    label="List Employees",
                    description="Lists employees",
                ),
                SemanticSearchResult(
                    action_name="breathehr_1.0.1_breathehr_list_employees_global",
                    connector_key="breathehr",
                    similarity_score=0.90,
                    label="List Employees v2",
                    description="Lists employees v2",
                ),
                SemanticSearchResult(
                    action_name="bamboohr_1.0.0_bamboohr_create_employee_global",
                    connector_key="bamboohr",
                    similarity_score=0.85,
                    label="Create Employee",
                    description="Creates employee",
                ),
            ],
            total_count=3,
            query="list employees",
        )

        mock_fetch.return_value = [
            _McpToolDefinition(
                name="breathehr_list_employees",
                description="Lists employees",
                input_schema={"type": "object", "properties": {}},
            ),
            _McpToolDefinition(
                name="bamboohr_create_employee",
                description="Creates employee",
                input_schema={"type": "object", "properties": {}},
            ),
        ]

        toolset = StackOneToolSet(api_key="test-key")
        tools = toolset.search_tools("list employees", top_k=5)

        # Should deduplicate: both breathehr versions -> breathehr_list_employees
        tool_names = [t.name for t in tools]
        assert tool_names.count("breathehr_list_employees") == 1
        assert "bamboohr_create_employee" in tool_names
        assert len(tools) == 2

    @patch.object(SemanticSearchClient, "search")
    def test_search_action_names_normalizes_versions(self, mock_search: MagicMock) -> None:
        """Test that search_action_names normalizes versioned API names."""
        from stackone_ai import StackOneToolSet

        mock_search.return_value = SemanticSearchResponse(
            results=[
                SemanticSearchResult(
                    action_name="breathehr_1.0.0_breathehr_list_employees_global",
                    connector_key="breathehr",
                    similarity_score=0.95,
                    label="List Employees",
                    description="Lists employees",
                ),
                SemanticSearchResult(
                    action_name="breathehr_1.0.1_breathehr_list_employees_global",
                    connector_key="breathehr",
                    similarity_score=0.90,
                    label="List Employees v2",
                    description="Lists employees v2",
                ),
            ],
            total_count=2,
            query="list employees",
        )

        toolset = StackOneToolSet(api_key="test-key")
        results = toolset.search_action_names("list employees", top_k=5)

        # Both results are returned with normalized names (no dedup in global path)
        assert len(results) == 2
        assert results[0].action_name == "breathehr_list_employees"
        assert results[1].action_name == "breathehr_list_employees"
        # Sorted by score descending
        assert results[0].similarity_score == 0.95
        assert results[1].similarity_score == 0.90

    @patch.object(SemanticSearchClient, "search")
    def test_semantic_tool_search_deduplicates_versions(self, mock_search: MagicMock) -> None:
        """Test that create_semantic_tool_search deduplicates API versions."""
        from stackone_ai.utility_tools import create_semantic_tool_search

        mock_search.return_value = SemanticSearchResponse(
            results=[
                SemanticSearchResult(
                    action_name="breathehr_1.0.0_breathehr_list_employees_global",
                    connector_key="breathehr",
                    similarity_score=0.95,
                    label="List Employees",
                    description="Lists employees",
                ),
                SemanticSearchResult(
                    action_name="breathehr_1.0.1_breathehr_list_employees_global",
                    connector_key="breathehr",
                    similarity_score=0.90,
                    label="List Employees v2",
                    description="Lists employees v2",
                ),
            ],
            total_count=2,
            query="list employees",
        )

        client = SemanticSearchClient(api_key="test-key")
        tool = create_semantic_tool_search(client)
        result = tool.execute({"query": "list employees", "limit": 10})

        # Should deduplicate: only one result
        assert len(result["tools"]) == 1
        assert result["tools"][0]["name"] == "breathehr_list_employees"
        assert result["tools"][0]["score"] == 0.95
