"""Semantic search client for StackOne action search API.

How Semantic Search Works
=========================

The SDK provides three ways to discover tools using semantic search.
Each path trades off between speed, filtering, and completeness.

1. ``search_tools(query)`` — Full tool discovery (recommended for agent frameworks)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This is the primary method used when integrating with OpenAI, LangChain, or CrewAI.
The internal flow is:

1. Fetch tools from linked accounts via MCP to discover available connectors
2. Search EACH connector in parallel via the semantic search API (/actions/search)
3. The search API returns results with full ``input_schema`` for each action
4. Build executable tools directly from search results (no match-back needed)
5. Deduplicate by action_id, sort by relevance score, apply top_k
6. Return Tools sorted by relevance score

Key point: only the user's own connectors are searched — no wasted results
from connectors the user doesn't have. The search API returns ``input_schema``
with each result, so tools can be built directly without a separate fetch.

If the semantic API is unavailable, the SDK falls back to a local
BM25 + TF-IDF hybrid search over the fetched tools (unless
``search="semantic"`` is specified).


2. ``search_action_names(query)`` — Lightweight discovery
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Queries the semantic API directly and returns action metadata
(action_id, connector, score, description, input_schema) **without**
building full tool objects. Useful for previewing results before
committing to a full fetch.

When ``account_ids`` are provided, each connector is searched in
parallel (same as ``search_tools``). Without ``account_ids``, results
come from the full StackOne catalog.


3. ``toolset.get_search_tool()`` — Agent-loop callable
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Returns a callable ``SearchTool`` that wraps ``search_tools()``.
Call it with a natural language query to get a ``Tools`` collection
back. Designed for agent loops where the LLM decides what to search for.
"""

from __future__ import annotations

import base64
from typing import Any

import httpx
from pydantic import BaseModel

from stackone_ai.constants import DEFAULT_BASE_URL


class SemanticSearchError(Exception):
    """Raised when semantic search fails."""

    pass


class SemanticSearchResult(BaseModel):
    """Single result from semantic search API."""

    id: str
    similarity_score: float


class SemanticSearchResponse(BaseModel):
    """Response from /actions/search endpoint."""

    results: list[SemanticSearchResult]
    total_count: int
    query: str
    connector_filter: str | None = None
    project_filter: str | None = None


class SemanticSearchClient:
    """Client for StackOne semantic search API.

    This client provides access to the semantic search endpoint which uses
    enhanced embeddings for higher accuracy than local BM25+TF-IDF search.

    Example:
        client = SemanticSearchClient(api_key="sk-xxx")
        response = client.search("create employee", connector="bamboohr", top_k=5)
        for result in response.results:
            print(f"{result.action_id}: {result.similarity_score:.2f}")
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = 30.0,
    ) -> None:
        """Initialize the semantic search client.

        Args:
            api_key: StackOne API key
            base_url: Base URL for API requests
            timeout: Request timeout in seconds
        """
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _build_auth_header(self) -> str:
        """Build the Basic auth header."""
        token = base64.b64encode(f"{self.api_key}:".encode()).decode()
        return f"Basic {token}"

    def search(
        self,
        query: str,
        connector: str | None = None,
        top_k: int | None = None,
        project_id: str | None = None,
        min_similarity: float | None = None,
    ) -> SemanticSearchResponse:
        """Search for relevant actions using semantic search.

        Args:
            query: Natural language query describing what tools/actions you need
            connector: Optional connector/provider filter (e.g., "bamboohr", "slack")
            top_k: Maximum number of results to return. If not provided, uses the backend default.
            project_id: Optional project scope (e.g., "103/dev-56501"). When provided,
                results include both global actions and project-specific actions.
            min_similarity: Minimum similarity score threshold (0-1). If not provided,
                the server uses its default (currently 0.4).

        Returns:
            SemanticSearchResponse containing matching actions with similarity scores

        Raises:
            SemanticSearchError: If the API call fails

        Example:
            response = client.search("onboard a new team member", top_k=5)
            for result in response.results:
                print(f"{result.action_id}: {result.similarity_score:.2f}")
        """
        url = f"{self.base_url}/actions/search"
        headers = {
            "Authorization": self._build_auth_header(),
            "Content-Type": "application/json",
        }
        payload: dict[str, Any] = {"query": query}
        if top_k is not None:
            payload["top_k"] = top_k
        if connector:
            payload["connector"] = connector
        if project_id:
            payload["project_id"] = project_id
        if min_similarity is not None:
            payload["min_similarity"] = min_similarity

        try:
            response = httpx.post(url, json=payload, headers=headers, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()
            return SemanticSearchResponse(**data)
        except httpx.HTTPStatusError as e:
            raise SemanticSearchError(f"API error: {e.response.status_code} - {e.response.text}") from e
        except httpx.RequestError as e:
            raise SemanticSearchError(f"Request failed: {e}") from e
        except Exception as e:
            raise SemanticSearchError(f"Search failed: {e}") from e

    def search_action_names(
        self,
        query: str,
        connector: str | None = None,
        top_k: int | None = None,
        min_similarity: float | None = None,
        project_id: str | None = None,
    ) -> list[str]:
        """Convenience method returning just action names.

        Args:
            query: Natural language query
            connector: Optional connector/provider filter
            top_k: Maximum number of results. If not provided, uses the backend default.
            min_similarity: Minimum similarity score threshold (0-1). If not provided,
                the server uses its default.
            project_id: Optional project scope for multi-tenant filtering

        Returns:
            List of action names sorted by relevance

        Example:
            action_names = client.search_action_names(
                "create employee",
                connector="bamboohr",
                min_similarity=0.5
            )
        """
        response = self.search(query, connector, top_k, project_id, min_similarity=min_similarity)
        return [r.id for r in response.results]
