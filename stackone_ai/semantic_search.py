"""Semantic search client for StackOne action search API.

How Semantic Search Works
=========================

The SDK provides three ways to discover tools using semantic search.
Each path trades off between speed, filtering, and completeness.

1. ``search_tools(query)`` — Full tool discovery (recommended for agent frameworks)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This is the primary method used when integrating with OpenAI, LangChain, or CrewAI.
The internal flow is:

::

    User query (e.g. "create an employee")
        │
        ▼
    ┌─────────────────────────────────────────────────────┐
    │ Step 1: Fetch ALL tools from linked accounts via MCP │
    │         (uses account_ids to scope the request)      │
    └────────────────────────┬────────────────────────────┘
                             │
                             ▼
    ┌─────────────────────────────────────────────────────┐
    │ Step 2: Extract available connectors from the       │
    │         fetched tools (e.g. {bamboohr, hibob})      │
    └────────────────────────┬────────────────────────────┘
                             │
                             ▼
    ┌─────────────────────────────────────────────────────┐
    │ Step 3: Query the semantic search API (/actions/    │
    │         search) with the natural language query     │
    └────────────────────────┬────────────────────────────┘
                             │
                             ▼
    ┌─────────────────────────────────────────────────────┐
    │ Step 4: Filter results — keep only connectors the   │
    │         user has access to + apply min_score cutoff  │
    │                                                     │
    │         If not enough results, make per-connector    │
    │         fallback queries for missing connectors      │
    └────────────────────────┬────────────────────────────┘
                             │
                             ▼
    ┌─────────────────────────────────────────────────────┐
    │ Step 5: Deduplicate by normalized action name       │
    │         (strips API version suffixes, keeps highest  │
    │         scoring version of each action)              │
    └────────────────────────┬────────────────────────────┘
                             │
                             ▼
    ┌─────────────────────────────────────────────────────┐
    │ Step 6: Match semantic results back to the fetched   │
    │         tool definitions from Step 1                 │
    │         Return Tools sorted by relevance score       │
    └─────────────────────────────────────────────────────┘

Key point: tools are fetched first, semantic search runs second, and only
tools that exist in the user's linked accounts AND match the semantic query
are returned. This prevents suggesting tools the user cannot execute.

If the semantic API is unavailable, the SDK falls back to a local
BM25 + TF-IDF hybrid search over the fetched tools (unless
``fallback_to_local=False``).


2. ``search_action_names(query)`` — Lightweight discovery
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Queries the semantic API directly and returns action name metadata
(name, connector, score, description) **without** fetching full tool
definitions. This is useful for previewing results before committing
to a full fetch.

When ``account_ids`` are provided, tools are fetched only to determine
available connectors — results are then filtered to those connectors.
Without ``account_ids``, results come from the full StackOne catalog.


3. ``utility_tools(semantic_client=...)`` — Agent-loop search + execute
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Creates a ``tool_search`` utility tool that agents can call inside a
loop. The agent searches for tools, inspects results, then calls
``tool_execute`` to run the chosen tool. When ``semantic_client`` is
passed, ``tool_search`` uses cloud-based semantic vectors instead of
local BM25 + TF-IDF.

Note: utility tool search queries the **full backend catalog** (all
connectors), not just the ones in the user's linked accounts.
"""

from __future__ import annotations

import base64
from typing import Any

import httpx
from pydantic import BaseModel


class SemanticSearchError(Exception):
    """Raised when semantic search fails."""

    pass


class SemanticSearchResult(BaseModel):
    """Single result from semantic search API."""

    action_name: str
    connector_key: str
    similarity_score: float
    label: str
    description: str


class SemanticSearchResponse(BaseModel):
    """Response from /actions/search endpoint."""

    results: list[SemanticSearchResult]
    total_count: int
    query: str


class SemanticSearchClient:
    """Client for StackOne semantic search API.

    This client provides access to the semantic search endpoint which uses
    enhanced embeddings for higher accuracy than local BM25+TF-IDF search.

    Example:
        client = SemanticSearchClient(api_key="sk-xxx")
        response = client.search("create employee", connector="bamboohr", top_k=5)
        for result in response.results:
            print(f"{result.action_name}: {result.similarity_score:.2f}")
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.stackone.com",
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
    ) -> SemanticSearchResponse:
        """Search for relevant actions using semantic search.

        Args:
            query: Natural language query describing what tools/actions you need
            connector: Optional connector/provider filter (e.g., "bamboohr", "slack")
            top_k: Maximum number of results to return. If not provided, uses the backend default.

        Returns:
            SemanticSearchResponse containing matching actions with similarity scores

        Raises:
            SemanticSearchError: If the API call fails

        Example:
            response = client.search("onboard a new team member", top_k=5)
            for result in response.results:
                print(f"{result.action_name}: {result.similarity_score:.2f}")
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
        min_score: float = 0.0,
    ) -> list[str]:
        """Convenience method returning just action names.

        Args:
            query: Natural language query
            connector: Optional connector/provider filter
            top_k: Maximum number of results. If not provided, uses the backend default.
            min_score: Minimum similarity score threshold (0-1)

        Returns:
            List of action names sorted by relevance

        Example:
            action_names = client.search_action_names(
                "create employee",
                connector="bamboohr",
                min_score=0.5
            )
        """
        response = self.search(query, connector, top_k)
        return [r.action_name for r in response.results if r.similarity_score >= min_score]
