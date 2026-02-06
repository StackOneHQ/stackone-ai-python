"""Utility tools for dynamic tool discovery and execution"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import bm25s
import numpy as np
from pydantic import BaseModel

from stackone_ai.constants import DEFAULT_HYBRID_ALPHA
from stackone_ai.models import ExecuteConfig, JsonDict, StackOneTool, ToolParameters
from stackone_ai.utils.tfidf_index import TfidfDocument, TfidfIndex

if TYPE_CHECKING:
    from stackone_ai.models import Tools
    from stackone_ai.semantic_search import SemanticSearchClient


class ToolSearchResult(BaseModel):
    """Result from tool_search"""

    name: str
    description: str
    score: float


class ToolIndex:
    """Hybrid BM25 + TF-IDF tool search index"""

    def __init__(self, tools: list[StackOneTool], hybrid_alpha: float | None = None) -> None:
        """Initialize tool index with hybrid search

        Args:
            tools: List of tools to index
            hybrid_alpha: Weight for BM25 in hybrid search (0-1). If not provided,
                uses DEFAULT_HYBRID_ALPHA (0.2), which gives more weight to BM25 scoring
                and has been shown to provide better tool discovery accuracy
                (10.8% improvement in validation testing).
        """
        self.tools = tools
        self.tool_map = {tool.name: tool for tool in tools}
        # Use default if not provided, then clamp to [0, 1]
        alpha = hybrid_alpha if hybrid_alpha is not None else DEFAULT_HYBRID_ALPHA
        self.hybrid_alpha = max(0.0, min(1.0, alpha))

        # Prepare corpus for both BM25 and TF-IDF
        corpus = []
        tfidf_docs = []
        self.tool_names = []

        for tool in tools:
            # Extract category and action from tool name
            parts = tool.name.split("_")
            category = parts[0] if parts else ""

            # Extract action types
            action_types = ["create", "update", "delete", "get", "list", "search"]
            actions = [p for p in parts if p in action_types]

            # Combine name, description, category and tags for indexing
            # For TF-IDF: use weighted approach similar to Node.js
            tfidf_text = " ".join(
                [
                    f"{tool.name} {tool.name} {tool.name}",  # boost name
                    f"{category} {' '.join(actions)}",
                    tool.description,
                    " ".join(parts),
                ]
            )

            # For BM25: simpler approach
            bm25_text = " ".join(
                [
                    tool.name,
                    tool.description,
                    category,
                    " ".join(parts),
                    " ".join(actions),
                ]
            )

            corpus.append(bm25_text)
            tfidf_docs.append(TfidfDocument(id=tool.name, text=tfidf_text))
            self.tool_names.append(tool.name)

        # Create BM25 index
        self.bm25_retriever = bm25s.BM25()
        corpus_tokens = bm25s.tokenize(corpus, stemmer=None, show_progress=False)  # ty: ignore[invalid-argument-type]
        self.bm25_retriever.index(corpus_tokens)

        # Create TF-IDF index
        self.tfidf_index = TfidfIndex()
        self.tfidf_index.build(tfidf_docs)

    def search(self, query: str, limit: int = 5, min_score: float = 0.0) -> list[ToolSearchResult]:
        """Search for relevant tools using hybrid BM25 + TF-IDF

        Args:
            query: Natural language query
            limit: Maximum number of results
            min_score: Minimum relevance score (0-1)

        Returns:
            List of search results sorted by relevance
        """
        # Get more results initially to have better candidate pool for fusion
        fetch_limit = max(50, limit)

        # Tokenize query for BM25
        query_tokens = bm25s.tokenize([query], stemmer=None, show_progress=False)  # ty: ignore[invalid-argument-type]

        # Search with BM25
        bm25_results, bm25_scores = self.bm25_retriever.retrieve(
            query_tokens, k=min(fetch_limit, len(self.tools))
        )

        # Search with TF-IDF
        tfidf_results = self.tfidf_index.search(query, k=min(fetch_limit, len(self.tools)))

        # Build score map for fusion
        score_map: dict[str, dict[str, float]] = {}

        # Add BM25 scores
        for idx, score in zip(bm25_results[0], bm25_scores[0], strict=True):
            tool_name = self.tool_names[idx]
            # Normalize BM25 score to 0-1 range
            normalized_score = float(1 / (1 + np.exp(-score / 10)))
            # Clamp to [0, 1]
            clamped_score = max(0.0, min(1.0, normalized_score))
            score_map[tool_name] = {"bm25": clamped_score}

        # Add TF-IDF scores
        for result in tfidf_results:
            if result.id not in score_map:
                score_map[result.id] = {}
            score_map[result.id]["tfidf"] = result.score

        # Fuse scores: hybrid_score = alpha * bm25 + (1 - alpha) * tfidf
        fused_results: list[tuple[str, float]] = []
        for tool_name, scores in score_map.items():
            bm25_score = scores.get("bm25", 0.0)
            tfidf_score = scores.get("tfidf", 0.0)
            hybrid_score = self.hybrid_alpha * bm25_score + (1 - self.hybrid_alpha) * tfidf_score
            fused_results.append((tool_name, hybrid_score))

        # Sort by score descending
        fused_results.sort(key=lambda x: x[1], reverse=True)

        # Build final results
        search_results = []
        for tool_name, score in fused_results:
            if score < min_score:
                continue

            tool = self.tool_map.get(tool_name)
            if tool is None:
                continue

            search_results.append(
                ToolSearchResult(
                    name=tool.name,
                    description=tool.description,
                    score=score,
                )
            )

            if len(search_results) >= limit:
                break

        return search_results


def create_tool_search(index: ToolIndex) -> StackOneTool:
    """Create the tool_search tool

    Args:
        index: Tool search index

    Returns:
        Utility tool for searching relevant tools
    """
    name = "tool_search"
    description = (
        f"Searches for relevant tools based on a natural language query using hybrid BM25 + TF-IDF search "
        f"(alpha={index.hybrid_alpha}). This tool should be called first to discover available tools "
        f"before executing them."
    )

    parameters = ToolParameters(
        type="object",
        properties={
            "query": {
                "type": "string",
                "description": (
                    "Natural language query describing what tools you need "
                    '(e.g., "tools for managing employees", "create time off request")'
                ),
            },
            "limit": {
                "type": "number",
                "description": "Maximum number of tools to return (default: 5)",
                "default": 5,
                "nullable": True,
            },
            "minScore": {
                "type": "number",
                "description": "Minimum relevance score (0-1) to filter results (default: 0.0)",
                "default": 0.0,
                "nullable": True,
            },
        },
    )

    def execute_filter(arguments: str | JsonDict | None = None) -> JsonDict:
        """Execute the filter tool"""
        # Parse arguments
        if isinstance(arguments, str):
            kwargs = json.loads(arguments)
        else:
            kwargs = arguments or {}

        query = kwargs.get("query", "")
        limit = int(kwargs["limit"]) if kwargs.get("limit") is not None else 5
        min_score = float(kwargs["minScore"]) if kwargs.get("minScore") is not None else 0.0

        # Search for tools
        results = index.search(query, limit, min_score)

        # Format results
        tools_data = [
            {
                "name": r.name,
                "description": r.description,
                "score": r.score,
            }
            for r in results
        ]

        return {"tools": tools_data}

    # Create execute config for the meta tool
    execute_config = ExecuteConfig(
        name=name,
        method="POST",
        url="",  # Utility tools don't make HTTP requests
        headers={},
    )

    # Create a wrapper class that delegates execute to our custom function
    class ToolSearchTool(StackOneTool):
        """Utility tool for searching relevant tools"""

        def __init__(self) -> None:
            super().__init__(
                description=description,
                parameters=parameters,
                _execute_config=execute_config,
                _api_key="",  # Utility tools don't need API key
                _account_id=None,
            )

        def execute(
            self, arguments: str | JsonDict | None = None, *, options: JsonDict | None = None
        ) -> JsonDict:
            return execute_filter(arguments)

    return ToolSearchTool()


def create_semantic_tool_search(semantic_client: SemanticSearchClient) -> StackOneTool:
    """Create a semantic search variant of tool_search.

    Uses cloud semantic search API instead of local BM25+TF-IDF for
    improved natural language tool discovery.

    Args:
        semantic_client: Initialized SemanticSearchClient instance

    Returns:
        Utility tool for searching relevant tools using semantic search
    """
    from stackone_ai.semantic_search import SemanticSearchClient  # noqa: F811

    if not isinstance(semantic_client, SemanticSearchClient):
        raise TypeError("semantic_client must be a SemanticSearchClient instance")

    name = "tool_search"
    description = (
        "Searches for relevant tools based on a natural language query using "
        "semantic vector search. Call this first to discover "
        "available tools before executing them."
    )

    parameters = ToolParameters(
        type="object",
        properties={
            "query": {
                "type": "string",
                "description": (
                    "Natural language query describing what tools you need "
                    '(e.g., "onboard a new team member", "request vacation days")'
                ),
            },
            "limit": {
                "type": "number",
                "description": "Maximum number of tools to return (default: 5)",
                "default": 5,
                "nullable": True,
            },
            "minScore": {
                "type": "number",
                "description": "Minimum similarity score (0-1) to filter results (default: 0.0)",
                "default": 0.0,
                "nullable": True,
            },
            "connector": {
                "type": "string",
                "description": "Optional: filter by connector/provider (e.g., 'bamboohr', 'slack')",
                "nullable": True,
            },
        },
    )

    def execute_search(arguments: str | JsonDict | None = None) -> JsonDict:
        """Execute the semantic search tool"""
        if isinstance(arguments, str):
            kwargs = json.loads(arguments)
        else:
            kwargs = arguments or {}

        query = kwargs.get("query", "")
        limit = int(kwargs["limit"]) if kwargs.get("limit") is not None else 5
        min_score = float(kwargs["minScore"]) if kwargs.get("minScore") is not None else 0.0
        connector = kwargs.get("connector")

        response = semantic_client.search(
            query=query,
            connector=connector,
            top_k=limit,
        )

        tools_data = [
            {
                "name": r.action_name,
                "description": r.description,
                "score": r.similarity_score,
                "connector": r.connector_key,
            }
            for r in response.results
            if r.similarity_score >= min_score
        ]

        return {"tools": tools_data[:limit]}

    execute_config = ExecuteConfig(
        name=name,
        method="POST",
        url="",  # Utility tools don't make HTTP requests
        headers={},
    )

    class SemanticToolSearchTool(StackOneTool):
        """Utility tool for searching relevant tools using semantic search"""

        def __init__(self) -> None:
            super().__init__(
                description=description,
                parameters=parameters,
                _execute_config=execute_config,
                _api_key="",  # Utility tools don't need API key
                _account_id=None,
            )

        def execute(
            self, arguments: str | JsonDict | None = None, *, options: JsonDict | None = None
        ) -> JsonDict:
            return execute_search(arguments)

    return SemanticToolSearchTool()


def create_tool_execute(tools_collection: Tools) -> StackOneTool:
    """Create the tool_execute tool

    Args:
        tools_collection: Collection of tools to execute from

    Returns:
        Utility tool for executing discovered tools
    """
    name = "tool_execute"
    description = (
        "Executes a tool by name with the provided parameters. "
        "Use this after discovering tools with tool_search."
    )

    parameters = ToolParameters(
        type="object",
        properties={
            "toolName": {
                "type": "string",
                "description": "Name of the tool to execute",
            },
            "params": {
                "type": "object",
                "description": "Parameters to pass to the tool",
                "additionalProperties": True,
            },
        },
    )

    def execute_tool(arguments: str | JsonDict | None = None) -> JsonDict:
        """Execute the meta execute tool"""
        # Parse arguments
        if isinstance(arguments, str):
            kwargs = json.loads(arguments)
        else:
            kwargs = arguments or {}

        tool_name = kwargs.get("toolName")
        params = kwargs.get("params", {})

        if not tool_name:
            raise ValueError("toolName is required")

        # Get the tool
        tool = tools_collection.get_tool(tool_name)
        if not tool:
            raise ValueError(f"Tool '{tool_name}' not found")

        # Execute the tool
        return tool.execute(params)

    # Create execute config for the meta tool
    execute_config = ExecuteConfig(
        name=name,
        method="POST",
        url="",  # Utility tools don't make HTTP requests
        headers={},
    )

    # Create a wrapper class that delegates execute to our custom function
    class ToolExecuteTool(StackOneTool):
        """Utility tool for executing discovered tools"""

        def __init__(self) -> None:
            super().__init__(
                description=description,
                parameters=parameters,
                _execute_config=execute_config,
                _api_key="",  # Utility tools don't need API key
                _account_id=None,
            )

        def execute(
            self, arguments: str | JsonDict | None = None, *, options: JsonDict | None = None
        ) -> JsonDict:
            return execute_tool(arguments)

    return ToolExecuteTool()
