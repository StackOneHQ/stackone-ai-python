"""Meta tools for dynamic tool discovery and execution"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import bm25s
import numpy as np
from pydantic import BaseModel

from stackone_ai.models import ExecuteConfig, JsonDict, StackOneTool, ToolParameters

if TYPE_CHECKING:
    from stackone_ai.models import Tools


class MetaToolSearchResult(BaseModel):
    """Result from meta_search_tools"""

    name: str
    description: str
    score: float


class ToolIndex:
    """BM25-based tool search index"""

    def __init__(self, tools: list[StackOneTool]) -> None:
        self.tools = tools
        self.tool_map = {tool.name: tool for tool in tools}

        # Prepare corpus for BM25
        corpus = []
        self.tool_names = []

        for tool in tools:
            # Extract category and action from tool name
            parts = tool.name.split("_")
            category = parts[0] if parts else ""

            # Extract action types
            action_types = ["create", "update", "delete", "get", "list", "search"]
            actions = [p for p in parts if p in action_types]

            # Combine name, description, category and tags for indexing
            doc_text = " ".join(
                [
                    tool.name,
                    tool.description,
                    category,
                    " ".join(parts),
                    " ".join(actions),
                ]
            )

            corpus.append(doc_text)
            self.tool_names.append(tool.name)

        # Create BM25 index
        self.retriever = bm25s.BM25()
        # Tokenize without stemming for simplicity
        corpus_tokens = bm25s.tokenize(corpus, stemmer=None, show_progress=False)
        self.retriever.index(corpus_tokens)

    def search(self, query: str, limit: int = 5, min_score: float = 0.0) -> list[MetaToolSearchResult]:
        """Search for relevant tools using BM25

        Args:
            query: Natural language query
            limit: Maximum number of results
            min_score: Minimum relevance score (0-1)

        Returns:
            List of search results sorted by relevance
        """
        # Tokenize query
        query_tokens = bm25s.tokenize([query], stemmer=None, show_progress=False)

        # Search with BM25
        results, scores = self.retriever.retrieve(query_tokens, k=min(limit * 2, len(self.tools)))

        # Process results
        search_results = []
        # TODO: Add strict=False when Python 3.9 support is dropped
        for idx, score in zip(results[0], scores[0]):
            if score < min_score:
                continue

            tool_name = self.tool_names[idx]
            tool = self.tool_map[tool_name]

            # Normalize score to 0-1 range
            normalized_score = float(1 / (1 + np.exp(-score / 10)))

            search_results.append(
                MetaToolSearchResult(
                    name=tool.name,
                    description=tool.description,
                    score=normalized_score,
                )
            )

            if len(search_results) >= limit:
                break

        return search_results


def create_meta_search_tools(index: ToolIndex) -> StackOneTool:
    """Create the meta_search_tools tool

    Args:
        index: Tool search index

    Returns:
        Meta tool for searching relevant tools
    """
    name = "meta_search_tools"
    description = (
        "Searches for relevant tools based on a natural language query. "
        "This tool should be called first to discover available tools before executing them."
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
            },
            "minScore": {
                "type": "number",
                "description": "Minimum relevance score (0-1) to filter results (default: 0.0)",
                "default": 0.0,
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
        limit = int(kwargs.get("limit", 5))
        min_score = float(kwargs.get("minScore", 0.0))

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
        url="",  # Meta tools don't make HTTP requests
        headers={},
    )

    # Create a wrapper class that delegates execute to our custom function
    class MetaSearchTool(StackOneTool):
        """Meta tool for searching relevant tools"""

        def __init__(self) -> None:
            super().__init__(
                description=description,
                parameters=parameters,
                _execute_config=execute_config,
                _api_key="",  # Meta tools don't need API key
                _account_id=None,
            )

        def execute(self, arguments: str | JsonDict | None = None) -> JsonDict:
            return execute_filter(arguments)

    return MetaSearchTool()


def create_meta_execute_tool(tools_collection: Tools) -> StackOneTool:
    """Create the meta_execute_tool

    Args:
        tools_collection: Collection of tools to execute from

    Returns:
        Meta tool for executing discovered tools
    """
    name = "meta_execute_tool"
    description = (
        "Executes a tool by name with the provided parameters. "
        "Use this after discovering tools with meta_search_tools."
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
        url="",  # Meta tools don't make HTTP requests
        headers={},
    )

    # Create a wrapper class that delegates execute to our custom function
    class MetaExecuteTool(StackOneTool):
        """Meta tool for executing discovered tools"""

        def __init__(self) -> None:
            super().__init__(
                description=description,
                parameters=parameters,
                _execute_config=execute_config,
                _api_key="",  # Meta tools don't need API key
                _account_id=None,
            )

        def execute(self, arguments: str | JsonDict | None = None) -> JsonDict:
            return execute_tool(arguments)

    return MetaExecuteTool()
