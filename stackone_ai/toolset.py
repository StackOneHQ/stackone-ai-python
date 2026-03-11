from __future__ import annotations

import asyncio
import base64
import concurrent.futures
import fnmatch
import json
import logging
import os
import threading
from collections.abc import Coroutine
from dataclasses import dataclass
from importlib import metadata
from typing import Any, Literal, TypedDict, TypeVar

from stackone_ai.constants import DEFAULT_BASE_URL
from stackone_ai.models import (
    ExecuteConfig,
    ParameterLocation,
    StackOneTool,
    ToolParameters,
    Tools,
)
from stackone_ai.semantic_search import (
    SemanticSearchClient,
    SemanticSearchError,
    SemanticSearchResult,
)
from stackone_ai.utils.normalize import _normalize_action_name

logger = logging.getLogger("stackone.tools")

SearchMode = Literal["auto", "semantic", "local"]


class SearchConfig(TypedDict, total=False):
    """Search configuration for the StackOneToolSet constructor.

    When provided as a dict, sets default search options that flow through
    to ``search_tools()``, ``get_search_tool()``, and ``search_action_names()``.
    Per-call options override these defaults.

    When set to ``None``, search is disabled entirely.
    When omitted, defaults to ``{"method": "auto"}``.
    """

    method: SearchMode
    """Search backend to use. Defaults to ``"auto"``."""
    top_k: int
    """Maximum number of tools to return."""
    min_similarity: float
    """Minimum similarity score threshold 0-1."""


_SEARCH_DEFAULT: SearchConfig = {"method": "auto"}

try:
    _SDK_VERSION = metadata.version("stackone-ai")
except metadata.PackageNotFoundError:  # pragma: no cover - best-effort fallback when running from source
    _SDK_VERSION = "dev"
_RPC_PARAMETER_LOCATIONS = {
    "action": ParameterLocation.BODY,
    "body": ParameterLocation.BODY,
    "headers": ParameterLocation.BODY,
    "path": ParameterLocation.BODY,
    "query": ParameterLocation.BODY,
}
_USER_AGENT = f"stackone-ai-python/{_SDK_VERSION}"


T = TypeVar("T")


@dataclass
class _McpToolDefinition:
    name: str
    description: str | None
    input_schema: dict[str, Any]


class ToolsetError(Exception):
    """Base exception for toolset errors"""

    pass


class ToolsetConfigError(ToolsetError):
    """Raised when there is an error in the toolset configuration"""

    pass


class ToolsetLoadError(ToolsetError):
    """Raised when there is an error loading tools"""

    pass


def _run_async(awaitable: Coroutine[Any, Any, T]) -> T:
    """Run a coroutine, even when called from an existing event loop."""

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(awaitable)

    result: dict[str, T] = {}
    error: dict[str, BaseException] = {}

    def runner() -> None:
        try:
            result["value"] = asyncio.run(awaitable)
        except BaseException as exc:  # pragma: no cover - surfaced in caller context
            error["error"] = exc

    thread = threading.Thread(target=runner, daemon=True)
    thread.start()
    thread.join()

    if "error" in error:
        raise error["error"]

    return result["value"]


def _build_auth_header(api_key: str) -> str:
    token = base64.b64encode(f"{api_key}:".encode()).decode()
    return f"Basic {token}"


def _fetch_mcp_tools(endpoint: str, headers: dict[str, str]) -> list[_McpToolDefinition]:
    try:
        from mcp import types as mcp_types  # ty: ignore[unresolved-import]
        from mcp.client.session import ClientSession  # ty: ignore[unresolved-import]
        from mcp.client.streamable_http import streamablehttp_client  # ty: ignore[unresolved-import]
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise ToolsetConfigError(
            "MCP dependencies are required for fetch_tools. Install with 'pip install \"stackone-ai[mcp]\"'."
        ) from exc

    async def _list() -> list[_McpToolDefinition]:
        async with streamablehttp_client(endpoint, headers=headers) as (read_stream, write_stream, _):
            session = ClientSession(
                read_stream,
                write_stream,
                client_info=mcp_types.Implementation(name="stackone-ai-python", version=_SDK_VERSION),
            )
            async with session:
                await session.initialize()
                cursor: str | None = None
                collected: list[_McpToolDefinition] = []
                while True:
                    result = await session.list_tools(cursor)
                    for tool in result.tools:
                        input_schema = tool.inputSchema or {}
                        collected.append(
                            _McpToolDefinition(
                                name=tool.name,
                                description=tool.description,
                                input_schema=dict(input_schema),
                            )
                        )
                    cursor = result.nextCursor
                    if cursor is None:
                        break
                return collected

    return _run_async(_list())


class _StackOneRpcTool(StackOneTool):
    """RPC-backed tool wired to the StackOne actions RPC endpoint."""

    def __init__(
        self,
        *,
        name: str,
        description: str,
        parameters: ToolParameters,
        api_key: str,
        base_url: str,
        account_id: str | None,
    ) -> None:
        execute_config = ExecuteConfig(
            method="POST",
            url=f"{base_url.rstrip('/')}/actions/rpc",
            name=name,
            headers={},
            body_type="json",
            parameter_locations=dict(_RPC_PARAMETER_LOCATIONS),
        )
        super().__init__(
            description=description,
            parameters=parameters,
            _execute_config=execute_config,
            _api_key=api_key,
            _account_id=account_id,
        )

    def execute(
        self, arguments: str | dict[str, Any] | None = None, *, options: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        parsed_arguments = self._parse_arguments(arguments)

        body_payload = self._extract_record(parsed_arguments.pop("body", None))
        headers_payload = self._extract_record(parsed_arguments.pop("headers", None))
        path_payload = self._extract_record(parsed_arguments.pop("path", None))
        query_payload = self._extract_record(parsed_arguments.pop("query", None))

        rpc_body: dict[str, Any] = dict(body_payload or {})
        for key, value in parsed_arguments.items():
            rpc_body[key] = value

        payload: dict[str, Any] = {
            "action": self.name,
            "body": rpc_body,
            "headers": self._build_action_headers(headers_payload),
        }
        if path_payload:
            payload["path"] = path_payload
        if query_payload:
            payload["query"] = query_payload

        return super().execute(payload, options=options)

    def _parse_arguments(self, arguments: str | dict[str, Any] | None) -> dict[str, Any]:
        if arguments is None:
            return {}
        if isinstance(arguments, str):
            parsed = json.loads(arguments)
        else:
            parsed = arguments
        if not isinstance(parsed, dict):
            raise ValueError("Tool arguments must be a JSON object")
        return dict(parsed)

    @staticmethod
    def _extract_record(value: Any) -> dict[str, Any] | None:
        if isinstance(value, dict):
            return dict(value)
        return None

    def _build_action_headers(self, additional_headers: dict[str, Any] | None) -> dict[str, str]:
        headers: dict[str, str] = {}
        account_id = self.get_account_id()
        if account_id:
            headers["x-account-id"] = account_id

        if additional_headers:
            for key, value in additional_headers.items():
                if value is None:
                    continue
                headers[str(key)] = str(value)

        headers.pop("Authorization", None)
        return headers


class SearchTool:
    """Callable search tool that wraps StackOneToolSet.search_tools().

    Designed for agent loops — call it with a query to get Tools back.

    Example::

        toolset = StackOneToolSet()
        search_tool = toolset.get_search_tool()
        tools = search_tool("manage employee records", account_ids=["acc-123"])
    """

    def __init__(self, toolset: StackOneToolSet, config: SearchConfig | None = None) -> None:
        self._toolset = toolset
        self._config: SearchConfig = config or {}

    def __call__(
        self,
        query: str,
        *,
        connector: str | None = None,
        top_k: int | None = None,
        min_similarity: float | None = None,
        account_ids: list[str] | None = None,
        search: SearchMode | None = None,
    ) -> Tools:
        """Search for tools using natural language.

        Args:
            query: Natural language description of needed functionality
            connector: Optional provider/connector filter (e.g., "bamboohr", "slack")
            top_k: Maximum number of tools to return. Overrides constructor default.
            min_similarity: Minimum similarity score threshold 0-1. Overrides constructor default.
            account_ids: Optional account IDs (uses set_accounts() if not provided)
            search: Override the default search mode for this call

        Returns:
            Tools collection with matched tools
        """
        effective_top_k = top_k if top_k is not None else self._config.get("top_k")
        effective_min_sim = (
            min_similarity if min_similarity is not None else self._config.get("min_similarity")
        )
        effective_search = search if search is not None else self._config.get("method", "auto")
        return self._toolset.search_tools(
            query,
            connector=connector,
            top_k=effective_top_k,
            min_similarity=effective_min_sim,
            account_ids=account_ids,
            search=effective_search,
        )


class StackOneToolSet:
    """Main class for accessing StackOne tools"""

    def __init__(
        self,
        api_key: str | None = None,
        account_id: str | None = None,
        base_url: str | None = None,
        search: SearchConfig | None = _SEARCH_DEFAULT,
    ) -> None:
        """Initialize StackOne tools with authentication

        Args:
            api_key: Optional API key. If not provided, will try to get from STACKONE_API_KEY env var
            account_id: Optional account ID
            base_url: Optional base URL override for API requests
            search: Search configuration. Controls default search behavior.
                Omit or pass ``{}`` for defaults (method="auto").
                Pass ``None`` to disable search.
                Pass ``{"method": "semantic", "top_k": 5}`` for custom defaults.
                Per-call options always override these defaults.

        Raises:
            ToolsetConfigError: If no API key is provided or found in environment
        """
        api_key_value = api_key or os.getenv("STACKONE_API_KEY")
        if not api_key_value:
            raise ToolsetConfigError(
                "API key must be provided either through api_key parameter or "
                "STACKONE_API_KEY environment variable"
            )
        self.api_key: str = api_key_value
        self.account_id = account_id
        self.base_url = base_url or DEFAULT_BASE_URL
        self._account_ids: list[str] = []
        self._semantic_client: SemanticSearchClient | None = None
        self._search_config: SearchConfig | None = search

    def set_accounts(self, account_ids: list[str]) -> StackOneToolSet:
        """Set account IDs for filtering tools

        Args:
            account_ids: List of account IDs to filter tools by

        Returns:
            This toolset instance for chaining
        """
        self._account_ids = account_ids
        return self

    def get_search_tool(self, *, search: SearchMode | None = None) -> SearchTool:
        """Get a callable search tool that returns Tools collections.

        Returns a callable that wraps :meth:`search_tools` for use in agent loops.
        The returned tool is directly callable: ``search_tool("query")`` returns
        :class:`Tools`.

        Uses the constructor's search config as defaults. Per-call options override.

        Args:
            search: Override the default search mode. If not provided, uses
                the constructor's search config.

        Returns:
            SearchTool instance

        Example::

            toolset = StackOneToolSet()
            search_tool = toolset.get_search_tool()
            tools = search_tool("manage employee records", account_ids=["acc-123"])
        """
        if self._search_config is None:
            raise ToolsetConfigError(
                "Search is disabled. Initialize StackOneToolSet with a search config to enable."
            )

        config: SearchConfig = {**self._search_config}
        if search is not None:
            config["method"] = search

        return SearchTool(self, config=config)

    def get_meta_tools(
        self,
        *,
        account_ids: list[str] | None = None,
        search: SearchMode | None = None,
        connector: str | None = None,
        top_k: int | None = None,
        min_similarity: float | None = None,
    ) -> Tools:
        """Get LLM-callable meta tools (tool_search + tool_execute) for agent-driven workflows.

        Returns a Tools collection that can be passed directly to any LLM framework.
        The LLM uses tool_search to discover available tools, then tool_execute to run them.

        Args:
            account_ids: Account IDs to scope tool discovery and execution
            search: Search mode ('auto', 'semantic', or 'local')
            connector: Optional connector filter (e.g. 'bamboohr')
            top_k: Maximum number of search results. Defaults to 5.
            min_similarity: Minimum similarity score threshold 0-1

        Returns:
            Tools collection containing tool_search and tool_execute

        Example::

            toolset = StackOneToolSet(account_id="acc-123")
            meta_tools = toolset.get_meta_tools()

            # Pass to OpenAI
            tools = meta_tools.to_openai()

            # Pass to LangChain
            tools = meta_tools.to_langchain()
        """
        if self._search_config is None:
            raise ToolsetConfigError(
                "Search is disabled. Initialize StackOneToolSet with a search config to enable."
            )

        from stackone_ai.meta_tools import MetaToolsOptions, create_meta_tools

        options = MetaToolsOptions(
            account_ids=account_ids,
            search=search,
            connector=connector,
            top_k=top_k,
            min_similarity=min_similarity,
        )
        return create_meta_tools(self, options)

    @property
    def semantic_client(self) -> SemanticSearchClient:
        """Lazy initialization of semantic search client.

        Returns:
            SemanticSearchClient instance configured with the toolset's API key and base URL
        """
        if self._semantic_client is None:
            self._semantic_client = SemanticSearchClient(
                api_key=self.api_key,
                base_url=self.base_url,
            )
        return self._semantic_client

    def _local_search(
        self,
        query: str,
        all_tools: Tools,
        *,
        connector: str | None = None,
        top_k: int | None = None,
        min_similarity: float | None = None,
    ) -> Tools:
        """Run local BM25+TF-IDF search over already-fetched tools."""
        from stackone_ai.local_search import ToolIndex

        available_connectors = all_tools.get_connectors()
        if not available_connectors:
            return Tools([])

        index = ToolIndex(list(all_tools))
        results = index.search(
            query,
            limit=top_k if top_k is not None else 5,
            min_score=min_similarity if min_similarity is not None else 0.0,
        )
        matched_names = [r.name for r in results]
        tool_map = {t.name: t for t in all_tools}
        filter_connectors = {connector.lower()} if connector else available_connectors
        matched_tools = [
            tool_map[name]
            for name in matched_names
            if name in tool_map and name.split("_")[0].lower() in filter_connectors
        ]
        return Tools(matched_tools[:top_k] if top_k is not None else matched_tools)

    def search_tools(
        self,
        query: str,
        *,
        connector: str | None = None,
        top_k: int | None = None,
        min_similarity: float | None = None,
        account_ids: list[str] | None = None,
        search: SearchMode | None = None,
    ) -> Tools:
        """Search for and fetch tools using semantic or local search.

        This method discovers relevant tools based on natural language queries.
        Constructor search config provides defaults; per-call args override.

        Args:
            query: Natural language description of needed functionality
                (e.g., "create employee", "send a message")
            connector: Optional provider/connector filter (e.g., "bamboohr", "slack")
            top_k: Maximum number of tools to return. Overrides constructor default.
            min_similarity: Minimum similarity score threshold 0-1. Overrides constructor default.
            account_ids: Optional account IDs (uses set_accounts() if not provided)
            search: Search backend to use. Overrides constructor default.
                - ``"auto"`` (default): try semantic search first, fall back to local
                  BM25+TF-IDF if the API is unavailable.
                - ``"semantic"``: use only the semantic search API; raises
                  ``SemanticSearchError`` on failure.
                - ``"local"``: use only local BM25+TF-IDF search (no API call to the
                  semantic search endpoint).

        Returns:
            Tools collection with matched tools from linked accounts

        Raises:
            ToolsetConfigError: If search is disabled (``search=None`` in constructor)
            SemanticSearchError: If the API call fails and search is ``"semantic"``

        Examples:
            # Semantic search (default with local fallback)
            tools = toolset.search_tools("manage employee records", top_k=5)

            # Explicit semantic search
            tools = toolset.search_tools("manage employees", search="semantic")

            # Local BM25+TF-IDF search
            tools = toolset.search_tools("manage employees", search="local")

            # Filter by connector
            tools = toolset.search_tools(
                "create time off request",
                connector="bamboohr",
                search="semantic",
            )
        """
        if self._search_config is None:
            raise ToolsetConfigError(
                "Search is disabled. Initialize StackOneToolSet with a search config to enable."
            )

        # Merge constructor defaults with per-call overrides
        effective_search: SearchMode = (
            search if search is not None else self._search_config.get("method", "auto")
        )
        effective_top_k = top_k if top_k is not None else self._search_config.get("top_k")
        effective_min_sim = (
            min_similarity if min_similarity is not None else self._search_config.get("min_similarity")
        )

        all_tools = self.fetch_tools(account_ids=account_ids)
        available_connectors = all_tools.get_connectors()

        if not available_connectors:
            return Tools([])

        # Local-only search — skip semantic API entirely
        if effective_search == "local":
            return self._local_search(
                query, all_tools, connector=connector, top_k=effective_top_k, min_similarity=effective_min_sim
            )

        try:
            # Determine which connectors to search
            if connector:
                connectors_to_search = {connector.lower()} & available_connectors
                if not connectors_to_search:
                    return Tools([])
            else:
                connectors_to_search = available_connectors

            # Search each connector in parallel
            def _search_one(c: str) -> list[SemanticSearchResult]:
                resp = self.semantic_client.search(
                    query=query, connector=c, top_k=effective_top_k, min_similarity=effective_min_sim
                )
                return list(resp.results)

            all_results: list[SemanticSearchResult] = []
            last_error: SemanticSearchError | None = None
            max_workers = min(len(connectors_to_search), 10)
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
                futures = {pool.submit(_search_one, c): c for c in connectors_to_search}
                for future in concurrent.futures.as_completed(futures):
                    try:
                        all_results.extend(future.result())
                    except SemanticSearchError as e:
                        last_error = e

            # If ALL connector searches failed, re-raise to trigger fallback
            if not all_results and last_error is not None:
                raise last_error

            # Sort by score, apply top_k
            all_results.sort(key=lambda r: r.similarity_score, reverse=True)
            if effective_top_k is not None:
                all_results = all_results[:effective_top_k]

            if not all_results:
                return Tools([])

            # Match back to fetched tool definitions
            action_names = {_normalize_action_name(r.action_name) for r in all_results}
            matched_tools = [t for t in all_tools if t.name in action_names]

            # Sort matched tools by semantic search score order
            action_order = {_normalize_action_name(r.action_name): i for i, r in enumerate(all_results)}
            matched_tools.sort(key=lambda t: action_order.get(t.name, float("inf")))

            return Tools(matched_tools)

        except SemanticSearchError as e:
            if effective_search == "semantic":
                raise

            logger.warning("Semantic search failed (%s), falling back to local BM25+TF-IDF search", e)
            return self._local_search(
                query, all_tools, connector=connector, top_k=effective_top_k, min_similarity=effective_min_sim
            )

    def search_action_names(
        self,
        query: str,
        *,
        connector: str | None = None,
        account_ids: list[str] | None = None,
        top_k: int | None = None,
        min_similarity: float | None = None,
    ) -> list[SemanticSearchResult]:
        """Search for action names without fetching tools.

        Useful when you need to inspect search results before fetching,
        or when building custom filtering logic.

        Args:
            query: Natural language description of needed functionality
            connector: Optional provider/connector filter (single connector)
            account_ids: Optional account IDs to scope results to connectors
                available in those accounts (uses set_accounts() if not provided).
                When provided, results are filtered to only matching connectors.
            top_k: Maximum number of results. If None, uses the backend default.
            min_similarity: Minimum similarity score threshold 0-1. If not provided,
                the server uses its default.

        Returns:
            List of SemanticSearchResult with action names, scores, and metadata.
            Versioned API names are normalized to MCP format but results are NOT
            deduplicated — multiple API versions of the same action may appear
            with their individual scores.

        Examples:
            # Lightweight: inspect results before fetching
            results = toolset.search_action_names("manage employees")
            for r in results:
                print(f"{r.action_name}: {r.similarity_score:.2f}")

            # Account-scoped: only results for connectors in linked accounts
            results = toolset.search_action_names(
                "create employee",
                account_ids=["acc-123"],
                top_k=5
            )

            # Then fetch specific high-scoring actions
            selected = [r.action_name for r in results if r.similarity_score > 0.7]
            tools = toolset.fetch_tools(actions=selected)
        """
        if self._search_config is None:
            raise ToolsetConfigError(
                "Search is disabled. Initialize StackOneToolSet with search config to enable."
            )

        # Merge constructor defaults with per-call overrides
        effective_top_k = top_k if top_k is not None else self._search_config.get("top_k")
        effective_min_sim = (
            min_similarity if min_similarity is not None else self._search_config.get("min_similarity")
        )

        # Resolve available connectors from account_ids (same pattern as search_tools)
        available_connectors: set[str] | None = None
        effective_account_ids = account_ids or self._account_ids
        if effective_account_ids:
            all_tools = self.fetch_tools(account_ids=effective_account_ids)
            available_connectors = all_tools.get_connectors()
            if not available_connectors:
                return []

        try:
            if available_connectors:
                # Parallel per-connector search (only user's connectors)
                if connector:
                    connectors_to_search = {connector.lower()} & available_connectors
                else:
                    connectors_to_search = available_connectors

                def _search_one(c: str) -> list[SemanticSearchResult]:
                    try:
                        resp = self.semantic_client.search(
                            query=query,
                            connector=c,
                            top_k=effective_top_k,
                            min_similarity=effective_min_sim,
                        )
                        return list(resp.results)
                    except SemanticSearchError:
                        return []

                all_results: list[SemanticSearchResult] = []
                if connectors_to_search:
                    max_workers = min(len(connectors_to_search), 10)
                    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
                        futures = [pool.submit(_search_one, c) for c in connectors_to_search]
                        for future in concurrent.futures.as_completed(futures):
                            all_results.extend(future.result())
            else:
                # No account filtering — single global search
                response = self.semantic_client.search(
                    query=query,
                    connector=connector,
                    top_k=effective_top_k,
                    min_similarity=effective_min_sim,
                )
                all_results = list(response.results)

        except SemanticSearchError as e:
            logger.warning("Semantic search failed: %s", e)
            return []

        # Sort by score, normalize action names
        all_results.sort(key=lambda r: r.similarity_score, reverse=True)
        normalized: list[SemanticSearchResult] = []
        for r in all_results:
            normalized.append(
                SemanticSearchResult(
                    action_name=_normalize_action_name(r.action_name),
                    connector_key=r.connector_key,
                    similarity_score=r.similarity_score,
                    label=r.label,
                    description=r.description,
                )
            )
        return normalized[:effective_top_k] if effective_top_k is not None else normalized

    def _filter_by_provider(self, tool_name: str, providers: list[str]) -> bool:
        """Check if a tool name matches any of the provider filters

        Args:
            tool_name: Name of the tool to check
            providers: List of provider names (case-insensitive)

        Returns:
            True if the tool matches any provider, False otherwise
        """
        # Extract provider from tool name (assuming format: provider_action)
        provider = tool_name.split("_")[0].lower()
        provider_set = {p.lower() for p in providers}
        return provider in provider_set

    def _filter_by_action(self, tool_name: str, actions: list[str]) -> bool:
        """Check if a tool name matches any of the action patterns

        Args:
            tool_name: Name of the tool to check
            actions: List of action patterns (supports glob patterns)

        Returns:
            True if the tool matches any action pattern, False otherwise
        """
        return any(fnmatch.fnmatch(tool_name, pattern) for pattern in actions)

    def fetch_tools(
        self,
        *,
        account_ids: list[str] | None = None,
        providers: list[str] | None = None,
        actions: list[str] | None = None,
    ) -> Tools:
        """Fetch tools with optional filtering by account IDs, providers, and actions

        Args:
            account_ids: Optional list of account IDs to filter by.
                If not provided, uses accounts set via set_accounts()
            providers: Optional list of provider names (e.g., ['hibob', 'bamboohr']).
                Case-insensitive matching.
            actions: Optional list of action patterns with glob support
                (e.g., ['*_list_employees', 'hibob_create_employees'])

        Returns:
            Collection of tools matching the filter criteria

        Raises:
            ToolsetLoadError: If there is an error loading the tools

        Examples:
            # Filter by account IDs
            tools = toolset.fetch_tools(account_ids=['123', '456'])

            # Filter by providers
            tools = toolset.fetch_tools(providers=['hibob', 'bamboohr'])

            # Filter by actions with glob patterns
            tools = toolset.fetch_tools(actions=['*_list_employees'])

            # Combine filters
            tools = toolset.fetch_tools(
                account_ids=['123'],
                providers=['hibob'],
                actions=['*_list_*']
            )

            # Use set_accounts() for account filtering
            toolset.set_accounts(['123', '456'])
            tools = toolset.fetch_tools()
        """
        try:
            effective_account_ids = account_ids or self._account_ids
            if not effective_account_ids and self.account_id:
                effective_account_ids = [self.account_id]

            if effective_account_ids:
                account_scope: list[str | None] = list(dict.fromkeys(effective_account_ids))
            else:
                account_scope = [None]

            endpoint = f"{self.base_url.rstrip('/')}/mcp"
            all_tools: list[StackOneTool] = []

            for account in account_scope:
                headers = self._build_mcp_headers(account)
                catalog = _fetch_mcp_tools(endpoint, headers)
                for tool_def in catalog:
                    all_tools.append(self._create_rpc_tool(tool_def, account))

            if providers:
                all_tools = [tool for tool in all_tools if self._filter_by_provider(tool.name, providers)]

            if actions:
                all_tools = [tool for tool in all_tools if self._filter_by_action(tool.name, actions)]

            return Tools(all_tools)

        except ToolsetError:
            raise
        except Exception as exc:  # pragma: no cover - unexpected runtime errors
            raise ToolsetLoadError(f"Error fetching tools: {exc}") from exc

    def _build_mcp_headers(self, account_id: str | None) -> dict[str, str]:
        headers = {
            "Authorization": _build_auth_header(self.api_key),
            "User-Agent": _USER_AGENT,
        }
        if account_id:
            headers["x-account-id"] = account_id
        return headers

    def _create_rpc_tool(self, tool_def: _McpToolDefinition, account_id: str | None) -> StackOneTool:
        schema = tool_def.input_schema or {}
        parameters = ToolParameters(
            type=str(schema.get("type") or "object"),
            properties=self._normalize_schema_properties(schema),
        )
        return _StackOneRpcTool(
            name=tool_def.name,
            description=tool_def.description or "",
            parameters=parameters,
            api_key=self.api_key,
            base_url=self.base_url,
            account_id=account_id,
        )

    def _normalize_schema_properties(self, schema: dict[str, Any]) -> dict[str, Any]:
        properties = schema.get("properties")
        if not isinstance(properties, dict):
            return {}

        required_fields = {str(name) for name in schema.get("required", [])}
        normalized: dict[str, Any] = {}

        for name, details in properties.items():
            if isinstance(details, dict):
                prop = dict(details)
            else:
                prop = {"description": str(details)}

            if name in required_fields:
                prop.setdefault("nullable", False)
            else:
                prop.setdefault("nullable", True)

            normalized[str(name)] = prop

        return normalized
