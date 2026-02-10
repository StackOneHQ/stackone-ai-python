from __future__ import annotations

import asyncio
import base64
import fnmatch
import json
import logging
import os
import re
import threading
from collections.abc import Coroutine
from dataclasses import dataclass
from importlib import metadata
from typing import Any, TypeVar

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

logger = logging.getLogger("stackone.tools")

try:
    _SDK_VERSION = metadata.version("stackone-ai")
except metadata.PackageNotFoundError:  # pragma: no cover - best-effort fallback when running from source
    _SDK_VERSION = "dev"

DEFAULT_BASE_URL = "https://api.stackone.com"
_RPC_PARAMETER_LOCATIONS = {
    "action": ParameterLocation.BODY,
    "body": ParameterLocation.BODY,
    "headers": ParameterLocation.BODY,
    "path": ParameterLocation.BODY,
    "query": ParameterLocation.BODY,
}
_USER_AGENT = f"stackone-ai-python/{_SDK_VERSION}"

_VERSIONED_ACTION_RE = re.compile(r"^[a-z][a-z0-9]*_\d+(?:\.\d+)+_(.+)_global$")


def _normalize_action_name(action_name: str) -> str:
    """Convert semantic search API action name to MCP tool name.

    API:  'calendly_1.0.0_calendly_create_scheduling_link_global'
    MCP:  'calendly_create_scheduling_link'
    """
    match = _VERSIONED_ACTION_RE.match(action_name)
    return match.group(1) if match else action_name


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


class StackOneToolSet:
    """Main class for accessing StackOne tools"""

    def __init__(
        self,
        api_key: str | None = None,
        account_id: str | None = None,
        base_url: str | None = None,
    ) -> None:
        """Initialize StackOne tools with authentication

        Args:
            api_key: Optional API key. If not provided, will try to get from STACKONE_API_KEY env var
            account_id: Optional account ID
            base_url: Optional base URL override for API requests

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

    def set_accounts(self, account_ids: list[str]) -> StackOneToolSet:
        """Set account IDs for filtering tools

        Args:
            account_ids: List of account IDs to filter tools by

        Returns:
            This toolset instance for chaining
        """
        self._account_ids = account_ids
        return self

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

    def search_tools(
        self,
        query: str,
        *,
        connector: str | None = None,
        top_k: int = 10,
        min_score: float = 0.0,
        account_ids: list[str] | None = None,
        fallback_to_local: bool = True,
    ) -> Tools:
        """Search for and fetch tools using semantic search.

        This method uses the StackOne semantic search API to find relevant tools
        based on natural language queries. It optimizes results by filtering to
        only connectors available in linked accounts.

        Args:
            query: Natural language description of needed functionality
                (e.g., "create employee", "send a message")
            connector: Optional provider/connector filter (e.g., "bamboohr", "slack")
            top_k: Maximum number of tools to return (default: 10)
            min_score: Minimum similarity score threshold 0-1 (default: 0.0)
            account_ids: Optional account IDs (uses set_accounts() if not provided)
            fallback_to_local: If True, fall back to local BM25+TF-IDF search on API failure

        Returns:
            Tools collection with semantically matched tools from linked accounts

        Raises:
            SemanticSearchError: If the API call fails and fallback_to_local is False

        Examples:
            # Basic semantic search
            tools = toolset.search_tools("manage employee records", top_k=5)

            # Filter by connector
            tools = toolset.search_tools(
                "create time off request",
                connector="bamboohr",
                min_score=0.5
            )

            # With account filtering
            tools = toolset.search_tools(
                "send message",
                account_ids=["acc-123"],
                top_k=3
            )
        """
        try:
            # Step 1: Fetch all tools to get available connectors from linked accounts
            all_tools = self.fetch_tools(account_ids=account_ids)
            available_connectors = all_tools.get_connectors()

            if not available_connectors:
                return Tools([])

            # Step 2: Fetch results from semantic API, then filter client-side
            response = self.semantic_client.search(
                query=query,
                connector=connector,
            )

            # Step 3: Filter results to only available connectors and min_score
            filtered_results = [
                r
                for r in response.results
                if r.connector_key.lower() in available_connectors and r.similarity_score >= min_score
            ]

            # Step 3b: If not enough results, make per-connector calls for missing connectors
            if len(filtered_results) < top_k and not connector:
                found_connectors = {r.connector_key.lower() for r in filtered_results}
                missing_connectors = available_connectors - found_connectors
                for missing in missing_connectors:
                    if len(filtered_results) >= top_k:
                        break
                    try:
                        extra = self.semantic_client.search(query=query, connector=missing, top_k=top_k)
                        for r in extra.results:
                            if r.similarity_score >= min_score and r.action_name not in {
                                fr.action_name for fr in filtered_results
                            }:
                                filtered_results.append(r)
                                if len(filtered_results) >= top_k:
                                    break
                    except SemanticSearchError:
                        continue

                # Re-sort by score after merging results from multiple calls
                filtered_results.sort(key=lambda r: r.similarity_score, reverse=True)

            # Deduplicate by normalized MCP name (keep highest score first, already sorted)
            seen_names: set[str] = set()
            deduped: list[SemanticSearchResult] = []
            for r in filtered_results:
                norm = _normalize_action_name(r.action_name)
                if norm not in seen_names:
                    seen_names.add(norm)
                    deduped.append(r)
            filtered_results = deduped[:top_k]

            if not filtered_results:
                return Tools([])

            # Step 4: Get matching tools from already-fetched tools
            action_names = {_normalize_action_name(r.action_name) for r in filtered_results}
            matched_tools = [t for t in all_tools if t.name in action_names]

            # Sort matched tools by semantic search score order
            action_order = {_normalize_action_name(r.action_name): i for i, r in enumerate(filtered_results)}
            matched_tools.sort(key=lambda t: action_order.get(t.name, float("inf")))

            return Tools(matched_tools)

        except SemanticSearchError as e:
            if not fallback_to_local:
                raise

            logger.warning("Semantic search failed (%s), falling back to local BM25+TF-IDF search", e)
            utility = all_tools.utility_tools()
            search_tool = utility.get_tool("tool_search")

            if search_tool:
                result = search_tool.execute(
                    {
                        "query": query,
                        "limit": top_k * 3,  # Over-fetch to account for connector filtering
                        "minScore": min_score,
                    }
                )
                matched_names = [t["name"] for t in result.get("tools", [])]
                # Filter by available connectors and preserve relevance order
                tool_map = {t.name: t for t in all_tools}
                filter_connectors = {connector.lower()} if connector else available_connectors
                matched_tools = [
                    tool_map[name]
                    for name in matched_names
                    if name in tool_map and name.split("_")[0].lower() in filter_connectors
                ]
                return Tools(matched_tools[:top_k])

            return all_tools

    def search_action_names(
        self,
        query: str,
        *,
        connector: str | None = None,
        account_ids: list[str] | None = None,
        top_k: int = 10,
        min_score: float = 0.0,
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
            top_k: Maximum number of results (default: 10)
            min_score: Minimum similarity score threshold 0-1 (default: 0.0)

        Returns:
            List of SemanticSearchResult with action names, scores, and metadata

        Examples:
            # Lightweight: inspect results before fetching
            results = toolset.search_action_names("manage employees", top_k=10)
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
        # Resolve available connectors from account_ids (same pattern as search_tools)
        available_connectors: set[str] | None = None
        effective_account_ids = account_ids or self._account_ids
        if effective_account_ids:
            all_tools = self.fetch_tools(account_ids=effective_account_ids)
            available_connectors = all_tools.get_connectors()
            if not available_connectors:
                return []

        try:
            response = self.semantic_client.search(
                query=query,
                connector=connector,
                top_k=None if available_connectors else top_k,
            )
        except SemanticSearchError as e:
            logger.warning("Semantic search failed: %s", e)
            return []

        # Filter by min_score
        results = [r for r in response.results if r.similarity_score >= min_score]

        # Filter by available connectors if resolved from accounts
        if available_connectors:
            connector_set = {c.lower() for c in available_connectors}
            results = [r for r in results if r.connector_key.lower() in connector_set]

            # If not enough results, make per-connector calls for missing connectors
            if len(results) < top_k and not connector:
                found_connectors = {r.connector_key.lower() for r in results}
                missing_connectors = connector_set - found_connectors
                for missing in missing_connectors:
                    if len(results) >= top_k:
                        break
                    try:
                        extra = self.semantic_client.search(query=query, connector=missing, top_k=top_k)
                        for r in extra.results:
                            if r.similarity_score >= min_score and r.action_name not in {
                                er.action_name for er in results
                            }:
                                results.append(r)
                                if len(results) >= top_k:
                                    break
                    except SemanticSearchError:
                        continue

                # Re-sort by score after merging
                results.sort(key=lambda r: r.similarity_score, reverse=True)

        # Normalize and deduplicate by MCP name (keep highest score first)
        seen: set[str] = set()
        normalized: list[SemanticSearchResult] = []
        for r in results:
            norm_name = _normalize_action_name(r.action_name)
            if norm_name not in seen:
                seen.add(norm_name)
                normalized.append(
                    SemanticSearchResult(
                        action_name=norm_name,
                        connector_key=r.connector_key,
                        similarity_score=r.similarity_score,
                        label=r.label,
                        description=r.description,
                    )
                )
        return normalized[:top_k]

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
