from __future__ import annotations

import asyncio
import base64
import fnmatch
import json
import os
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

    def set_accounts(self, account_ids: list[str]) -> StackOneToolSet:
        """Set account IDs for filtering tools

        Args:
            account_ids: List of account IDs to filter tools by

        Returns:
            This toolset instance for chaining
        """
        self._account_ids = account_ids
        return self

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
