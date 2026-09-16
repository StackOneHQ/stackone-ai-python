"""The StackOne toolset: fetches the served tool catalog and exposes it to agent frameworks."""

from __future__ import annotations

import concurrent.futures
import fnmatch
import logging
import os
from typing import Any

import httpx

from stackone_ai.tools import (
    MCP_PARAM_STYLE,
    USER_AGENT,
    McpToolDefinition,
    StackOneMcpTool,
    StackOneRpcTool,
    StackOneTool,
    Tools,
    build_auth_header,
    fetch_mcp_tools,
)
from stackone_ai.types import (
    DEFAULT_BASE_URL,
    ExecuteToolsConfig,
    JsonDict,
    StackOneAPIError,
    StackOneError,
    ToolMode,
    ToolParameters,
    ToolsetConfigError,
    ToolsetError,
    ToolsetLoadError,
)

logger = logging.getLogger("stackone.tools")


class StackOneToolSet:
    """Main class for accessing StackOne tools.

    The toolset is a thin client over the served catalog: it lists tools from the
    MCP endpoint and executes them against the actions RPC endpoint. It does not
    rewrite, filter or invent schemas.
    """

    def __init__(
        self,
        api_key: str | None = None,
        account_id: str | None = None,
        base_url: str | None = None,
        execute: ExecuteToolsConfig | None = None,
        timeout: float | None = None,
        tool_mode: ToolMode | None = None,
    ) -> None:
        """Initialize StackOne tools with authentication

        Args:
            api_key: Optional API key. If not provided, will try to get from STACKONE_API_KEY env var
            account_id: Optional account ID
            base_url: Optional base URL override for API requests
            execute: Execution configuration. Controls default account scoping
                for tool execution. Pass ``{"account_ids": ["acc-1"]}`` to scope
                tools to specific accounts.
            timeout: Request timeout in seconds for tool execution HTTP calls.
                Default: 60. Takes precedence over ``execute.timeout`` if set.
                Increase for slow providers (e.g. Workday).
            tool_mode: How the endpoint lists tools. ``"search_execute"`` returns
                two meta tools per connector instead of one tool per action,
                keeping the catalog small enough for a model's context.

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
        self._account_ids: list[str] = execute.get("account_ids", []) if execute else []
        self._execute_config: ExecuteToolsConfig | None = execute
        execute_timeout = execute.get("timeout") if execute else None
        self._timeout: float = timeout if timeout is not None else (execute_timeout or 60.0)
        self._catalog_cache: dict[tuple[Any, ...], Tools] = {}
        self._discovered_account_ids: list[str] | None = None
        self._tool_mode: ToolMode | None = tool_mode

    def set_accounts(self, account_ids: list[str]) -> StackOneToolSet:
        """Set account IDs for filtering tools

        Returns:
            This toolset instance for chaining
        """
        self._account_ids = account_ids
        self.clear_catalog_cache()
        return self

    def clear_catalog_cache(self) -> None:
        """Invalidate the cached tool catalog.

        Call when linked accounts change outside of ``set_accounts`` or when
        you need to force a fresh fetch from the StackOne MCP endpoint.
        """
        self._catalog_cache.clear()
        self._discovered_account_ids = None

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
            tools = toolset.fetch_tools(account_ids=['123', '456'])
            tools = toolset.fetch_tools(providers=['hibob', 'bamboohr'])
            tools = toolset.fetch_tools(actions=['*_list_employees'])
        """
        try:
            effective_account_ids = account_ids or self._account_ids
            if not effective_account_ids and self.account_id:
                effective_account_ids = [self.account_id]
            if not effective_account_ids:
                effective_account_ids = self._discover_account_ids()

            account_scope: list[str | None] = list(dict.fromkeys(effective_account_ids))

            cache_key = (
                tuple(sorted(account_scope, key=lambda a: (a is None, a))),
                tuple(sorted(p.lower() for p in providers)) if providers else None,
                tuple(sorted(actions)) if actions else None,
                self._tool_mode,
            )
            cached = self._catalog_cache.get(cache_key)
            if cached is not None:
                return cached

            endpoint = f"{self.base_url.rstrip('/')}/mcp?param-style={MCP_PARAM_STYLE}"
            if self._tool_mode:
                endpoint = f"{endpoint}&tool-mode={self._tool_mode}"

            def _fetch_for_account(account: str | None) -> list[StackOneTool]:
                headers = self._build_mcp_headers(account)
                catalog = fetch_mcp_tools(endpoint, headers)
                return [self._create_tool(tool_def, account, endpoint, headers) for tool_def in catalog]

            all_tools: list[StackOneTool] = []
            if len(account_scope) == 1:
                all_tools.extend(_fetch_for_account(account_scope[0]))
            else:
                max_workers = min(len(account_scope), 10)
                failures: list[str] = []
                with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
                    futures = {pool.submit(_fetch_for_account, acc): acc for acc in account_scope}
                    for future, account in futures.items():
                        try:
                            all_tools.extend(future.result())
                        except Exception as exc:
                            # One unusable account must not cost the caller every other
                            # account's tools; report which one, keep the rest.
                            failures.append(f"{account}: {exc}")
                if failures and not all_tools:
                    raise ToolsetLoadError("No account returned tools. " + " | ".join(failures))
                for failure in failures:
                    logger.warning("Skipping account that failed to list tools — %s", failure)

            if providers:
                all_tools = [tool for tool in all_tools if self._filter_by_provider(tool.name, providers)]

            if actions:
                all_tools = [tool for tool in all_tools if self._filter_by_action(tool.name, actions)]

            result = Tools(all_tools)
            self._catalog_cache[cache_key] = result
            return result

        except ToolsetError:
            raise
        except Exception as exc:  # pragma: no cover - unexpected runtime errors
            raise ToolsetLoadError(f"Error fetching tools: {exc}") from exc

    def _meta_tools(self, suffix: str, account_ids: list[str] | None = None) -> list[StackOneTool]:
        """The server's per-connector meta tools, regardless of this toolset's mode."""
        previous = self._tool_mode
        self._tool_mode = "search_execute"
        try:
            tools = self.fetch_tools(account_ids=account_ids)
        finally:
            self._tool_mode = previous
        return [tool for tool in tools if tool.name.endswith(suffix)]

    def search(
        self,
        query: str,
        *,
        top_k: int = 10,
        account_ids: list[str] | None = None,
    ) -> list[JsonDict]:
        """Find actions matching a natural-language query.

        Searches every linked connector and merges the results, so the catalog
        never has to be loaded into a model's context.

        Args:
            query: What you want to do, e.g. "list recent comments".
            top_k: Maximum results per connector.
            account_ids: Restrict to these accounts. Defaults to all active ones.

        Returns:
            Action dicts carrying at least ``action_id`` and ``description``.
        """
        results: list[JsonDict] = []
        failures: list[str] = []
        tools = self._meta_tools("_search_actions", account_ids)
        for tool in tools:
            try:
                found = tool.execute({"query": query, "top_k": top_k})
            except StackOneError as exc:
                # One connector erroring must not hide every other connector's
                # results — the same rule fetch_tools() applies to listing.
                failures.append(f"{tool.name}: {exc}")
                continue
            results.extend(found.get("actions", []))
        if failures and not results:
            raise ToolsetLoadError("No connector returned results. " + " | ".join(failures))
        for failure in failures:
            logger.warning("Skipping connector that failed to search — %s", failure)
        return results

    def execute(
        self,
        action_id: str,
        arguments: JsonDict | None = None,
        *,
        account_ids: list[str] | None = None,
    ) -> JsonDict:
        """Execute an action by id, as returned by :meth:`search`.

        Always runs through the connector's ``_execute_action`` meta tool, so
        ``arguments`` is the nested envelope every action's ``example_request``
        shows — ``{"query": {...}, "path": {...}, "body": {...}}``. The flat,
        prefixed form belongs to ``fetch_tools()`` tools, whose own served schema
        names the keys; routing by whether an id happened to be in the catalog
        would make the argument shape depend on something the caller cannot see.

        Raises:
            ToolsetLoadError: If no connector matches.
        """
        connector = action_id.split("_")[0].lower()
        for tool in self._meta_tools("_execute_action", account_ids):
            if tool.name.split("_")[0].lower() == connector:
                return tool.execute({"action_id": action_id, **(arguments or {})})

        raise ToolsetLoadError(
            f'No connector found for "{action_id}". Use search() to discover valid action ids.'
        )

    def openai(self, *, account_ids: list[str] | None = None) -> list[JsonDict]:
        """Get tools in OpenAI function calling format."""
        return self.fetch_tools(account_ids=account_ids).to_openai()

    def langchain(self, *, account_ids: list[str] | None = None) -> Any:
        """Get tools in LangChain format."""
        return self.fetch_tools(account_ids=account_ids).to_langchain()

    def pydantic_ai(self, *, account_ids: list[str] | None = None) -> list[Any]:
        """Get tools as Pydantic AI ``Tool`` instances.

        Requires ``stackone-ai[pydantic-ai]`` (installs ``pydantic-ai-slim``).
        """
        return self.fetch_tools(account_ids=account_ids).to_pydantic_ai()

    def _filter_by_provider(self, tool_name: str, providers: list[str]) -> bool:
        """Whether a tool belongs to one of the given providers (case-insensitive)."""
        connector = tool_name.split("_")[0].lower()
        return connector in {provider.lower() for provider in providers}

    def _filter_by_action(self, tool_name: str, actions: list[str]) -> bool:
        """Whether a tool name matches any of the given glob patterns."""
        return any(fnmatch.fnmatch(tool_name, pattern) for pattern in actions)

    def fetch_accounts(self) -> list[JsonDict]:
        """List the accounts linked to this API key.

        Each entry carries at least ``id``, ``provider`` and ``status``. Only
        accounts with ``status == "active"`` can serve tools.
        """
        url = f"{self.base_url.rstrip('/')}/accounts"
        response = httpx.get(
            url,
            headers={
                "Authorization": build_auth_header(self.api_key),
                "User-Agent": USER_AGENT,
            },
            timeout=self._timeout,
        )
        if response.is_error:
            # Without this the catch-all in fetch_tools flattens it to a message and
            # the status is lost, so a caller cannot tell 401 from 429.
            raise StackOneAPIError(
                f"Listing accounts at {url} failed with "
                f"{response.status_code} {response.reason_phrase}: {response.text.strip()}",
                response.status_code,
                response.text,
            )
        body = response.json()
        accounts = body.get("data", body) if isinstance(body, dict) else body
        return list(accounts)

    def _discover_account_ids(self) -> list[str]:
        """List the linked accounts this API key can use.

        The MCP endpoint requires an ``x-account-id`` on every request, so an API
        key on its own is not enough to list tools. Rather than make every caller
        supply one, ask the API which accounts the key has.

        Raises:
            ToolsetConfigError: If the key has no accounts, or none are usable.
        """
        if self._discovered_account_ids is not None:
            return self._discovered_account_ids

        accounts = self.fetch_accounts()

        active = [a["id"] for a in accounts if a.get("status") == "active" and a.get("id")]
        if not active:
            if not accounts:
                raise ToolsetConfigError(
                    "This API key has no linked accounts. Link one in the StackOne "
                    "dashboard, or pass account_id explicitly."
                )
            listed = ", ".join(f"{a.get('provider')} ({a.get('status')})" for a in accounts)
            raise ToolsetConfigError(
                f"None of this API key's {len(accounts)} linked accounts are active: {listed}. "
                "Re-link them in the StackOne dashboard, or pass account_id explicitly."
            )

        self._discovered_account_ids = active
        return active

    def _build_mcp_headers(self, account_id: str | None) -> dict[str, str]:
        headers = {
            "Authorization": build_auth_header(self.api_key),
            "User-Agent": USER_AGENT,
        }
        if account_id:
            headers["x-account-id"] = account_id
        return headers

    def _create_tool(
        self,
        tool_def: McpToolDefinition,
        account_id: str | None,
        endpoint: str,
        headers: dict[str, str],
    ) -> StackOneTool:
        """Build an executable tool from a served catalog entry.

        In ``search_execute`` mode the served tools are MCP meta tools with no
        action behind them on ``/actions/rpc``, so they are executed over
        ``tools/call`` instead.
        """
        schema = tool_def.input_schema or {}
        parameters = ToolParameters(
            type=str(schema.get("type") or "object"),
            properties=self._normalize_schema_properties(schema),
        )
        if self._tool_mode == "search_execute":
            return StackOneMcpTool(
                name=tool_def.name,
                description=tool_def.description or "",
                parameters=parameters,
                api_key=self.api_key,
                endpoint=endpoint,
                headers=headers,
                account_id=account_id,
                timeout=self._timeout,
            )
        return StackOneRpcTool(
            name=tool_def.name,
            description=tool_def.description or "",
            parameters=parameters,
            api_key=self.api_key,
            base_url=self.base_url,
            account_id=account_id,
            timeout=self._timeout,
        )

    def _normalize_schema_properties(self, schema: dict[str, Any]) -> dict[str, Any]:
        """Mirror the served schema's properties, recording requiredness per property.

        The served schema expresses requiredness as a top-level ``required`` list.
        The SDK carries it per-property as ``nullable`` so each property is
        self-describing; everything else in the property is preserved verbatim.
        """
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

            # Assign, never setdefault: a served `nullable` (OpenAPI 3.0 style) means
            # "accepts null", not "optional". Letting it stand would drop the field
            # from `required` and the model would omit it.
            prop["nullable"] = name not in required_fields
            normalized[str(name)] = prop

        return normalized
