"""The StackOne toolset: fetches the served tool catalog and exposes it to agent frameworks."""

from __future__ import annotations

import concurrent.futures
import copy
import fnmatch
import logging
import os
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, TypeVar

import httpx

from stackone_ai.tools import (
    McpToolDefinition,
    RateLimitRetryingClient,
    RateLimitTimeout,
    StackOneMcpTool,
    StackOneTool,
    Tools,
    _js_number,
    _json_text,
    _json_type,
    build_request_headers,
    fetch_mcp_tools,
    is_rate_limited,
    is_sdk_owned_header,
)
from stackone_ai.types import (
    DEFAULT_BASE_URL,
    SUBMIT_FEEDBACK_TOOL_NAME,
    ExecuteToolsConfig,
    FeedbackCategory,
    FeedbackRating,
    FeedbackSource,
    Headers,
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

_UNSET = object()
"""Sentinel: `mode=None` is a real mode (individual), so it cannot mean "use the default"."""

_Listing = tuple[McpToolDefinition, str | None, str]
"""A served tool, the account that listed it, and the endpoint it was listed from."""

# The search_actions meta tool's served schema caps top_k at 50.
_MAX_TOP_K = 50

FAILED_ACCOUNT_RETRY_SECONDS = 30.0
"""How long an account that failed to list tools is left out of a cached catalog.

Its healthy siblings' listings are cached all the same. A later call within this window
serves them without the failed account, and without warning about it again; the first
call after it lists the failed account again.
"""

# Looked up at call time, so tests can move it on rather than wait.
_clock = time.monotonic


@dataclass(frozen=True)
class _Catalog:
    """A cached catalog: each healthy account's listing, and when each failed one failed."""

    listings: dict[str | None, list[_Listing]]
    failed_at: dict[str | None, float] = field(default_factory=dict)

    def in_order(self, account_scope: list[str | None]) -> list[_Listing]:
        return [listing for account in account_scope for listing in self.listings.get(account, [])]


_Item = TypeVar("_Item")
_Result = TypeVar("_Result")


def _fan_out(
    fn: Callable[[_Item], _Result], items: Sequence[_Item]
) -> list[tuple[_Item, _Result | Exception]]:
    """Run ``fn`` over ``items`` concurrently; each item's result or failure, in item order.

    A 429 that outlasted its retries is raised at once instead of being returned: the
    caller would skip it like any other failure and hand back a partial result. Work not
    yet started is cancelled; calls already running finish in the background, unread.
    """
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=min(len(items), 10))
    try:
        futures = {pool.submit(fn, item): index for index, item in enumerate(items)}
        outcomes: dict[int, _Result | Exception] = {}
        for future in concurrent.futures.as_completed(futures):
            try:
                outcomes[futures[future]] = future.result()
            except Exception as exc:
                if is_rate_limited(exc):
                    raise
                outcomes[futures[future]] = exc
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
    return [(item, outcomes[index]) for index, item in enumerate(items)]


def _all_accounts_failed(failures: list[tuple[str | None, Exception]]) -> Exception:
    """The error for a listing in which every account failed, in account order.

    When every failure is a StackOneAPIError with the same status, the first of them: two
    accounts answering 401 are a revoked key, and a caller must be able to tell that from a
    429 as it can with one account. Otherwise a ToolsetLoadError, caused by an
    ExceptionGroup of the failures and keeping them in ``failures`` too.
    """
    errors = [failure for _, failure in failures]
    first = errors[0]
    if isinstance(first, StackOneAPIError) and all(
        isinstance(error, StackOneAPIError) and error.status_code == first.status_code for error in errors
    ):
        return first
    error = ToolsetLoadError(
        "Every account failed to list tools: "
        + "; ".join(f"{account}: {failure}" for account, failure in failures),
        failures=errors,
    )
    error.__cause__ = ExceptionGroup("Every account failed to list tools", errors)
    return error


class StackOneToolSet:
    """Main class for accessing StackOne tools.

    The toolset is a thin client over the served catalog: it lists tools from the
    MCP endpoint and executes each one over MCP ``tools/call`` on the endpoint, and
    with the account, that listed it. It does not rewrite, filter or invent schemas.
    """

    def __init__(
        self,
        api_key: str | None = None,
        account_id: str | None = None,
        base_url: str | None = None,
        execute: ExecuteToolsConfig | None = None,
        timeout: float | None = None,
        tool_mode: ToolMode | None = None,
        headers: Headers | None = None,
    ) -> None:
        """Initialize StackOne tools with authentication

        Args:
            api_key: Optional API key. If not provided, will try to get from STACKONE_API_KEY env var
            account_id: Optional account ID
            base_url: Optional base URL override for API requests. Falls back to
                STACKONE_BASE_URL, then to the default, https://api.stackone.com. An
                empty string, whether passed here or read from the environment, is
                treated as not set rather than as a literal empty host.
            execute: Execution configuration. Controls default account scoping
                for tool execution. Pass ``{"account_ids": ["acc-1"]}`` to scope
                tools to specific accounts.
            timeout: Request timeout in seconds for tool listing and execution.
                Default: 60. Takes precedence over ``execute.timeout`` if set.
                Increase for slow providers (e.g. Workday).
            tool_mode: How the endpoint lists tools. ``"search_execute"`` returns
                two meta tools per connector instead of one tool per action,
                keeping the catalog small enough for a model's context.
            headers: Extra HTTP headers sent with every request this toolset makes
                (``GET /accounts`` and every MCP request). ``Authorization``,
                ``x-account-id`` and ``User-Agent`` are always the SDK's own and
                cannot be set here. ``x-end-user-id`` can: it is passed through as
                given, unless ``GET /accounts`` reported the account's end user,
                which then replaces it.

        Raises:
            ToolsetConfigError: If no API key is provided or found in environment, or
                ``account_id`` is an empty string
        """
        # An empty account_id is usually an unset variable, and treating it as unset would
        # silently widen every call to all active accounts.
        if account_id == "":
            raise ToolsetConfigError("account_id must not be an empty string")
        api_key_value = api_key or os.getenv("STACKONE_API_KEY")
        if not api_key_value:
            raise ToolsetConfigError(
                "An API key must be provided, either to the toolset or in the STACKONE_API_KEY "
                "environment variable"
            )
        ignored_headers = [name for name in (headers or {}) if is_sdk_owned_header(name)]
        if ignored_headers:
            logger.warning(
                "Ignoring headers %s: the SDK sets them itself. Pass the API key and account ids "
                "through their own options instead.",
                ", ".join(_json_text(name) for name in ignored_headers),
            )

        self.api_key: str = api_key_value
        self.account_id = account_id
        self.base_url = base_url or os.getenv("STACKONE_BASE_URL") or DEFAULT_BASE_URL
        self._headers: Headers = dict(headers or {})
        self._account_ids: list[str] = self._validate_account_ids(
            execute.get("account_ids") if execute else None
        )
        self._execute_config: ExecuteToolsConfig | None = execute
        execute_timeout = execute.get("timeout") if execute else None
        self._timeout: float = (
            timeout if timeout is not None else (execute_timeout if execute_timeout is not None else 60.0)
        )
        # Cache the listing, not the Tools wrapper. StackOneTool objects are mutable
        # (Tools.set_account_id rebinds them), so handing the same instances back on a
        # cache hit let one caller silently rescope every later caller's tools.
        self._catalog_cache: dict[tuple[Any, ...], _Catalog] = {}
        self._discovered_account_ids: list[str] | None = None
        # Shared while in flight, so concurrent fetch_tools() and search() calls on a
        # fresh toolset make one GET /accounts between them rather than one each.
        self._discovering: concurrent.futures.Future[list[str]] | None = None
        # Bumped by clear_catalog_cache(). A listing already in flight when the cache is
        # cleared captured the generation it started under, and refuses to write back if
        # it has moved — otherwise the stale catalog lands *after* the clear and is
        # served for the life of the process, which is the one thing the clear exists
        # to prevent.
        self._cache_generation = 0
        self._cache_lock = threading.Lock()
        # Each non-shared account's end-user id, from the last successful GET /accounts.
        # Replaced whole by the next one, and kept by clear_catalog_cache(): it describes
        # the accounts, not the catalog.
        self._end_user_ids: dict[str, str] = {}
        # Bumped at the start of every fetch_accounts() call, so two overlapping calls —
        # fetch_accounts() racing discovery, say — are ordered by when they started
        # rather than when they returned: a slower, older-started response must not
        # overwrite a newer one that already landed.
        self._accounts_sequence = 0
        self._end_user_ids_sequence = 0
        self._tool_mode: ToolMode | None = tool_mode

        # 2.x read STACKONE_ACCOUNT_ID; 3.x never does, and without an account it uses every
        # active account on the key. Anyone upgrading with only the variable set would
        # otherwise widen to other end users' accounts with no sign of it. An empty
        # account_ids list is unset too, the same as _resolve_account_ids() treats it.
        scoped = account_id is not None or bool(self._account_ids)
        if os.getenv("STACKONE_ACCOUNT_ID") and not scoped:
            logger.warning(
                "STACKONE_ACCOUNT_ID is set, but the SDK does not read it: with no account id passed, "
                "every active account on this API key is used. Pass an account id to scope the toolset."
            )

    def set_accounts(self, account_ids: list[str] | None) -> StackOneToolSet:
        """Set account IDs for filtering tools

        Returns:
            This toolset instance for chaining
        """
        self._account_ids = self._validate_account_ids(account_ids)
        self.clear_catalog_cache()
        return self

    def clear_catalog_cache(self) -> None:
        """Invalidate the cached tool catalog.

        Call when linked accounts change outside of ``set_accounts`` or when
        you need to force a fresh fetch from the StackOne MCP endpoint. The end-user ids
        recorded from GET /accounts are kept until the next one replaces them.
        """
        with self._cache_lock:
            self._cache_generation += 1
            self._catalog_cache.clear()
            self._discovered_account_ids = None
            self._discovering = None

    def fetch_tools(
        self,
        *,
        account_ids: list[str] | None = None,
        providers: list[str] | None = None,
        actions: list[str] | None = None,
        mode: Any = _UNSET,
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
            ToolsetLoadError: If there is an error loading the tools. When every account
                fails for differing reasons, its ``failures`` holds each account's error,
                and its ``__cause__`` is an ExceptionGroup of them.
            StackOneAPIError: With ``status_code`` 429 if the API is still rate limiting
                after retries, even when only one of several accounts is; or with the API's
                status when every account fails with that same status, so a caller can tell
                a revoked key's 401 from a 429. Other per-account failures skip that account
                with a warning, and leave it out for ``FAILED_ACCOUNT_RETRY_SECONDS``.

        A 429 is retried up to three times, after the server's ``Retry-After`` (capped at
        30 seconds) or else a 1s, 2s, 4s backoff with jitter. A wait that would not end
        within ``timeout`` is not started, and the 429 is raised instead.

        Examples:
            tools = toolset.fetch_tools(account_ids=['123', '456'])
            tools = toolset.fetch_tools(providers=['hibob', 'bamboohr'])
            tools = toolset.fetch_tools(actions=['*_list_employees'])

        Note:
            For organizations with multiple connected accounts, calling `fetch_tools()`
            without `account_ids` discovers and fetches the catalog for every active account.
            If your organization has many accounts, pass explicit `account_ids` to avoid
            excessive round trips and blowing model context limits.
        """
        try:
            mode = self._tool_mode if mode is _UNSET else mode
            # Captured before account resolution, which may discover accounts: a
            # clear_catalog_cache() landing mid-discovery must stop the catalog it
            # resolves to from being cached too, not just the discovered list itself.
            generation = self._cache_generation
            account_scope: list[str | None] = sorted(
                dict.fromkeys(self._resolve_account_ids(account_ids)), key=lambda a: (a is None, a)
            )

            # Keyed on what was fetched, not on how it is filtered: providers and
            # actions narrow the list in memory, so they must not force a refetch.
            # base_url and api_key belong here — leaving them out meant reassigning
            # either one kept serving the old catalog, still pointed at the old host.
            listings = self._catalog(account_scope, mode, generation)

            all_tools = [
                self._create_tool(tool_def, account, endpoint)
                for tool_def, account, endpoint in self._dedupe_global_tools(listings)
            ]

            if providers:
                all_tools = [tool for tool in all_tools if self._filter_by_provider(tool.name, providers)]

            if actions:
                all_tools = [tool for tool in all_tools if self._filter_by_action(tool.name, actions)]

            return Tools(all_tools)

        except (ToolsetError, StackOneError):
            # StackOneAPIError carries the HTTP status. Re-wrapping it below would throw
            # that away, so a caller could not tell a 401 from a 429.
            raise
        except Exception as exc:  # pragma: no cover - unexpected runtime errors
            raise ToolsetLoadError(f"Error fetching tools: {exc}") from exc

    @staticmethod
    def _validate_account_ids(account_ids: Any) -> list[str]:
        """A copy of the account ids, refusing a bare string, a non-list, a non-string id or
        an empty id. ``None`` means not given: no ids.

        An empty or ``None`` id would be sent with no ``x-account-id``, so it is rejected
        rather than letting the server answer for an account nobody chose, as in Node. A
        dict is refused rather than iterated, which read ``{"id": "acc-1"}`` as the account
        ``"id"``.
        """
        if account_ids is None:
            return []
        if isinstance(account_ids, str):
            raise ToolsetConfigError(
                "account_ids must be a list of account ids, not a string. "
                f"Did you mean [{_json_text(account_ids)}]?"
            )
        if not isinstance(account_ids, list) or not all(isinstance(i, str) for i in account_ids):
            raise ToolsetConfigError("account_ids must be a list of account id strings")
        ids = list(account_ids)
        if "" in ids:
            raise ToolsetConfigError("account_ids must not contain an empty account id")
        return ids

    def _resolve_account_ids(self, account_ids: list[str] | None) -> list[str]:
        """The accounts a call is scoped to, in the order they were given or discovered.

        The argument, then ``set_accounts()``, then the constructor's ``account_id``, then
        every active account ``GET /accounts`` lists.
        """
        if account_ids is not None:
            account_ids = self._validate_account_ids(account_ids)
        resolved = account_ids or self._account_ids
        if not resolved and self.account_id:
            resolved = [self.account_id]
        if not resolved:
            resolved = self._discover_account_ids()
        return list(resolved)

    def _catalog(
        self, account_scope: list[str | None], mode: ToolMode | None, generation: int
    ) -> list[_Listing]:
        """Every scoped account's catalog, from the cache where it can be.

        One unusable account must not cost the caller every other account's tools, so a
        failing account is skipped with a warning. The healthy accounts' listings are cached
        all the same, and the failed account is left out, silently, until
        ``FAILED_ACCOUNT_RETRY_SECONDS`` have passed, when the next call lists it again. Not
        caching at all made every call re-list every account, and wait out the full timeout
        on one that hangs.

        A 429 that outlasts its retries is raised, not skipped: it is the key's, not the
        account's, and skipping it would hand back a catalog missing whichever accounts
        happened to be throttled. When every account fails, see :func:`_all_accounts_failed`.

        ``generation`` is the cache generation as of the start of the call that resolved
        ``account_scope``: a clear_catalog_cache() during account discovery must stop the
        catalog this scope resolves to from being cached, same as the discovered list.
        """
        key = self._cache_key(account_scope, mode)
        with self._cache_lock:
            cached = self._catalog_cache.get(key)
        now = _clock()
        if cached is None:
            due = account_scope
        else:
            due = [
                account
                for account, failed_at in cached.failed_at.items()
                if now - failed_at >= FAILED_ACCOUNT_RETRY_SECONDS
            ]
            if not due:
                return cached.in_order(account_scope)

        # No param-style pin: arguments are sent verbatim and the server maps them with its
        # own reverse map, so the model sees whatever style the server serves.
        endpoint = f"{self.base_url.rstrip('/')}/mcp"
        if mode:
            endpoint = f"{endpoint}?tool-mode={mode}"

        def _fetch_for_account(
            account: str | None,
        ) -> list[_Listing]:
            headers = self._build_mcp_headers(account)
            listed = fetch_mcp_tools(endpoint, headers, timeout=self._timeout)
            return [(tool_def, account, endpoint) for tool_def in listed]

        def _store(catalog: _Catalog) -> None:
            # Merged with whatever is cached now, not overwritten: a concurrent call for the
            # same key may have stored a healthier catalog since this call read it, and a
            # wholesale overwrite would hide its successes behind this call's failures for
            # FAILED_ACCOUNT_RETRY_SECONDS. A listing wins over a failure for the same account.
            with self._cache_lock:
                if generation != self._cache_generation:
                    return
                existing = self._catalog_cache.get(key)
                if existing is not None:
                    merged_listings = {**existing.listings, **catalog.listings}
                    merged_failed_at = {**existing.failed_at, **catalog.failed_at}
                    for account in catalog.listings:
                        merged_failed_at.pop(account, None)
                    catalog = _Catalog(merged_listings, merged_failed_at)
                self._catalog_cache[key] = catalog

        if len(account_scope) == 1:
            catalog = _Catalog({account_scope[0]: _fetch_for_account(account_scope[0])})
            _store(catalog)
            return catalog.in_order(account_scope)

        listings = dict(cached.listings) if cached else {}
        failed_at = dict(cached.failed_at) if cached else {}
        failures: list[tuple[str | None, Exception]] = []
        for account, outcome in _fan_out(_fetch_for_account, due):
            if isinstance(outcome, Exception):
                failures.append((account, outcome))
            else:
                listings[account] = outcome
                failed_at.pop(account, None)
        if not listings:
            raise _all_accounts_failed(failures)
        failed_now = _clock()
        for account, failure in failures:
            logger.warning("Skipping account that failed to list tools — %s: %s", account, failure)
            failed_at[account] = failed_now

        catalog = _Catalog(listings, failed_at)
        _store(catalog)
        return catalog.in_order(account_scope)

    @staticmethod
    def _dedupe_global_tools(
        listings: list[_Listing],
    ) -> list[_Listing]:
        """Keep only the first listing of the feedback tool.

        It is global rather than account-scoped, so every account's listing carries an
        identical copy. Left in, N accounts meant N same-named tools and a duplicate-name
        warning about a clash that cannot misroute anything — noise that teaches callers
        to ignore the warning that does matter.
        """
        seen_feedback = False
        kept: list[_Listing] = []
        for listing in listings:
            if listing[0].name == SUBMIT_FEEDBACK_TOOL_NAME:
                if seen_feedback:
                    continue
                seen_feedback = True
            kept.append(listing)
        return kept

    def _cache_key(self, account_scope: list[str | None], mode: ToolMode | None) -> tuple[Any, ...]:
        return (
            tuple(sorted(account_scope, key=lambda a: (a is None, a))),
            mode,
            self.base_url,
            self.api_key,
        )

    def _meta_tools(self, suffix: str, account_ids: list[str] | None = None) -> list[StackOneTool]:
        """The server's per-connector meta tools, regardless of this toolset's mode.

        The mode is passed down rather than assigned to ``self``. Flipping instance
        state here raced with any concurrent ``fetch_tools()``: that call could read
        the flipped mode partway through and cache search_execute meta tools under the
        individual-mode key, permanently, for every later caller.
        """
        tools = self.fetch_tools(account_ids=account_ids, mode="search_execute")
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
            top_k: Maximum results, across every connector searched.
            account_ids: Restrict to these accounts. Defaults to all active ones.

        Returns:
            At most ``top_k`` action dicts, best first across every connector, each carrying
            at least ``action_id`` and ``description``, the ``account_id`` of the account
            whose connector found it, and the ``session_id`` of the search when the server
            issued one. The same action linked on two accounts is two hits. Pass
            ``session_id`` to :meth:`execute` and :meth:`submit_feedback` to link the calls,
            and ``account_id`` in ``account_ids`` to run the action on that account.

        Raises:
            StackOneAPIError: With ``status_code`` 429 if the API is still rate limiting
                after retries, for any account or connector. A 429 is retried as in
                :meth:`fetch_tools`; other per-connector failures are skipped with a
                warning.
        """
        # The server rejects anything outside 1..50, but only after a round trip per
        # connector — and reports it as a load failure, which reads like an outage
        # rather than a typo. Fail here instead, where the caller can see why.
        if not isinstance(top_k, int) or isinstance(top_k, bool) or not 1 <= top_k <= _MAX_TOP_K:
            raise ToolsetConfigError(
                f"top_k must be an integer between 1 and {_MAX_TOP_K}, got {_json_text(top_k)}"
            )

        tools = self._meta_tools("_search_actions", account_ids)
        if not tools:
            return []

        def _search_one(tool: StackOneTool) -> list[JsonDict]:
            found = tool.execute({"query": query, "top_k": top_k})
            actions = list(found.get("actions", []))
            # The server returns session_id once per search, beside the actions. Results
            # from every connector are merged and re-ranked below, so this is the last
            # point at which a hit can still be traced to the search, and the account,
            # that produced it.
            session_id = found.get("session_id")
            traced: JsonDict = {}
            if isinstance(session_id, str) and session_id:
                traced["session_id"] = session_id
            account_id = tool.get_account_id()
            if account_id:
                traced["account_id"] = account_id
            return [{**action, **traced} for action in actions]

        results: list[JsonDict] = []
        failures: list[str] = []
        # Fan out the way fetch_tools() does. Serially, a customer with a dozen
        # connectors pays the sum of every connector's latency on the headline call.
        for tool, outcome in _fan_out(_search_one, tools):
            if isinstance(outcome, Exception):
                # Skip everything but a rate limit, as fetch_tools() does. Catching only
                # StackOneError let a transport failure — which _describe_mcp_failure
                # reports as a ToolsetLoadError — abort the whole search, which is
                # precisely the flakiness this guard exists to absorb.
                failures.append(f"{tool.name}: {outcome}")
            else:
                results.extend(outcome)

        if failures and not results:
            raise ToolsetLoadError("No connector returned results. " + " | ".join(failures))
        for failure in failures:
            logger.warning("Skipping connector that failed to search — %s", failure)

        # Concatenating per-connector results leaves the list grouped by connector, so
        # results[0] would be the best hit of whichever connector answered first rather
        # than the best hit overall. Rank globally; the server scores every action on the
        # same scale. Actions without a score sort last rather than raising.
        def _score(action: JsonDict) -> float:
            raw = action.get("similarity_score")
            if isinstance(raw, bool) or not isinstance(raw, int | float):
                return 0.0
            return float(raw)

        # Then cut to top_k: each connector was asked for top_k, so the merged list can
        # hold many more.
        results.sort(key=_score, reverse=True)
        return results[:top_k]

    def execute(
        self,
        action_id: str,
        arguments: JsonDict | None = None,
        *,
        account_ids: list[str] | None = None,
        session_id: str | None = None,
    ) -> JsonDict:
        """Execute an action by id, as returned by :meth:`search`.

        Always runs through the connector's ``_execute_action`` meta tool, so
        ``arguments`` is the nested form every action's ``example_request``
        shows — ``{"query": {...}, "path": {...}, "body": {...}}``. A
        ``fetch_tools()`` tool takes the keys its own served schema names instead;
        routing by whether an id happened to be in the catalog would make the
        argument shape depend on something the caller cannot see.

        ``arguments["headers"]`` is forwarded to the action: ``*_execute_action`` serves an
        open ``headers`` object, so any header name is accepted except Authorization,
        x-account-id, User-Agent and x-end-user-id, which the SDK sets itself and which are
        dropped.

        ``session_id`` is the value a :meth:`search` hit carries. Passing it links
        this call to that search server-side.

        ``account_ids`` restricts routing to these accounts. Pass a search hit's
        ``account_id`` to run the action on the account that found it.

        Raises:
            ToolsetConfigError: If ``action_id``, ``arguments`` or ``session_id`` is malformed,
                or the action's connector is linked on more than one account in scope.
            ToolArgumentsError: If the arguments cannot be encoded as JSON.
            ToolsetLoadError: If no connector matches.
            StackOneAPIError: If the action fails, including when the server rejects the
                arguments.
        """
        if not isinstance(action_id, str) or not action_id:
            raise ToolsetConfigError(f"action_id must be a non-empty string, got {_json_text(action_id)}")
        if arguments is not None and not isinstance(arguments, dict):
            raise ToolsetConfigError(f"arguments must be a JSON object, got {_json_type(arguments)}")
        if session_id is not None and (not isinstance(session_id, str) or not session_id):
            raise ToolsetConfigError(f"session_id must be a non-empty string, got {_json_text(session_id)}")

        meta_tools = self._meta_tools("_execute_action", account_ids)
        matches = [
            tool
            for tool in meta_tools
            if action_id.lower().startswith(self._connector_of(tool, "_execute_action") + "_")
        ]
        if matches:
            # Longest connector wins: with both `browser` and `browser_linkedin` linked,
            # the first token alone would route every browser_linkedin action to browser.
            best = max(len(self._connector_of(t, "_execute_action")) for t in matches)
            finalists = [t for t in matches if len(self._connector_of(t, "_execute_action")) == best]
            if len(finalists) > 1:
                # The same provider linked twice, which discovery makes common. Picking one
                # would run the action against an account the caller never chose — another
                # end user's, possibly.
                ordered = sorted(finalists, key=lambda t: t.get_account_id() or "")
                raise ToolsetConfigError(
                    f"{_json_text(action_id)} matches {len(finalists)} connectors on different accounts "
                    f"({', '.join(f'{t.name} on {t.get_account_id()}' for t in ordered)}). "
                    "Pass the account id to use, such as a search hit's account_id."
                )
            tool = finalists[0]
            # action_id LAST, removed first so it is last in key order too, as in Node.
            # Spreading arguments over it let a model-supplied "action_id" silently replace
            # the action the caller pinned — the exact thing a host app pins it for.
            # session_id only when given: the served schema makes it an optional string, so
            # an absent key is valid and a null is not.
            call_arguments: JsonDict = dict(arguments or {})
            call_arguments.pop("action_id", None)
            if session_id is not None:
                call_arguments.pop("session_id", None)
                call_arguments["session_id"] = session_id
            call_arguments["action_id"] = action_id
            # Returned as the server wrote it, the same shape tool.execute() returns.
            return tool.execute(call_arguments)

        raise ToolsetLoadError(
            f"No connector found for {_json_text(action_id)}. Use search() to discover valid action ids."
        )

    def submit_feedback(
        self,
        rating: FeedbackRating,
        tool_names: Sequence[str],
        *,
        feedback: str | None = None,
        category: FeedbackCategory | None = None,
        session_id: str | None = None,
        source: FeedbackSource = "model",
        account_ids: list[str] | None = None,
        action_run_id: str | None = None,
    ) -> JsonDict:
        """Record a verdict on how well the tools served this session.

        Calls the server's ``stackone_submit_feedback`` tool once, through the account with
        the lowest id among those the call is scoped to: ``account_ids``, or else the
        toolset's, or else every active one. Pass the ``session_id`` from a :meth:`search`
        hit to attach the feedback to that session.

        Args:
            rating: ``"positive"``, ``"negative"`` or ``"neutral"``.
            tool_names: The tools or action ids the feedback is about.
            feedback: An optional one-line reason.
            category: What the feedback is about, e.g. ``"search"`` or ``"execute"``.
            session_id: The session to link this feedback to.
            source: Who produced the feedback.
            account_ids: Accounts to choose from; only the lowest id is used. Defaults to
                the toolset's own.
            action_run_id: The action run the feedback is about. Sent only when given.

        Raises:
            ToolsetConfigError: If ``tool_names`` is a string or ``session_id`` is empty or
                not a string.
            ToolsetLoadError: If feedback is not enabled for this project.
        """
        if isinstance(tool_names, str):
            raise ToolsetConfigError(
                "tool_names must be a list of tool names, not a string. "
                f"Did you mean [{_json_text(tool_names)}]?"
            )
        if session_id is not None and (not isinstance(session_id, str) or not session_id):
            raise ToolsetConfigError(f"session_id must be a non-empty string, got {_json_text(session_id)}")

        # Found in the served catalog, never built here: the server only serves the tool
        # when the org flag and the project setting are both on, and a client-side stand-in
        # would report success for feedback that went nowhere. It is global and served in
        # every mode, so one account's search_execute listing — two tools per connector,
        # not one per action — is enough to find it.
        # The lowest id is the one a caller can predict, whatever order GET /accounts lists
        # them in.
        first_account = min(self._resolve_account_ids(account_ids))
        tool = self.fetch_tools(account_ids=[first_account], mode="search_execute").get_tool(
            SUBMIT_FEEDBACK_TOOL_NAME
        )
        if tool is None:
            raise ToolsetLoadError(
                f"The server did not serve {SUBMIT_FEEDBACK_TOOL_NAME}: feedback is not enabled "
                "for this project."
            )

        # Unset optionals are omitted, never sent as null: the served schema declares them
        # as optional strings, so a null is a present key of the wrong type and fails
        # validation where an absent one would not.
        arguments: JsonDict = {"rating": rating, "tool_names": list(tool_names)}
        optional = {
            "feedback": feedback,
            "category": category,
            "session_id": session_id,
            "action_run_id": action_run_id,
            "source": source,
        }
        arguments.update({key: value for key, value in optional.items() if value is not None})
        return tool.execute(arguments)

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

    @staticmethod
    def _connector_of(tool: StackOneTool, suffix: str) -> str:
        """The connector a meta tool belongs to: its name minus the account id and suffix.

        The account id is stripped by identity, not by splitting on the last underscore.
        Account ids are nanoid-shaped and nanoid's default alphabet includes ``_``, so
        splitting turned ``mock_acc_1_execute_action`` into connector ``mock_acc`` and
        made every action on that account unroutable — with an error blaming the
        caller's action id.
        """
        stem = tool.name[: -len(suffix)] if tool.name.endswith(suffix) else tool.name
        account = tool.get_account_id()
        if account and stem.endswith(f"_{account}"):
            stem = stem[: -(len(account) + 1)]
        return stem.lower()

    def _filter_by_provider(self, tool_name: str, providers: list[str]) -> bool:
        """Whether a tool belongs to one of the given providers (case-insensitive).

        Matched as a full prefix rather than on the first underscore-separated token:
        splitting on "_" reads `browser_linkedin_search_people` as provider `browser`,
        so asking for `browser_linkedin` returned nothing at all — silently, since an
        empty result is indistinguishable from a provider with no tools.
        """
        lowered = tool_name.lower()
        return any(lowered.startswith(provider.lower() + "_") for provider in providers)

    def _filter_by_action(self, tool_name: str, actions: list[str]) -> bool:
        """Whether a tool name matches any of the given glob patterns."""
        return any(fnmatch.fnmatch(tool_name, pattern) for pattern in actions)

    def fetch_accounts(self) -> list[JsonDict]:
        """List the accounts linked to this API key.

        Each entry carries at least ``id``, ``provider`` and ``status``. Only
        accounts with ``status == "active"`` can serve tools.

        For each account with ``shared`` false and an ``origin_username``, that username is
        recorded as the account's end-user id, replacing the record of any call that
        started before this one; a call that started earlier but finishes later records nothing. Every
        MCP request this toolset's tools make for that account then carries it in
        ``x-end-user-id``, which the API requires for a non-shared account.

        Raises:
            StackOneAPIError: If the API answers with an error, including a 429 that
                outlasted its retries.
        """
        with self._cache_lock:
            self._accounts_sequence += 1
            sequence = self._accounts_sequence
        url = f"{self.base_url.rstrip('/')}/accounts"
        try:
            with RateLimitRetryingClient(timeout=self._timeout, retry_within=self._timeout) as client:
                response = client.get(
                    url,
                    headers=build_request_headers(api_key=self.api_key, extra_headers=self._headers),
                )
        except RateLimitTimeout as exc:
            # Timing out while a 429 is retried is still the rate limit, as it is over MCP.
            raise StackOneAPIError(
                f"Listing accounts at {url} was rate limited (429) and timed out after "
                f"{_js_number(self._timeout)}s while retrying",
                429,
                None,
            ) from exc
        except httpx.HTTPError as exc:
            # The only public method with no error handling at all: a dead host, a bad
            # scheme or a timeout leaked httpx's own exception type straight out of the
            # SDK, outside the documented hierarchy.
            raise ToolsetLoadError(f"Could not reach {url}: {exc}") from exc

        if response.is_error:
            # Without this the catch-all in fetch_tools flattens it to a message and
            # the status is lost, so a caller cannot tell 401 from 429.
            raise StackOneAPIError(
                f"Listing accounts at {url} failed with "
                f"{response.status_code} {response.reason_phrase}: {response.text.strip()}",
                response.status_code,
                response.text,
            )
        try:
            body = response.json()
        except (ValueError, UnicodeDecodeError) as exc:
            raise ToolsetLoadError(f"Invalid JSON returned by {url}: {exc}") from exc
        accounts = body.get("data", body) if isinstance(body, dict) else body
        if not isinstance(accounts, list):
            # list(dict) yields the KEYS, so coercing here turned an unexpected wrapper
            # into a list of strings that blew up much later as an AttributeError.
            raise ToolsetLoadError(
                f"Unexpected /accounts response shape: expected a list, got {_json_type(accounts)}"
            )
        end_user_ids = {
            account["id"]: account["origin_username"]
            for account in accounts
            if isinstance(account, dict)
            and isinstance(account.get("id"), str)
            and account["id"]
            and account.get("shared") is False
            and isinstance(account.get("origin_username"), str)
            and account["origin_username"]
        }
        with self._cache_lock:
            # Only if no later-started call's result is already in: a call that started
            # first but finished last is answering a question a newer call has since
            # answered better, and must not clobber it.
            if sequence >= self._end_user_ids_sequence:
                self._end_user_ids = end_user_ids
                self._end_user_ids_sequence = sequence
        return accounts

    def _end_user_id(self, account_id: str) -> str | None:
        """The end-user id GET /accounts recorded for an account, if it is not shared."""
        with self._cache_lock:
            return self._end_user_ids.get(account_id)

    def _discover_account_ids(self) -> list[str]:
        """List the linked accounts this API key can use.

        The MCP endpoint requires an ``x-account-id`` on every request, so an API
        key on its own is not enough to list tools. Rather than make every caller
        supply one, ask the API which accounts the key has.

        Warning: For organizations with many connected accounts, relying on discovery
        fetches the tool catalog for every account. If you have a large number of
        accounts, it is highly recommended to supply specific ``account_id`` or
        ``account_ids`` to avoid excessive API round trips and huge model contexts.

        Raises:
            ToolsetConfigError: If the key has no accounts, or none are usable.
        """
        if self._discovered_account_ids is not None:
            return self._discovered_account_ids

        with self._cache_lock:
            if self._discovered_account_ids is not None:
                return self._discovered_account_ids
            future = self._discovering
            owner = future is None
            if owner:
                future = self._discovering = concurrent.futures.Future()

        if future is None:
            raise RuntimeError("account discovery future was not initialized")
        if not owner:
            # A discovery already in flight: wait for it rather than issue a second
            # GET /accounts. Its result, or its exception, is shared with every waiter.
            return future.result()

        try:
            active = self._fetch_active_account_ids()
        except BaseException as exc:
            with self._cache_lock:
                if self._discovering is future:
                    self._discovering = None
            future.set_exception(exc)
            raise
        with self._cache_lock:
            if self._discovering is future:
                self._discovering = None
        future.set_result(active)
        return active

    def _fetch_active_account_ids(self) -> list[str]:
        generation = self._cache_generation
        accounts = self.fetch_accounts()

        active = [a["id"] for a in accounts if a.get("status") == "active" and a.get("id")]
        if not active:
            if not accounts:
                raise ToolsetConfigError(
                    "This API key has no linked accounts. Link one in the StackOne "
                    "dashboard, or pass an account id explicitly."
                )
            listed = ", ".join(f"{a.get('provider')} ({a.get('status')})" for a in accounts)
            raise ToolsetConfigError(
                f"None of this API key's {len(accounts)} linked accounts are active: {listed}. "
                "Re-link them in the StackOne dashboard, or pass an account id explicitly."
            )

        # Not kept if clear_catalog_cache() ran while the list was in flight: it may name
        # accounts the clear was meant to forget.
        with self._cache_lock:
            if generation == self._cache_generation:
                self._discovered_account_ids = active
        return active

    def _build_mcp_headers(self, account_id: str | None) -> dict[str, str]:
        end_user_id = self._end_user_id(account_id) if account_id else None
        return build_request_headers(
            api_key=self.api_key,
            account_id=account_id,
            end_user_id=end_user_id,
            extra_headers=self._headers,
        )

    def _create_tool(
        self,
        tool_def: McpToolDefinition,
        account_id: str | None,
        endpoint: str,
    ) -> StackOneTool:
        """Build an executable tool from a served catalog entry.

        Every tool, in every mode, executes over MCP ``tools/call`` on the endpoint that
        listed it, with the ``x-account-id`` of the account that listed it, and the
        ``x-end-user-id`` recorded for that account when the request is made.
        """
        schema = tool_def.input_schema or {}
        # Pop keys we explicitly override to avoid "multiple values for keyword argument"
        rest = dict(schema)
        # A string type is passed through; anything else, such as ["object", "null"], is
        # "object", as in Node. str() handed the model "['object', 'null']".
        served_type = rest.pop("type", "object")
        schema_type = served_type if isinstance(served_type, str) else "object"
        rest.pop("properties", None)
        schema_properties = self._normalize_schema_properties(schema)

        parameters = ToolParameters(
            **rest,
            type=schema_type,
            properties=schema_properties,
        )
        tool = StackOneMcpTool(
            name=tool_def.name,
            description=tool_def.description or "",
            parameters=parameters,
            api_key=self.api_key,
            endpoint=endpoint,
            account_id=account_id,
            headers=self._headers,
            timeout=self._timeout,
        )
        tool._end_user_id_of = self._end_user_id
        return tool

    def _normalize_schema_properties(self, schema: dict[str, Any]) -> dict[str, Any]:
        """Mirror the served schema's properties, recording requiredness per property.

        The served schema expresses requiredness as a top-level ``required`` list.
        The SDK carries it per-property as ``nullable`` so each property is
        self-describing; everything else in the property is preserved verbatim.
        """
        properties = schema.get("properties")
        if not isinstance(properties, dict):
            return {}

        raw_required = schema.get("required")
        # A string `required` would iterate as characters and mark every real property
        # optional; a null would raise and fail the whole catalog over one bad tool. A
        # non-string entry is dropped, as to_openai_function() drops it, so the marker and
        # the `required` a model is shown cannot disagree.
        served_required = raw_required if isinstance(raw_required, list) else []
        required_fields = {name for name in served_required if isinstance(name, str)}
        normalized: dict[str, Any] = {}

        for name, details in properties.items():
            if isinstance(details, dict):
                prop = copy.deepcopy(details)
            else:
                prop = {"description": str(details)}

            # Assign, never setdefault: a served `nullable` (OpenAPI 3.0 style) means
            # "accepts null", not "optional". Letting it stand would drop the field
            # from `required` and the model would omit it.
            prop["nullable"] = name not in required_fields
            normalized[str(name)] = prop

        return normalized
