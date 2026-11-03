"""More than one account: the catalog cache, the search_execute trio, search hits and adapters.

With discovery as the default, the same provider linked on two accounts is common, so
every surface has to say which account it means.
"""

from __future__ import annotations

import logging
import threading
from typing import Any

import httpx
import pytest

from stackone_ai import tools as tools_module
from stackone_ai import toolset as toolset_module
from stackone_ai.tools import McpToolDefinition, StackOneMcpTool, Tools
from stackone_ai.toolset import FAILED_ACCOUNT_RETRY_SECONDS, StackOneToolSet
from stackone_ai.types import (
    StackOneAPIError,
    ToolArgumentsError,
    ToolParameters,
    ToolsetConfigError,
    ToolsetLoadError,
)


def _tool(name: str) -> McpToolDefinition:
    return McpToolDefinition(name=name, description="", input_schema={})


class _Accounts:
    """A fake MCP listing per account: a list of tool names, or an exception to raise."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch, **accounts: list[str] | Exception) -> None:
        self.accounts = accounts
        self.listed: list[str] = []
        monkeypatch.setattr("stackone_ai.toolset.fetch_mcp_tools", self._fetch)

    def _fetch(self, _endpoint: str, headers: dict[str, str], **_kwargs: object) -> list[McpToolDefinition]:
        account = headers["x-account-id"]
        self.listed.append(account)
        served = self.accounts[account]
        if isinstance(served, Exception):
            raise served
        return [_tool(name) for name in served]


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """The failed-account clock, as a one-element list a test moves on by hand."""
    now = [1000.0]
    monkeypatch.setattr(toolset_module, "_clock", lambda: now[0])
    return now


class TestPartialFailureCaching:
    """The healthy accounts are cached; a failed one is left out until it is due a retry.

    Not caching at all made every call re-list every account, and wait out the whole
    timeout on one that hangs.
    """

    def test_the_healthy_accounts_are_served_from_the_cache(self, monkeypatch, clock, caplog):
        accounts = _Accounts(monkeypatch, a=["tool_a"], b=RuntimeError("boom"), c=["tool_c"])
        toolset = StackOneToolSet(api_key="k")
        with caplog.at_level(logging.WARNING, logger="stackone.tools"):
            first = toolset.fetch_tools(account_ids=["a", "b", "c"])
            clock[0] += FAILED_ACCOUNT_RETRY_SECONDS - 0.001
            second = toolset.fetch_tools(account_ids=["a", "b", "c"])

        assert [t.name for t in first] == [t.name for t in second] == ["tool_a", "tool_c"]
        assert sorted(accounts.listed) == ["a", "b", "c"]
        # Warned once, when it failed; not again while it is left out.
        assert [r.getMessage() for r in caplog.records] == [
            "Skipping account that failed to list tools — b: boom"
        ]

    def test_only_the_failed_account_is_retried_once_it_is_due(self, monkeypatch, clock, caplog):
        accounts = _Accounts(monkeypatch, a=["tool_a"], b=RuntimeError("boom"))
        toolset = StackOneToolSet(api_key="k")
        toolset.fetch_tools(account_ids=["a", "b"])
        accounts.listed.clear()

        clock[0] += FAILED_ACCOUNT_RETRY_SECONDS
        with caplog.at_level(logging.WARNING, logger="stackone.tools"):
            toolset.fetch_tools(account_ids=["a", "b"])
        assert accounts.listed == ["b"]
        assert "b: boom" in caplog.text

        # Failed again: left out for another full window from now.
        accounts.listed.clear()
        clock[0] += FAILED_ACCOUNT_RETRY_SECONDS - 0.001
        toolset.fetch_tools(account_ids=["a", "b"])
        assert accounts.listed == []

    def test_a_recovered_account_joins_the_cached_catalog_in_account_order(self, monkeypatch, clock):
        accounts = _Accounts(monkeypatch, a=RuntimeError("boom"), b=["tool_b"])
        toolset = StackOneToolSet(api_key="k")
        assert [t.name for t in toolset.fetch_tools(account_ids=["a", "b"])] == ["tool_b"]

        accounts.accounts["a"] = ["tool_a"]
        clock[0] += FAILED_ACCOUNT_RETRY_SECONDS
        assert [t.name for t in toolset.fetch_tools(account_ids=["a", "b"])] == ["tool_a", "tool_b"]

        accounts.listed.clear()
        clock[0] += 3600
        assert [t.name for t in toolset.fetch_tools(account_ids=["a", "b"])] == ["tool_a", "tool_b"]
        assert accounts.listed == []

    def test_a_clear_forgets_the_failure_too(self, monkeypatch, clock):
        accounts = _Accounts(monkeypatch, a=["tool_a"], b=RuntimeError("boom"))
        toolset = StackOneToolSet(api_key="k")
        toolset.fetch_tools(account_ids=["a", "b"])
        accounts.listed.clear()
        toolset.clear_catalog_cache()
        toolset.fetch_tools(account_ids=["a", "b"])
        assert sorted(accounts.listed) == ["a", "b"]


class TestConcurrentCallsDoNotClobberEachOthersPartialCatalog:
    """Two concurrent fetch_tools() calls for the same scope must not overwrite each
    other's cache entry: a transient failure in one must not hide a success the other
    just cached for FAILED_ACCOUNT_RETRY_SECONDS.
    """

    def test_a_slower_failure_does_not_clobber_a_faster_success(self, monkeypatch, clock):
        b_started = threading.Event()
        second_call_done = threading.Event()
        b_calls: list[int] = []

        def fetch(_endpoint: str, headers: dict[str, str], **_kwargs: object) -> list[McpToolDefinition]:
            account = headers["x-account-id"]
            if account == "a":
                return [_tool("tool_a")]
            b_calls.append(1)
            if len(b_calls) == 1:
                # The first caller to reach "b" blocks until the second caller's whole
                # fetch_tools() call has finished and cached both accounts, then fails —
                # so its own store is the last one to run.
                b_started.set()
                assert second_call_done.wait(timeout=5)
                raise RuntimeError("boom")
            return [_tool("tool_b")]

        monkeypatch.setattr(toolset_module, "fetch_mcp_tools", fetch)
        toolset = StackOneToolSet(api_key="k")
        results: dict[str, list[str]] = {}

        def call_first() -> None:
            results["first"] = [t.name for t in toolset.fetch_tools(account_ids=["a", "b"])]

        def call_second() -> None:
            assert b_started.wait(timeout=5)
            results["second"] = [t.name for t in toolset.fetch_tools(account_ids=["a", "b"])]
            second_call_done.set()

        first = threading.Thread(target=call_first)
        second = threading.Thread(target=call_second)
        first.start()
        second.start()
        first.join(timeout=5)
        second.join(timeout=5)

        assert results["second"] == ["tool_a", "tool_b"]
        # The slower, failing call must not have evicted "b" from the cache behind it.
        assert [t.name for t in toolset.fetch_tools(account_ids=["a", "b"])] == ["tool_a", "tool_b"]


class TestEveryAccountFails:
    def test_the_same_status_everywhere_is_raised_as_itself(self, monkeypatch):
        """A revoked key is a 401 with two accounts, as it is with one."""
        first, second = StackOneAPIError("a: 401", 401, None), StackOneAPIError("b: 401", 401, None)
        _Accounts(monkeypatch, b=second, a=first)
        with pytest.raises(StackOneAPIError) as excinfo:
            StackOneToolSet(api_key="k").fetch_tools(account_ids=["b", "a"])
        assert excinfo.value is first

    def test_differing_failures_are_kept(self, monkeypatch):
        unauthorised, broken = StackOneAPIError("401", 401, None), StackOneAPIError("412", 412, None)
        _Accounts(monkeypatch, a=unauthorised, b=broken)
        with pytest.raises(ToolsetLoadError) as excinfo:
            StackOneToolSet(api_key="k").fetch_tools(account_ids=["a", "b"])
        assert str(excinfo.value) == "Every account failed to list tools: a: 401; b: 412"
        assert excinfo.value.failures == [unauthorised, broken]
        cause = excinfo.value.__cause__
        assert isinstance(cause, ExceptionGroup)
        assert list(cause.exceptions) == [unauthorised, broken]

    def test_a_401_and_a_403_are_kept_rather_than_either_raised(self, monkeypatch):
        """Differing statuses are not one cause: raising the first would hide the other."""
        unauthorised, forbidden = StackOneAPIError("401", 401, None), StackOneAPIError("403", 403, None)
        _Accounts(monkeypatch, a=unauthorised, b=forbidden)
        with pytest.raises(ToolsetLoadError) as excinfo:
            StackOneToolSet(api_key="k").fetch_tools(account_ids=["a", "b"])
        assert str(excinfo.value) == "Every account failed to list tools: a: 401; b: 403"
        assert excinfo.value.failures == [unauthorised, forbidden]

    def test_an_account_that_lists_nothing_has_not_failed(self, monkeypatch):
        _Accounts(monkeypatch, a=[], b=RuntimeError("boom"))
        assert len(StackOneToolSet(api_key="k").fetch_tools(account_ids=["a", "b"])) == 0

    def test_nothing_is_cached(self, monkeypatch, clock):
        accounts = _Accounts(monkeypatch, a=RuntimeError("x"), b=RuntimeError("y"))
        toolset = StackOneToolSet(api_key="k")
        for _ in range(2):
            with pytest.raises(ToolsetLoadError):
                toolset.fetch_tools(account_ids=["a", "b"])
        assert len(accounts.listed) == 4


_TRIO = ["stackone_list_accounts", "stackone_search_actions", "stackone_execute_action"]
_TRIO_ENDPOINT = "https://api.stackone.com/mcp?tool-mode=search_execute"


class _Trio:
    """A fake search_execute listing of the trio, and a record of every listing and call.

    ``served`` is the tool names listed, or an exception to raise; ``hits`` what search returns.
    """

    def __init__(self, monkeypatch: pytest.MonkeyPatch, served: list[str] | Exception | None = None) -> None:
        self.served: list[str] | Exception = _TRIO if served is None else served
        self.hits: dict[str, Any] = {"session_id": "s-1", "actions": []}
        self.listings: list[tuple[str, dict[str, str]]] = []
        self.calls: list[tuple[str, str, str | None, Any]] = []
        monkeypatch.setattr("stackone_ai.toolset.fetch_mcp_tools", self._fetch)
        monkeypatch.setattr(StackOneMcpTool, "execute", lambda tool, args: self._execute(tool, args))

    def _fetch(self, endpoint: str, headers: dict[str, str], **_kwargs: object) -> list[McpToolDefinition]:
        self.listings.append((endpoint, headers))
        if isinstance(self.served, Exception):
            raise self.served
        return [_tool(name) for name in self.served]

    def _execute(self, tool: StackOneMcpTool, arguments: Any) -> dict[str, Any]:
        self.calls.append((tool.name, tool._endpoint, tool.get_account_id(), arguments))
        if tool.name == "stackone_search_actions":
            return self.hits
        return {"isError": False, "result": {}}


class TestSearchExecuteListing:
    """One listing spans the whole scope: the trio, bound to no account."""

    def test_every_account_is_in_one_sorted_deduped_comma_list_not_a_header(self, monkeypatch):
        trio = _Trio(monkeypatch)
        toolset = StackOneToolSet(api_key="k", execute={"account_ids": ["acc2", "acc1", "acc2"]})

        tools = toolset.fetch_tools(mode="search_execute")

        assert [t.name for t in tools] == _TRIO
        assert all(t.get_account_id() is None for t in tools)
        [(endpoint, headers)] = trio.listings
        assert endpoint == f"{_TRIO_ENDPOINT}&x-account-id=acc1%2Cacc2"
        assert "x-account-id" not in {name.lower() for name in headers}

    def test_the_trio_executes_on_the_endpoint_that_listed_it(self, monkeypatch):
        trio = _Trio(monkeypatch)
        toolset = StackOneToolSet(api_key="k", execute={"account_ids": ["acc1", "acc2"]})
        toolset.fetch_tools(mode="search_execute").get_tool("stackone_list_accounts").execute({})
        [(_name, endpoint, account, _args)] = trio.calls
        assert (endpoint, account) == (f"{_TRIO_ENDPOINT}&x-account-id=acc1%2Cacc2", None)

    def test_it_is_cached_per_scope(self, monkeypatch):
        trio = _Trio(monkeypatch)
        toolset = StackOneToolSet(api_key="k")
        for _ in range(2):
            toolset.fetch_tools(account_ids=["acc1", "acc2"], mode="search_execute")
        toolset.fetch_tools(account_ids=["acc1"], mode="search_execute")
        assert [endpoint for endpoint, _ in trio.listings] == [
            f"{_TRIO_ENDPOINT}&x-account-id=acc1%2Cacc2",
            f"{_TRIO_ENDPOINT}&x-account-id=acc1",
        ]

    def test_a_failure_is_raised_and_not_cached(self, monkeypatch):
        """There is one request, so no account to skip."""
        trio = _Trio(monkeypatch, served=StackOneAPIError("down", 503, None))
        toolset = StackOneToolSet(api_key="k", execute={"account_ids": ["acc1", "acc2"]})
        for _ in range(2):
            with pytest.raises(StackOneAPIError) as excinfo:
                toolset.fetch_tools(mode="search_execute")
            assert excinfo.value.status_code == 503
        assert len(trio.listings) == 2

    def test_providers_narrow_the_accounts_with_one_accounts_listing(self, monkeypatch):
        trio = _Trio(monkeypatch)
        calls = _providers(monkeypatch, acc1="HRIS", acc2="crm", acc3="hris")
        toolset = StackOneToolSet(api_key="k", execute={"account_ids": ["acc1", "acc2", "acc3"]})

        tools = toolset.fetch_tools(providers=["hris"], mode="search_execute")
        toolset.fetch_tools(providers=["crm"], mode="search_execute")

        # The trio is named after no provider, so it is kept whole.
        assert [t.name for t in tools] == _TRIO
        assert [endpoint for endpoint, _ in trio.listings] == [
            f"{_TRIO_ENDPOINT}&x-account-id=acc1%2Cacc3",
            f"{_TRIO_ENDPOINT}&x-account-id=acc2",
        ]
        assert len(calls) == 1

    def test_no_account_on_the_providers_means_no_tools_and_no_listing(self, monkeypatch):
        trio = _Trio(monkeypatch)
        _providers(monkeypatch, acc1="hris")
        toolset = StackOneToolSet(api_key="k", execute={"account_ids": ["acc1", "unlisted"]})
        assert len(toolset.fetch_tools(providers=["crm"], mode="search_execute")) == 0
        assert trio.listings == []

    def test_actions_still_filter_the_tool_names(self, monkeypatch):
        _Trio(monkeypatch)
        toolset = StackOneToolSet(api_key="k", account_id="acc1")
        tools = toolset.fetch_tools(actions=["*_actions"], mode="search_execute")
        assert [t.name for t in tools] == ["stackone_search_actions"]


class TestSearchHits:
    def test_one_search_spans_the_scope_and_top_k_cuts_its_hits(self, monkeypatch):
        trio = _Trio(monkeypatch)
        trio.hits = {
            "session_id": "s-1",
            "actions": [
                {"action_id": "hris_list", "available_accounts": [{"id": "acc1", "name": "A"}]},
                {"action_id": "crm_list", "available_accounts": [{"id": "acc2", "name": "B"}]},
                {"action_id": "hris_get", "available_accounts": [{"id": "acc1", "name": "A"}]},
            ],
        }
        toolset = StackOneToolSet(api_key="k", execute={"account_ids": ["acc1", "acc2"]})

        hits = toolset.search("list", top_k=2)

        assert [(name, args) for name, _, _, args in trio.calls] == [
            ("stackone_search_actions", {"query": "list", "top_k": 2})
        ]
        assert hits == [
            {
                "action_id": "hris_list",
                "available_accounts": [{"id": "acc1", "name": "A"}],
                "session_id": "s-1",
            },
            {
                "action_id": "crm_list",
                "available_accounts": [{"id": "acc2", "name": "B"}],
                "session_id": "s-1",
            },
        ]

    def test_account_ids_scope_the_listing_searched(self, monkeypatch):
        trio = _Trio(monkeypatch)
        toolset = StackOneToolSet(api_key="k", execute={"account_ids": ["acc1", "acc2"]})
        toolset.search("list", account_ids=["acc2"])
        [(_name, endpoint, _account, _args)] = trio.calls
        assert endpoint == f"{_TRIO_ENDPOINT}&x-account-id=acc2"

    def test_a_listing_without_the_search_tool_raises(self, monkeypatch):
        _Trio(monkeypatch, served=["stackone_submit_feedback"])
        toolset = StackOneToolSet(api_key="k", account_id="acc1")
        with pytest.raises(ToolsetLoadError) as excinfo:
            toolset.search("list")
        assert str(excinfo.value) == (
            "The server did not serve stackone_search_actions for this account scope."
        )


class TestExecuteArguments:
    """Pinned keys are set however the model filled the arguments, and action_id goes last,
    as in Node."""

    def _arguments(self, monkeypatch: pytest.MonkeyPatch, *args: Any, **kwargs: Any) -> Any:
        trio = _Trio(monkeypatch)
        toolset = StackOneToolSet(api_key="k", execute={"account_ids": ["acc1", "acc2"]})
        toolset.execute(*args, **kwargs)
        [(name, _endpoint, _account, arguments)] = trio.calls
        assert name == "stackone_execute_action"
        return arguments

    def test_one_account_id_becomes_stackone_account_id(self, monkeypatch):
        arguments = self._arguments(
            monkeypatch,
            "hris_list_employees",
            {"stackone_account_id": "model", "session_id": "model", "action_id": "x", "query": {"a": 1}},
            account_ids=["acc2", "acc2"],
            session_id="s-1",
        )
        assert list(arguments.items()) == [
            ("query", {"a": 1}),
            ("stackone_account_id", "acc2"),
            ("session_id", "s-1"),
            ("action_id", "hris_list_employees"),
        ]

    def test_several_or_no_account_ids_leave_the_choice_to_the_arguments(self, monkeypatch):
        arguments = self._arguments(
            monkeypatch,
            "hris_list_employees",
            {"session_id": "s-0", "stackone_account_id": "acc1", "query": {}},
            account_ids=["acc1", "acc2"],
        )
        assert list(arguments.items()) == [
            ("session_id", "s-0"),
            ("stackone_account_id", "acc1"),
            ("query", {}),
            ("action_id", "hris_list_employees"),
        ]

    def test_a_listing_without_the_execute_tool_raises(self, monkeypatch):
        _Trio(monkeypatch, served=["stackone_list_accounts"])
        toolset = StackOneToolSet(api_key="k", account_id="acc1")
        with pytest.raises(ToolsetLoadError) as excinfo:
            toolset.execute("hris_list_employees")
        assert str(excinfo.value) == (
            "The server did not serve stackone_execute_action for this account scope."
        )


class TestTooManyAccounts:
    def test_more_than_50_raise_before_any_request(self, monkeypatch):
        trio = _Trio(monkeypatch)
        toolset = StackOneToolSet(api_key="k", execute={"account_ids": [f"acc{i:02}" for i in range(51)]})
        with pytest.raises(ToolsetConfigError) as excinfo:
            toolset.search("list")
        assert str(excinfo.value) == (
            "51 accounts are in scope, but search and execute span at most 50 in one request. "
            "Pass account ids to narrow the scope."
        )
        assert trio.listings == []

    def test_50_are_one_request(self, monkeypatch):
        trio = _Trio(monkeypatch)
        accounts = [f"acc{i:02}" for i in range(50)]
        StackOneToolSet(api_key="k").fetch_tools(account_ids=accounts, mode="search_execute")
        [(endpoint, _headers)] = trio.listings
        assert endpoint == f"{_TRIO_ENDPOINT}&x-account-id={'%2C'.join(accounts)}"

    def test_individual_mode_has_no_cap(self, monkeypatch):
        accounts = _Accounts(monkeypatch, **{f"acc{i:02}": [] for i in range(51)})
        StackOneToolSet(api_key="k").fetch_tools(account_ids=sorted(accounts.accounts))
        assert len(accounts.listed) == 51


_PAIR_SEARCH = "mock_acc1_search_actions"
_PAIR_EXECUTE = "mock_acc1_execute_action"


class _Pair:
    """A fake one-account search_execute listing: acc1's pair and the feedback tool, and a
    record of every tools/call. ``answer`` is what the next call returns, or raises."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.answer: dict[str, Any] | Exception = {"isError": False, "result": {}}
        self.calls: list[tuple[str, str, dict[str, Any]]] = []
        served = [
            McpToolDefinition(
                _PAIR_SEARCH,
                f"Search acc1's actions. Run a hit with {_PAIR_EXECUTE}.",
                {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "session_id": {"type": "string", "description": f"From an earlier {_PAIR_SEARCH}."},
                    },
                    "required": ["query"],
                },
            ),
            McpToolDefinition(
                _PAIR_EXECUTE,
                f"Execute an action {_PAIR_SEARCH} found.",
                {
                    "type": "object",
                    "properties": {"action_id": {"type": "string"}},
                    "required": ["action_id"],
                },
            ),
            McpToolDefinition("stackone_submit_feedback", "Feedback.", {"type": "object", "properties": {}}),
        ]
        monkeypatch.setattr("stackone_ai.toolset.fetch_mcp_tools", lambda *_args, **_kwargs: served)
        monkeypatch.setattr(tools_module, "call_mcp_tool", self._call)

    def _call(self, endpoint: str, _headers: Any, name: str, arguments: Any, **_kwargs: Any) -> Any:
        self.calls.append((name, endpoint, arguments))
        if isinstance(self.answer, Exception):
            raise self.answer
        return self.answer


class TestOneAccountPair:
    """One account is served its own pair, which the SDK presents as the trio's two tools."""

    def _toolset(self) -> StackOneToolSet:
        return StackOneToolSet(api_key="k", account_id="acc1")

    def test_it_is_listed_under_the_trio_names_with_every_served_name_rewritten(self, monkeypatch):
        _Pair(monkeypatch)
        tools = self._toolset().fetch_tools(mode="search_execute")

        assert [t.name for t in tools] == [
            "stackone_search_actions",
            "stackone_execute_action",
            "stackone_submit_feedback",
        ]
        search, execute = tools[0], tools[1]
        assert search.description == "Search acc1's actions. Run a hit with stackone_execute_action."
        assert execute.description == "Execute an action stackone_search_actions found."
        assert search.to_openai_function()["function"]["parameters"]["properties"]["session_id"] == {
            "type": "string",
            "description": "From an earlier stackone_search_actions.",
        }
        assert all(t.get_account_id() is None for t in tools)

    def test_it_calls_the_served_tool_on_the_endpoint_that_listed_it(self, monkeypatch):
        pair = _Pair(monkeypatch)
        self._toolset().execute("mock_list_items", account_ids=["acc1"])
        assert pair.calls == [
            (_PAIR_EXECUTE, f"{_TRIO_ENDPOINT}&x-account-id=acc1", {"action_id": "mock_list_items"})
        ]

    def test_its_results_read_in_trio_names(self, monkeypatch):
        pair = _Pair(monkeypatch)
        pair.answer = {"session_id": "s-1", "actions": [], "hint": f"Call {_PAIR_EXECUTE} next."}
        tool = self._toolset().fetch_tools(mode="search_execute").get_tool("stackone_search_actions")
        assert tool is not None
        assert tool.execute({"query": "q"})["hint"] == "Call stackone_execute_action next."

    def test_its_errors_read_in_trio_names(self, monkeypatch):
        pair = _Pair(monkeypatch)
        pair.answer = StackOneAPIError(
            f'Tool "{_PAIR_EXECUTE}" failed: call {_PAIR_SEARCH} first', 400, {"error": f"see {_PAIR_SEARCH}"}
        )
        with pytest.raises(StackOneAPIError) as excinfo:
            self._toolset().execute("nope")
        assert str(excinfo.value) == (
            'Tool "stackone_execute_action" failed: call stackone_search_actions first'
        )
        assert (excinfo.value.status_code, excinfo.value.response_body) == (
            400,
            {"error": "see stackone_search_actions"},
        )

    def test_search_hits_gain_the_account_and_connector_when_absent(self, monkeypatch):
        pair = _Pair(monkeypatch)
        pair.answer = {
            "provider": "mock",
            "session_id": "s-1",
            "actions": [
                {"action_id": "mock_a"},
                {
                    "action_id": "mock_b",
                    "provider": "served",
                    "available_accounts": [{"id": "acc1", "name": "A"}],
                },
            ],
        }
        assert self._toolset().search("list") == [
            {
                "available_accounts": [{"id": "acc1"}],
                "provider": "mock",
                "action_id": "mock_a",
                "session_id": "s-1",
            },
            {
                "action_id": "mock_b",
                "provider": "served",
                "available_accounts": [{"id": "acc1", "name": "A"}],
                "session_id": "s-1",
            },
        ]
        assert pair.calls[0][:1] == (_PAIR_SEARCH,)

    def test_its_own_account_and_connector_are_dropped_from_the_arguments(self, monkeypatch):
        pair = _Pair(monkeypatch)
        pair.answer = {"actions": []}
        tools = self._toolset().fetch_tools(mode="search_execute")
        tools.get_tool("stackone_search_actions").execute(  # type: ignore[union-attr]
            {"query": "q", "provider": "MOCK", "stackone_account_id": "acc1"}
        )
        tools.get_tool("stackone_execute_action").execute(  # type: ignore[union-attr]
            {"action_id": "mock_a", "stackone_account_id": "acc1"}
        )
        # None is not given, as for session_id.
        tools.get_tool("stackone_execute_action").execute(  # type: ignore[union-attr]
            {"action_id": "mock_a", "stackone_account_id": None, "provider": None}
        )
        assert [arguments for _, _, arguments in pair.calls] == [
            {"query": "q"},
            {"action_id": "mock_a"},
            {"action_id": "mock_a"},
        ]

    def test_another_account_is_refused_before_any_call(self, monkeypatch):
        pair = _Pair(monkeypatch)
        tool = self._toolset().fetch_tools(mode="search_execute").get_tool("stackone_execute_action")
        assert tool is not None
        with pytest.raises(ToolArgumentsError) as excinfo:
            tool.execute({"action_id": "mock_a", "stackone_account_id": "acc2"})
        assert str(excinfo.value) == (
            'stackone_account_id "acc2" is not an account "stackone_execute_action" can run on; '
            'it runs on "acc1".'
        )
        assert pair.calls == []

    def test_another_provider_is_refused_before_any_call(self, monkeypatch):
        pair = _Pair(monkeypatch)
        tool = self._toolset().fetch_tools(mode="search_execute").get_tool("stackone_search_actions")
        assert tool is not None
        with pytest.raises(ToolArgumentsError) as excinfo:
            tool.execute({"query": "q", "provider": "hibob"})
        assert str(excinfo.value) == 'provider "hibob" is not linked in this account scope, which has "mock".'
        assert pair.calls == []

    def test_feedback_passes_through(self, monkeypatch):
        pair = _Pair(monkeypatch)
        self._toolset().submit_feedback("positive", ["stackone_search_actions"])
        [(name, _endpoint, arguments)] = pair.calls
        assert (name, arguments["tool_names"]) == ("stackone_submit_feedback", ["stackone_search_actions"])

    def test_a_tool_named_for_another_account_is_left_as_served(self, monkeypatch):
        _Pair(monkeypatch)
        tools = StackOneToolSet(api_key="k", account_id="acc2").fetch_tools(mode="search_execute")
        assert [t.name for t in tools] == [_PAIR_SEARCH, _PAIR_EXECUTE, "stackone_submit_feedback"]


def _providers(monkeypatch: pytest.MonkeyPatch, status: int = 200, **providers: str) -> list[int]:
    """Answer GET /accounts with each account on its provider, or fail with ``status``."""
    body = [
        {"id": account, "provider": provider, "status": "active"} for account, provider in providers.items()
    ]
    calls: list[int] = []

    def handle(_self: Any, request: httpx.Request) -> httpx.Response:
        calls.append(1)
        if status != 200:
            return httpx.Response(status, text="nope", request=request)
        return httpx.Response(200, json=body, request=request)

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", handle)
    return calls


class TestExtraHeadersCannotReplaceTheSdksOwn:
    def test_a_case_or_space_variant_of_an_sdk_owned_header_is_dropped(self):
        tool = StackOneMcpTool(
            name="t",
            description="",
            parameters=ToolParameters(type="object", properties={}),
            api_key="k",
            endpoint="https://api.example.com/mcp",
            account_id="acc1",
            headers={
                "authorization": "Bearer stolen",
                " X-Account-Id ": "someone-else",
                "USER-AGENT": "spoofed",
                "x-custom": "kept",
            },
        )
        headers = tool._prepare_headers()
        assert headers == {
            "x-custom": "kept",
            "User-Agent": headers["User-Agent"],
            "Authorization": "Basic azo=",
            "x-account-id": "acc1",
        }
        assert headers["User-Agent"].startswith("stackone-ai-python/")


def _duplicated() -> Tools:
    """hris_list on acc1 and acc2. Constructing it warns, so a test clears caplog after."""

    def tool(name: str, account: str) -> StackOneMcpTool:
        return StackOneMcpTool(
            name=name,
            description=f"on {account}",
            parameters=ToolParameters(type="object", properties={}),
            api_key="k",
            endpoint="https://api.example.com/mcp",
            account_id=account,
        )

    return Tools([tool("hris_list", "acc1"), tool("other", "acc1"), tool("hris_list", "acc2")])


class TestAdaptersKeepTheFirstOfEachName:
    DUPLICATE_WARNING = (
        "1 tool name(s) are served by more than one account (hris_list). The first one listed, "
        "from the lowest account id, is used — pass account ids to choose."
    )

    def test_openai(self, caplog):
        tools = _duplicated()
        caplog.clear()
        with caplog.at_level(logging.WARNING, logger="stackone.tools"):
            functions = tools.to_openai()
        assert [(f["function"]["name"], f["function"]["description"]) for f in functions] == [
            ("hris_list", "on acc1"),
            ("other", "on acc1"),
        ]
        assert [r.getMessage() for r in caplog.records] == [self.DUPLICATE_WARNING]

    def test_langchain(self, caplog):
        tools = _duplicated()
        caplog.clear()
        with caplog.at_level(logging.WARNING, logger="stackone.tools"):
            converted = tools.to_langchain()
        assert [(t.name, t.description) for t in converted] == [
            ("hris_list", "on acc1"),
            ("other", "on acc1"),
        ]
        assert [r.getMessage() for r in caplog.records] == [self.DUPLICATE_WARNING]

    def test_pydantic_ai(self, caplog):
        tools = _duplicated()
        caplog.clear()
        with caplog.at_level(logging.WARNING, logger="stackone.tools"):
            converted = tools.to_pydantic_ai()
        assert [(t.name, t.description) for t in converted] == [
            ("hris_list", "on acc1"),
            ("other", "on acc1"),
        ]
        assert [r.getMessage() for r in caplog.records] == [self.DUPLICATE_WARNING]

    def test_no_warning_without_a_clash(self, caplog):
        unique = _duplicated().tools[:2]
        caplog.clear()
        with caplog.at_level(logging.WARNING, logger="stackone.tools"):
            Tools(unique).to_openai()
        assert caplog.records == []


class TestAccountIdEnvironmentVariable:
    """3.x never reads STACKONE_ACCOUNT_ID; a 2.x setup relying on it is warned, once."""

    WARNING = (
        "STACKONE_ACCOUNT_ID is set, but the SDK does not read it: with no account id passed, every "
        "active shared account on this API key is used. Pass an account id to scope the toolset."
    )

    @pytest.mark.parametrize(
        ("value", "kwargs", "warned"),
        [
            pytest.param("acc1", {}, True, id="set-and-unscoped"),
            pytest.param("", {}, False, id="empty"),
            pytest.param(None, {}, False, id="unset"),
            pytest.param("acc1", {"account_id": "acc2"}, False, id="account-id"),
            pytest.param("acc1", {"execute": {"account_ids": ["acc2"]}}, False, id="account-ids"),
            pytest.param("acc1", {"execute": {"timeout": 5}}, True, id="execute-without-account-ids"),
            # An empty account_ids list is unset too, the same as _resolve_account_ids()
            # treats it: it must not swallow the warning.
            pytest.param("acc1", {"execute": {"account_ids": []}}, True, id="empty-account-ids"),
        ],
    )
    def test_warning(self, monkeypatch, caplog, value, kwargs, warned):
        if value is None:
            monkeypatch.delenv("STACKONE_ACCOUNT_ID", raising=False)
        else:
            monkeypatch.setenv("STACKONE_ACCOUNT_ID", value)
        with caplog.at_level(logging.WARNING, logger="stackone.tools"):
            StackOneToolSet(api_key="k", **kwargs)
        assert [r.getMessage() for r in caplog.records] == ([self.WARNING] if warned else [])

    def test_the_variable_is_still_never_read(self, monkeypatch):
        monkeypatch.setenv("STACKONE_ACCOUNT_ID", "from-env")
        accounts = _Accounts(monkeypatch, discovered=["t"])
        monkeypatch.setattr(
            StackOneToolSet, "fetch_accounts", lambda _self: [{"id": "discovered", "status": "active"}]
        )
        StackOneToolSet(api_key="k").fetch_tools()
        assert accounts.listed == ["discovered"]
