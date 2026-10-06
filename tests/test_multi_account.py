"""More than one account: the catalog cache, search hits, routing and adapters.

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


def _meta_tools(monkeypatch: pytest.MonkeyPatch, per_account: dict[str, list[str]]) -> dict[str, Any]:
    """Serve ``per_account``'s meta tools, and record what each one is called with."""
    seen: dict[str, Any] = {}

    def fake_execute(self: StackOneMcpTool, arguments: Any) -> dict[str, Any]:
        seen["tool"], seen["account"], seen["arguments"] = self.name, self.get_account_id(), arguments
        if self.name.endswith("_search_actions"):
            return {"session_id": f"s-{self.get_account_id()}", "actions": seen["hits"][self.name]}
        return {"isError": False, "result": {}}

    monkeypatch.setattr(StackOneMcpTool, "execute", fake_execute)
    monkeypatch.setattr(
        "stackone_ai.toolset.fetch_mcp_tools",
        lambda _e, headers, **_k: [_tool(name) for name in per_account[headers["x-account-id"]]],
    )
    return seen


class TestSearchHits:
    def test_each_hit_names_its_account_and_top_k_cuts_the_merged_ranking(self, monkeypatch):
        seen = _meta_tools(
            monkeypatch,
            {
                "acc1": ["hris_acc1_search_actions"],
                "acc2": ["hris_acc2_search_actions", "crm_acc2_search_actions"],
            },
        )
        seen["hits"] = {
            "hris_acc1_search_actions": [
                {"action_id": "hris_list", "similarity_score": 0.9},
                {"action_id": "hris_get", "similarity_score": 0.4},
            ],
            "hris_acc2_search_actions": [
                {"action_id": "hris_list", "similarity_score": 0.8},
                {"action_id": "hris_get", "similarity_score": 0.3},
            ],
            "crm_acc2_search_actions": [{"action_id": "crm_list", "similarity_score": 0.7}],
        }
        toolset = StackOneToolSet(api_key="k", execute={"account_ids": ["acc1", "acc2"]})

        hits = toolset.search("list", top_k=3)

        # top_k is the total, not per connector: 3 connectors asked for 3 each served 5.
        assert seen["arguments"]["top_k"] == 3
        assert [(h["action_id"], h["account_id"], h["session_id"]) for h in hits] == [
            ("hris_list", "acc1", "s-acc1"),
            ("hris_list", "acc2", "s-acc2"),
            ("crm_list", "acc2", "s-acc2"),
        ]


class TestExecuteRouting:
    PER_ACCOUNT = {
        "acc1": ["hris_acc1_execute_action"],
        "acc2": ["hris_acc2_execute_action"],
    }

    def test_a_connector_on_two_accounts_is_refused(self, monkeypatch):
        seen = _meta_tools(monkeypatch, self.PER_ACCOUNT)
        toolset = StackOneToolSet(api_key="k", execute={"account_ids": ["acc2", "acc1"]})
        with pytest.raises(ToolsetConfigError) as excinfo:
            toolset.execute("hris_list_employees")
        assert str(excinfo.value) == (
            '"hris_list_employees" matches 2 connectors on different accounts '
            "(hris_acc1_execute_action on acc1, hris_acc2_execute_action on acc2). "
            "Pass the account id to use, such as a search hit's account_id."
        )
        assert "tool" not in seen

    @pytest.mark.parametrize("account", ["acc1", "acc2"])
    def test_a_hits_account_id_routes_to_it(self, monkeypatch, account):
        seen = _meta_tools(monkeypatch, self.PER_ACCOUNT)
        toolset = StackOneToolSet(api_key="k", execute={"account_ids": ["acc1", "acc2"]})
        toolset.execute("hris_list_employees", account_ids=[account])
        assert (seen["tool"], seen["account"]) == (f"hris_{account}_execute_action", account)


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


class TestExecuteWithAFailedAccount:
    """An account that failed to list could have served the action: running it on another
    account instead would run it on an account the caller never chose.
    """

    def _execute(self, monkeypatch: pytest.MonkeyPatch, **accounts: list[str] | Exception) -> Any:
        listing = _Accounts(monkeypatch, **accounts)
        seen: dict[str, Any] = {}

        def fake_execute(tool: StackOneMcpTool, arguments: Any) -> dict[str, Any]:
            seen["account"] = tool.get_account_id()
            return {"isError": False}

        monkeypatch.setattr(StackOneMcpTool, "execute", fake_execute)
        return listing, seen

    def test_an_account_on_the_actions_connector_refuses(self, monkeypatch):
        _, seen = self._execute(monkeypatch, acc1=RuntimeError("boom"), acc2=["hris_acc2_execute_action"])
        _providers(monkeypatch, acc1="HRIS", acc2="hris")
        toolset = StackOneToolSet(api_key="k", execute={"account_ids": ["acc2", "acc1"]})
        toolset.fetch_accounts()
        with pytest.raises(ToolsetLoadError) as excinfo:
            toolset.execute("hris_list_employees")
        assert str(excinfo.value) == (
            '"hris_list_employees" may be served by an account that failed to list (acc1: boom). '
            "Pass the account id to use, such as a search hit's account_id."
        )
        assert seen == {}

    def test_accounts_with_no_known_provider_refuse_in_account_order(self, monkeypatch):
        """GET /accounts failed, so either could be on the action's connector."""
        _, seen = self._execute(
            monkeypatch,
            acc3=RuntimeError("down"),
            acc1=RuntimeError("boom"),
            acc2=["hris_acc2_execute_action"],
        )
        calls = _providers(monkeypatch, status=403)
        toolset = StackOneToolSet(api_key="k", execute={"account_ids": ["acc3", "acc2", "acc1"]})
        with pytest.raises(ToolsetLoadError, match=r"failed to list \(acc1: boom; acc3: down\)\."):
            toolset.execute("hris_list_employees")
        assert seen == {}
        assert len(calls) == 1

    def test_explicit_ids_learn_their_providers_with_one_accounts_listing(self, monkeypatch, clock):
        """With explicit ids no GET /accounts was made, so execute() makes one to learn the
        failed account's provider before deciding."""
        listing, seen = self._execute(
            monkeypatch, acc1=RuntimeError("dead"), acc2=["hris_acc2_execute_action"]
        )
        calls = _providers(monkeypatch, acc1="crm", acc2="hris")
        toolset = StackOneToolSet(api_key="k", execute={"account_ids": ["acc1", "acc2"]})
        toolset.execute("hris_list_employees")
        assert seen == {"account": "acc2"}
        assert len(calls) == 1

        # Known now: the next call neither lists accounts nor re-lists the dead one.
        listing.listed.clear()
        toolset.execute("hris_list_employees")
        assert (len(calls), listing.listed) == (1, [])

    def test_a_rate_limited_lookup_is_raised(self, monkeypatch):
        """The 429 is the key's: swallowed, it would read as a provider no lookup has named."""
        monkeypatch.setattr(tools_module, "_sleep", lambda _delay: None)
        self._execute(monkeypatch, acc1=RuntimeError("boom"), acc2=["hris_acc2_execute_action"])
        _providers(monkeypatch, status=429)
        toolset = StackOneToolSet(api_key="k", execute={"account_ids": ["acc1", "acc2"]})
        with pytest.raises(StackOneAPIError) as excinfo:
            toolset.execute("hris_list_employees")
        assert excinfo.value.status_code == 429

    @pytest.mark.parametrize(
        "accounts",
        [{"status": 403}, {"acc1": "hris"}],
        ids=["the lookup fails", "the lookup does not list it"],
    )
    def test_a_lookup_that_misses_waits_out_the_failure_window(self, monkeypatch, clock, accounts):
        """Looking up and listing it again on every call would pay a GET /accounts each
        time, and the listing's timeout if it hangs."""
        listing, seen = self._execute(
            monkeypatch, acc1=["hris_acc1_execute_action"], acc2=RuntimeError("dead")
        )
        calls = _providers(monkeypatch, **accounts)
        toolset = StackOneToolSet(api_key="k", execute={"account_ids": ["acc1", "acc2"]})
        for _ in range(3):
            with pytest.raises(ToolsetLoadError, match=r"failed to list \(acc2: dead\)"):
                toolset.execute("hris_list_employees")
        assert seen == {}
        assert (len(calls), listing.listed.count("acc2")) == (1, 1)

        clock[0] += FAILED_ACCOUNT_RETRY_SECONDS
        with pytest.raises(ToolsetLoadError, match=r"failed to list \(acc2: dead\)"):
            toolset.execute("hris_list_employees")
        assert (len(calls), listing.listed.count("acc2")) == (2, 2)

    def test_clearing_the_cache_forgets_a_lookups_miss(self, monkeypatch, clock):
        self._execute(monkeypatch, acc1=["hris_acc1_execute_action"], acc2=RuntimeError("dead"))
        calls = _providers(monkeypatch, status=403)
        toolset = StackOneToolSet(api_key="k", execute={"account_ids": ["acc1", "acc2"]})
        with pytest.raises(ToolsetLoadError):
            toolset.execute("hris_list_employees")
        toolset.clear_catalog_cache()
        with pytest.raises(ToolsetLoadError):
            toolset.execute("hris_list_employees")
        assert len(calls) == 2

    def test_the_call_whose_lookup_misses_does_not_list_it_again(self, monkeypatch, clock):
        """Its own miss holds it out at once, so concurrent calls sharing the lookup don't
        each pay its timeout."""
        listing, _ = self._execute(monkeypatch, acc1=["hris_acc1_execute_action"], acc2=RuntimeError("dead"))
        _providers(monkeypatch, status=403)
        toolset = StackOneToolSet(api_key="k", execute={"account_ids": ["acc1", "acc2"]})
        toolset.fetch_tools(mode="search_execute")
        clock[0] += 1
        with pytest.raises(ToolsetLoadError, match=r"failed to list \(acc2: dead\)"):
            toolset.execute("hris_list_employees")
        assert listing.listed.count("acc2") == 1

    def test_a_miss_is_not_recorded_across_a_clear_during_its_lookup(self, monkeypatch, clock):
        self._execute(monkeypatch, acc1=["hris_acc1_execute_action"], acc2=RuntimeError("dead"))
        toolset = StackOneToolSet(api_key="k", execute={"account_ids": ["acc1", "acc2"]})
        calls: list[int] = []

        def handle(_self: Any, request: httpx.Request) -> httpx.Response:
            calls.append(1)
            if len(calls) == 1:
                toolset.clear_catalog_cache()
            return httpx.Response(403, text="nope", request=request)

        monkeypatch.setattr(httpx.HTTPTransport, "handle_request", handle)
        for _ in range(2):
            with pytest.raises(ToolsetLoadError):
                toolset.execute("hris_list_employees")
        assert len(calls) == 2

    def test_a_miss_is_forgotten_once_get_accounts_names_the_account(self, monkeypatch, clock):
        listing, _ = self._execute(monkeypatch, acc1=["hris_acc1_execute_action"], acc2=RuntimeError("dead"))
        _providers(monkeypatch, status=403)
        toolset = StackOneToolSet(api_key="k", execute={"account_ids": ["acc1", "acc2"]})
        with pytest.raises(ToolsetLoadError):
            toolset.execute("hris_list_employees")
        _providers(monkeypatch, acc1="hris", acc2="hris")
        toolset.fetch_accounts()

        clock[0] += 1
        with pytest.raises(ToolsetLoadError):
            toolset.execute("hris_list_employees")
        # acc2 is hris's, so it is listed again at once rather than after the window.
        assert listing.listed.count("acc2") == 2

    def test_an_account_on_another_provider_does_not(self, monkeypatch):
        _, seen = self._execute(monkeypatch, acc1=RuntimeError("boom"), acc2=["hris_acc2_execute_action"])
        _providers(monkeypatch, acc1="crm", acc2="hris")
        toolset = StackOneToolSet(api_key="k", execute={"account_ids": ["acc1", "acc2"]})
        toolset.fetch_accounts()
        toolset.execute("hris_list_employees")
        assert seen == {"account": "acc2"}

    def test_a_failed_account_on_a_shorter_provider_prefix_does_not(self, monkeypatch):
        """acc1's provider ``hris`` prefixes the action, but its connector is ``hris_beta``."""
        _, seen = self._execute(
            monkeypatch, acc1=RuntimeError("boom"), acc2=["hris_beta_acc2_execute_action"]
        )
        _providers(monkeypatch, acc1="hris", acc2="hris_beta")
        toolset = StackOneToolSet(api_key="k", execute={"account_ids": ["acc1", "acc2"]})
        toolset.fetch_accounts()
        toolset.execute("hris_beta_list_employees")
        assert seen == {"account": "acc2"}

    def test_a_failed_account_is_listed_again_at_once_and_then_routes_as_usual(self, monkeypatch, clock):
        listing, _ = self._execute(monkeypatch, acc1=RuntimeError("boom"), acc2=["hris_acc2_execute_action"])
        _providers(monkeypatch, acc1="hris", acc2="hris")
        toolset = StackOneToolSet(api_key="k", execute={"account_ids": ["acc1", "acc2"]})
        # Left out of the search_execute catalog, which execute() lists in, for 30s from now.
        assert [t.name for t in toolset.fetch_tools(mode="search_execute")] == ["hris_acc2_execute_action"]

        listing.accounts["acc1"] = ["hris_acc1_execute_action"]
        listing.listed.clear()
        with pytest.raises(ToolsetConfigError, match="matches 2 connectors on different accounts"):
            toolset.execute("hris_list_employees")
        # Only the failed account, and without waiting for it to be due.
        assert listing.listed == ["acc1"]

        # Its listing is cached for the next call, like any other.
        listing.listed.clear()
        toolset.fetch_tools(mode="search_execute")
        assert listing.listed == []

    def test_an_account_that_fails_during_the_call_is_not_listed_again_in_it(self, monkeypatch, clock):
        """Listing it again at once, as Node once did, would only double the wait."""
        listing, _ = self._execute(monkeypatch, acc1=RuntimeError("hangs"), acc2=["hris_acc2_execute_action"])
        _providers(monkeypatch, acc1="hris", acc2="hris")
        toolset = StackOneToolSet(api_key="k", execute={"account_ids": ["acc1", "acc2"]})
        with pytest.raises(ToolsetLoadError):
            toolset.execute("hris_list_employees")
        assert sorted(listing.listed) == ["acc1", "acc2"]

    def test_a_failed_account_on_another_provider_is_not_listed_again(self, monkeypatch, clock):
        """A hanging crm account must not cost every hris action its timeout."""
        listing, seen = self._execute(
            monkeypatch, acc1=RuntimeError("hangs"), acc2=["hris_acc2_execute_action"]
        )
        _providers(monkeypatch, acc1="crm", acc2="hris")
        toolset = StackOneToolSet(api_key="k", execute={"account_ids": ["acc1", "acc2"]})
        toolset.fetch_accounts()
        toolset.execute("hris_list_employees")

        listing.listed.clear()
        toolset.execute("hris_list_employees")
        assert listing.listed == []
        assert seen == {"account": "acc2"}

        # One that could serve the action is still listed again at once.
        with pytest.raises(ToolsetLoadError, match=r"failed to list \(acc1: hangs\)"):
            toolset.execute("crm_list_contacts")
        assert listing.listed == ["acc1"]


class TestPinnedArgumentsGoLast:
    """A pinned session_id and action_id are moved to the end, as in Node."""

    def test_a_supplied_session_id_is_moved_before_action_id(self, monkeypatch):
        seen = _meta_tools(monkeypatch, {"acc1": ["hris_acc1_execute_action"]})
        toolset = StackOneToolSet(api_key="k", account_id="acc1")
        toolset.execute(
            "hris_list_employees",
            {"session_id": "model-chosen", "action_id": "x", "query": {"a": 1}},
            session_id="s-1",
        )
        assert list(seen["arguments"].items()) == [
            ("query", {"a": 1}),
            ("session_id", "s-1"),
            ("action_id", "hris_list_employees"),
        ]

    def test_an_unpinned_session_id_stays_where_it_was(self, monkeypatch):
        seen = _meta_tools(monkeypatch, {"acc1": ["hris_acc1_execute_action"]})
        toolset = StackOneToolSet(api_key="k", account_id="acc1")
        toolset.execute("hris_list_employees", {"session_id": "s-0", "query": {}})
        assert list(seen["arguments"]) == ["session_id", "query", "action_id"]


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
