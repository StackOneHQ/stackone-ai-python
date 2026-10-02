"""More than one account: the catalog cache, search hits, routing and adapters.

With discovery as the default, the same provider linked on two accounts is common, so
every surface has to say which account it means.
"""

from __future__ import annotations

import logging

import pytest

from stackone_ai import toolset as toolset_module
from stackone_ai.tools import McpToolDefinition
from stackone_ai.toolset import FAILED_ACCOUNT_RETRY_SECONDS, StackOneToolSet
from stackone_ai.types import (
    StackOneAPIError,
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
        assert excinfo.value.__cause__ is unauthorised

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
