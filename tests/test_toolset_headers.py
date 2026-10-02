"""The toolset-level ``headers`` option: extra HTTP headers sent with every request."""

from __future__ import annotations

import logging
from typing import Any

import httpx
import pytest

from stackone_ai.tools import McpToolDefinition
from stackone_ai.toolset import StackOneToolSet


def _accounts_response(monkeypatch: pytest.MonkeyPatch, body: Any = None) -> list[httpx.Request]:
    """Answer GET /accounts with ``body`` (``[]`` if omitted); every request it received."""
    seen: list[httpx.Request] = []

    def handle(_self: Any, request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=body if body is not None else [], request=request)

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", handle)
    return seen


class _Mcp:
    """A fake MCP transport recording the headers of every listing and call."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.listed: list[dict[str, str]] = []
        self.called: list[dict[str, str]] = []
        monkeypatch.setattr("stackone_ai.toolset.fetch_mcp_tools", self._list)
        monkeypatch.setattr("stackone_ai.tools.call_mcp_tool", self._call)

    def _list(self, _endpoint: str, headers: dict[str, str], **_kwargs: Any) -> list[McpToolDefinition]:
        self.listed.append(headers)
        return [McpToolDefinition(name=f"tool_{headers['x-account-id']}", description="", input_schema={})]

    def _call(self, _endpoint: str, headers: dict[str, str], *_args: Any, **_kwargs: Any) -> dict[str, Any]:
        self.called.append(headers)
        return {"isError": False}


def test_warns_once_about_sdk_owned_names_and_ignores_them(caplog: pytest.LogCaptureFixture):
    with caplog.at_level(logging.WARNING, logger="stackone"):
        toolset = StackOneToolSet(
            api_key="k", headers={"Authorization": "Bearer x", "X-Account-Id": "spoofed", "X-Trace": "t"}
        )

    [message] = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert message == (
        'Ignoring headers "Authorization", "X-Account-Id": the SDK sets them itself. '
        "Pass the API key and account ids through their own options instead."
    )
    assert toolset._headers == {
        "Authorization": "Bearer x",
        "X-Account-Id": "spoofed",
        "X-Trace": "t",
    }


def test_does_not_warn_about_x_end_user_id_or_unrelated_names(caplog: pytest.LogCaptureFixture):
    with caplog.at_level(logging.WARNING, logger="stackone"):
        StackOneToolSet(api_key="k", headers={"x-end-user-id": "dave", "X-Trace": "t"})

    assert [r for r in caplog.records if r.levelno == logging.WARNING] == []


def test_reaches_get_accounts(monkeypatch: pytest.MonkeyPatch):
    seen = _accounts_response(monkeypatch)
    StackOneToolSet(api_key="k", headers={"X-Trace": "t"}).fetch_accounts()

    assert seen[0].headers["x-trace"] == "t"


def test_sdk_owned_names_are_dropped_beneath_the_sdks_own_on_get_accounts(monkeypatch: pytest.MonkeyPatch):
    seen = _accounts_response(monkeypatch)
    StackOneToolSet(api_key="k", headers={"Authorization": "Bearer spoofed"}).fetch_accounts()

    assert seen[0].headers["authorization"] != "Bearer spoofed"


def test_reaches_every_mcp_listing_and_call_request(monkeypatch: pytest.MonkeyPatch):
    _accounts_response(monkeypatch)
    mcp = _Mcp(monkeypatch)
    toolset = StackOneToolSet(api_key="k", headers={"X-Trace": "t"})

    tools = toolset.fetch_tools(account_ids=["acc1", "acc2"])
    for tool in tools:
        tool.execute({})

    assert all(h.get("X-Trace") == "t" for h in mcp.listed)
    assert all(h.get("X-Trace") == "t" for h in mcp.called)


def test_sdk_owned_names_are_dropped_beneath_the_sdks_own_on_mcp_requests(monkeypatch: pytest.MonkeyPatch):
    _accounts_response(monkeypatch)
    mcp = _Mcp(monkeypatch)
    toolset = StackOneToolSet(api_key="k", headers={"x-account-id": "spoofed", "User-Agent": "spoofed"})

    tool = toolset.fetch_tools(account_ids=["acc1"])[0]
    tool.execute({})

    assert mcp.listed[0]["x-account-id"] == "acc1"
    assert mcp.called[0]["x-account-id"] == "acc1"
    assert "spoofed" not in mcp.listed[0]["User-Agent"]
    assert "spoofed" not in mcp.called[0]["User-Agent"]


def test_x_end_user_id_is_passed_through_as_given(monkeypatch: pytest.MonkeyPatch):
    _accounts_response(monkeypatch)
    mcp = _Mcp(monkeypatch)
    toolset = StackOneToolSet(api_key="k", account_id="acc1", headers={"x-end-user-id": "caller"})

    tool = toolset.fetch_tools()[0]
    tool.execute({})

    assert mcp.listed[0]["x-end-user-id"] == "caller"
    assert mcp.called[0]["x-end-user-id"] == "caller"


def test_a_recorded_end_user_id_overrides_the_configured_one(monkeypatch: pytest.MonkeyPatch):
    _accounts_response(
        monkeypatch,
        [
            {
                "id": "acc1",
                "provider": "p",
                "status": "active",
                "shared": False,
                "origin_username": "recorded-user",
            }
        ],
    )
    mcp = _Mcp(monkeypatch)
    toolset = StackOneToolSet(api_key="k", account_id="acc1", headers={"x-end-user-id": "caller"})
    toolset.fetch_accounts()

    tool = toolset.fetch_tools()[0]
    tool.execute({})

    assert mcp.listed[0]["x-end-user-id"] == "recorded-user"
    assert mcp.called[0]["x-end-user-id"] == "recorded-user"
