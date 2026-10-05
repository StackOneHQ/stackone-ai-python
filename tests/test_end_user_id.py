"""x-end-user-id: the end user a non-shared account belongs to.

The API refuses an MCP request for a non-shared account that does not carry the account's
end-user id. GET /accounts names it (``origin_username``), so whenever the SDK lists
accounts it records it, and sends it on every MCP request for that account.
"""

from __future__ import annotations

import logging
import threading
from typing import Any

import httpx
import pytest

from stackone_ai.tools import McpToolDefinition, StackOneMcpTool
from stackone_ai.toolset import StackOneToolSet
from stackone_ai.types import StackOneAPIError, ToolParameters

UCA_REFUSAL = "x-end-user-id header does not match account end user id for account acc1"


def _reset(base_url: str) -> None:
    httpx.delete(f"{base_url}/__requests").raise_for_status()


def _headers(base_url: str) -> list[dict[str, Any]]:
    """Every JSON-RPC message the mock received: its method, x-account-id and x-end-user-id."""
    response = httpx.get(f"{base_url}/__headers")
    response.raise_for_status()
    return response.json()


def _end_user_ids(base_url: str, account: str) -> set[str | None]:
    return {r["endUserId"] for r in _headers(base_url) if r["accountId"] == account}


def _methods(base_url: str, account: str) -> set[str]:
    return {r["method"] for r in _headers(base_url) if r["accountId"] == account}


class TestOverTheWire:
    """Against a mock that, like the API, refuses acc1's requests without acc1's end user."""

    def test_a_query_scoped_request_without_the_header_is_refused(self, mcp_mock_server_with_end_users: str):
        """The account id can arrive as a query parameter, as the real mounted MCP app allows."""
        base_url = mcp_mock_server_with_end_users
        response = httpx.post(
            f"{base_url}/mcp?x-account-id=acc1",
            auth=("test-key", ""),
            json={"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}},
        )
        assert response.status_code == 400
        assert response.json()["message"] == UCA_REFUSAL

    def test_discovery_sends_it_on_every_listing_request(self, mcp_mock_server_with_end_users: str):
        base_url = mcp_mock_server_with_end_users
        _reset(base_url)
        tools = StackOneToolSet(api_key="test-key", base_url=base_url).fetch_tools()

        assert {"acc1_tool_1", "acc2_tool_1"} <= {t.name for t in tools}
        assert {"initialize", "tools/list"} <= _methods(base_url, "acc1")
        assert _end_user_ids(base_url, "acc1") == {"end-user-1"}
        # Shared: no end user, so no header.
        assert _end_user_ids(base_url, "acc2") == {None}

    def test_a_listed_tool_sends_it_on_every_call_request(self, mcp_mock_server_with_end_users: str):
        base_url = mcp_mock_server_with_end_users
        tools = StackOneToolSet(api_key="test-key", base_url=base_url).fetch_tools()
        _reset(base_url)

        for name in ("acc1_tool_1", "acc2_tool_1"):
            tool = tools.get_tool(name)
            assert tool is not None
            tool.execute({"fields": "id"})

        assert {"initialize", "tools/call"} <= _methods(base_url, "acc1")
        assert _end_user_ids(base_url, "acc1") == {"end-user-1"}
        assert _end_user_ids(base_url, "acc2") == {None}

    def test_meta_tools_and_feedback_send_it(self, mcp_mock_server_with_end_users: str):
        base_url = mcp_mock_server_with_end_users
        toolset = StackOneToolSet(api_key="test-key", base_url=base_url)
        _reset(base_url)

        hits = toolset.search("list items", account_ids=None)
        assert any(hit["account_id"] == "acc1" for hit in hits)
        toolset.execute("mock_list_items", {}, account_ids=["acc1"])
        toolset.submit_feedback("positive", ["mock_list_items"], account_ids=["acc1"])

        called = {
            r.get("name") for r in httpx.get(f"{base_url}/__requests").json() if r["accountId"] == "acc1"
        }
        assert {"mock_acc1_search_actions", "mock_acc1_execute_action", "stackone_submit_feedback"} <= called
        assert _end_user_ids(base_url, "acc1") == {"end-user-1"}

    def test_a_direct_fetch_accounts_records_it_for_explicit_ids(self, mcp_mock_server_with_end_users: str):
        base_url = mcp_mock_server_with_end_users
        toolset = StackOneToolSet(api_key="test-key", base_url=base_url, account_id="acc1")
        toolset.fetch_accounts()
        _reset(base_url)

        tool = toolset.fetch_tools().get_tool("acc1_tool_1")
        assert tool is not None
        tool.execute({})

        assert _end_user_ids(base_url, "acc1") == {"end-user-1"}

    def test_explicit_ids_list_no_accounts_and_send_no_end_user(self, mcp_mock_server_with_end_users: str):
        """No GET /accounts was made, so there is nothing to send: the API refuses acc1."""
        base_url = mcp_mock_server_with_end_users
        toolset = StackOneToolSet(api_key="test-key", base_url=base_url)
        _reset(base_url)

        assert toolset.fetch_tools(account_ids=["acc2"]).get_tool("acc2_tool_1") is not None
        with pytest.raises(StackOneAPIError, match=UCA_REFUSAL) as excinfo:
            toolset.fetch_tools(account_ids=["acc1"])
        assert excinfo.value.status_code == 400
        assert _end_user_ids(base_url, "acc2") == {None}
        assert _end_user_ids(base_url, "acc1") == {None}

    def test_a_model_supplied_end_user_is_not_forwarded(self, mcp_mock_server_with_end_users: str):
        base_url = mcp_mock_server_with_end_users
        toolset = StackOneToolSet(api_key="test-key", base_url=base_url)
        toolset.fetch_accounts()
        _reset(base_url)

        toolset.execute(
            "mock_list_items",
            {"headers": {"x-custom": "kept", "X-End-User-Id": "victim"}, "headers_x-end-user-id": "victim"},
            account_ids=["acc1"],
        )

        [call] = [r for r in httpx.get(f"{base_url}/__requests").json() if r["method"] == "tools/call"]
        assert call["arguments"] == {"headers": {"x-custom": "kept"}, "action_id": "mock_list_items"}
        assert _end_user_ids(base_url, "acc1") == {"end-user-1"}


def _accounts_response(monkeypatch: pytest.MonkeyPatch, *bodies: Any) -> list[int]:
    """Answer each GET /accounts with the next body: a JSON value, or an int status to fail with."""
    queue = list(bodies)
    calls: list[int] = []

    def handle(_self: Any, request: httpx.Request) -> httpx.Response:
        calls.append(1)
        body = queue.pop(0)
        if isinstance(body, int):
            return httpx.Response(body, text="nope", request=request)
        return httpx.Response(200, json=body, request=request)

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", handle)
    return calls


def _account(account_id: str, **fields: Any) -> dict[str, Any]:
    return {"id": account_id, "provider": "p", "status": "active", **fields}


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


class TestRecording:
    def test_only_a_non_shared_account_with_a_username_is_recorded(self, monkeypatch):
        _accounts_response(
            monkeypatch,
            [
                _account("private", shared=False, origin_username="user-1"),
                _account("shared", shared=True, origin_username="user-2"),
                _account("unknown", origin_username="user-3"),
                _account("truthy", shared="false", origin_username="user-4"),
                _account("zero", shared=0, origin_username="user-5"),
                _account("empty", shared=False, origin_username=""),
                _account("null", shared=False, origin_username=None),
                _account("number", shared=False, origin_username=7),
                {"shared": False, "origin_username": "no-id"},
                "not-an-account",
            ],
        )
        toolset = StackOneToolSet(api_key="k")
        toolset.fetch_accounts()
        recorded = {
            account: toolset._end_user_id(account)
            for account in ("private", "shared", "unknown", "truthy", "zero", "empty", "null", "number")
        }
        assert recorded == {
            "private": "user-1",
            "shared": None,
            "unknown": None,
            "truthy": None,
            "zero": None,
            "empty": None,
            "null": None,
            "number": None,
        }

    def test_each_successful_listing_replaces_the_record(self, monkeypatch):
        _accounts_response(
            monkeypatch,
            {"data": [_account("a", shared=False, origin_username="user-a")]},
            [_account("b", shared=False, origin_username="user-b"), _account("a", shared=True)],
            500,
        )
        toolset = StackOneToolSet(api_key="k")
        toolset.fetch_accounts()
        assert toolset._end_user_id("a") == "user-a"

        toolset.fetch_accounts()
        assert (toolset._end_user_id("a"), toolset._end_user_id("b")) == (None, "user-b")

        # A failed listing records nothing, so it keeps the last one.
        with pytest.raises(StackOneAPIError):
            toolset.fetch_accounts()
        assert toolset._end_user_id("b") == "user-b"

    def test_clearing_the_catalog_cache_keeps_the_record(self, monkeypatch):
        _accounts_response(monkeypatch, [_account("a", shared=False, origin_username="user-a")])
        toolset = StackOneToolSet(api_key="k")
        toolset.fetch_accounts()
        toolset.clear_catalog_cache()
        toolset.set_accounts(["a"])
        assert toolset._end_user_id("a") == "user-a"

    def test_an_older_started_call_does_not_overwrite_a_newer_ones_record(self, monkeypatch):
        """fetch_accounts() racing another, e.g. via discovery: start order wins, not finish order."""
        toolset = StackOneToolSet(api_key="k")
        first_started = threading.Event()
        second_started = threading.Event()
        release_first = threading.Event()
        release_second = threading.Event()

        def handle(_self: Any, request: httpx.Request) -> httpx.Response:
            if not first_started.is_set():
                first_started.set()
                assert release_first.wait(timeout=5)
                body = [_account("a", shared=False, origin_username="user-a")]
            else:
                second_started.set()
                assert release_second.wait(timeout=5)
                body = [_account("b", shared=False, origin_username="user-b")]
            return httpx.Response(200, json=body, request=request)

        monkeypatch.setattr(httpx.HTTPTransport, "handle_request", handle)

        first = threading.Thread(target=toolset.fetch_accounts)
        first.start()
        assert first_started.wait(timeout=5)

        second = threading.Thread(target=toolset.fetch_accounts)
        second.start()
        assert second_started.wait(timeout=5)

        # Started second, finishes first.
        release_second.set()
        second.join(timeout=5)
        # Started first, finishes last: its stale response must not overwrite the
        # newer-started call's record.
        release_first.set()
        first.join(timeout=5)

        assert toolset._end_user_id("b") == "user-b"
        assert toolset._end_user_id("a") is None


class TestHeaders:
    def test_discovery_sends_it_on_listing_and_execution(self, monkeypatch):
        _accounts_response(
            monkeypatch,
            [_account("a", shared=False, origin_username="user-a"), _account("b", shared=True)],
        )
        mcp = _Mcp(monkeypatch)
        tools = StackOneToolSet(api_key="k").fetch_tools()
        for tool in tools:
            tool.execute({})

        by_account = {h["x-account-id"]: h.get("x-end-user-id") for h in mcp.listed}
        assert by_account == {"a": "user-a", "b": None}
        assert {h["x-account-id"]: h.get("x-end-user-id") for h in mcp.called} == by_account

    def test_explicit_ids_make_no_accounts_request_and_send_none(self, monkeypatch):
        calls = _accounts_response(monkeypatch)
        mcp = _Mcp(monkeypatch)
        StackOneToolSet(api_key="k", account_id="a").fetch_tools()[0].execute({})

        assert calls == []
        assert "x-end-user-id" not in mcp.listed[0]
        assert "x-end-user-id" not in mcp.called[0]

    def test_a_tool_built_before_the_listing_picks_it_up(self, monkeypatch):
        _accounts_response(
            monkeypatch,
            [
                _account("a", shared=False, origin_username="user-a"),
                _account("b", shared=False, origin_username="user-b"),
            ],
        )
        mcp = _Mcp(monkeypatch)
        toolset = StackOneToolSet(api_key="k", account_id="a")
        tool = toolset.fetch_tools()[0]
        toolset.fetch_accounts()

        tool.execute({})
        # And follows the tool's account when it is rebound.
        tool.set_account_id("b")
        tool.execute({})

        assert [h.get("x-end-user-id") for h in mcp.called] == ["user-a", "user-b"]

    def test_it_overrides_a_configured_header_which_is_otherwise_sent_as_given(self, monkeypatch):
        _accounts_response(monkeypatch, [_account("a", shared=False, origin_username="user-a")])
        mcp = _Mcp(monkeypatch)
        toolset = StackOneToolSet(api_key="k")

        def configured(account_id: str) -> StackOneMcpTool:
            tool = StackOneMcpTool(
                name="t",
                description="",
                parameters=ToolParameters(type="object", properties={}),
                api_key="k",
                endpoint="https://api.example.com/mcp",
                account_id=account_id,
                headers={" X-End-User-Id": "caller"},
            )
            tool._end_user_id_of = toolset._end_user_id
            return tool

        with_record, without_record = configured("a"), configured("other")
        without_record.execute({})
        toolset.fetch_accounts()
        with_record.execute({})

        assert mcp.called[0][" X-End-User-Id"] == "caller"
        assert "x-end-user-id" not in mcp.called[0]
        assert [name for name in mcp.called[1] if name.strip().lower() == "x-end-user-id"] == [
            "x-end-user-id"
        ]
        assert mcp.called[1]["x-end-user-id"] == "user-a"


class TestModelSuppliedHeader:
    """A model may never pick the end user, even when the schema declares the header."""

    @pytest.fixture
    def called(self, monkeypatch) -> list[dict[str, Any]]:
        seen: list[dict[str, Any]] = []
        monkeypatch.setattr(
            "stackone_ai.tools.call_mcp_tool",
            lambda _endpoint, headers, _name, arguments, **_kwargs: seen.append(
                {"headers": headers, "arguments": arguments}
            )
            or {},
        )
        return seen

    @staticmethod
    def _tool(properties: dict[str, Any]) -> StackOneMcpTool:
        return StackOneMcpTool(
            name="t",
            description="",
            parameters=ToolParameters(type="object", properties=properties),
            api_key="k",
            endpoint="https://api.example.com/mcp",
            account_id="a",
        )

    @pytest.mark.parametrize("name", ["x-end-user-id", "X-End-User-Id", " X-END-USER-ID\t"])
    @pytest.mark.parametrize(
        "headers_schema",
        [
            pytest.param({"type": "object"}, id="open"),
            pytest.param(
                {"type": "object", "properties": {"x-end-user-id": {"type": "string"}}}, id="declared"
            ),
        ],
    )
    def test_a_nested_entry_is_dropped(self, called, caplog, name, headers_schema):
        with caplog.at_level(logging.WARNING, logger="stackone.tools"):
            self._tool({"headers": headers_schema}).execute({"headers": {name: "victim", "x-other": "1"}})

        expected = {"x-other": "1"} if headers_schema.get("properties") is None else {}
        assert called[0]["arguments"] == {"headers": expected}
        assert "x-end-user-id" not in called[0]["headers"]
        assert f'Dropping header "{name.strip()}" from a tool call: set by the SDK' in caplog.text

    @pytest.mark.parametrize(
        "key", ["headers_x-end-user-id", "headers_X-End-User-Id", "headers_ x-end-user-id"]
    )
    def test_a_declared_flat_argument_is_dropped(self, called, caplog, key):
        with caplog.at_level(logging.WARNING, logger="stackone.tools"):
            self._tool({key: {"type": "string"}}).execute({key: "victim", "q": 1})

        assert called[0]["arguments"] == {"q": 1}
        assert f'Dropping header argument "{key}" from a tool call: set by the SDK' in caplog.text
