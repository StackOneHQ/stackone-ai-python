"""x-end-user-id: the end user a non-shared account belongs to.

The API refuses an MCP request for a non-shared account that does not carry the account's
end-user id. GET /accounts names it (``origin_username``), so whenever the SDK lists
accounts it records it, and sends it on every MCP request for that account.
"""

from __future__ import annotations

import concurrent.futures
import json
import logging
import threading
from typing import Any

import httpx
import pytest

from stackone_ai.tools import McpToolDefinition, StackOneMcpTool
from stackone_ai.toolset import StackOneToolSet
from stackone_ai.types import StackOneAPIError, ToolParameters, ToolsetLoadError

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
        tools = StackOneToolSet(api_key="test-key", base_url=base_url, include_non_shared=True).fetch_tools()

        assert {"acc1_tool_1", "acc2_tool_1"} <= {t.name for t in tools}
        assert {"initialize", "tools/list"} <= _methods(base_url, "acc1")
        assert _end_user_ids(base_url, "acc1") == {"end-user-1"}
        # Shared: no end user, so no header.
        assert _end_user_ids(base_url, "acc2") == {None}

    def test_a_listed_tool_sends_it_on_every_call_request(self, mcp_mock_server_with_end_users: str):
        base_url = mcp_mock_server_with_end_users
        tools = StackOneToolSet(api_key="test-key", base_url=base_url, include_non_shared=True).fetch_tools()
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
        toolset = StackOneToolSet(api_key="test-key", base_url=base_url, include_non_shared=True)
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

    def test_an_explicit_id_looks_its_end_user_up_when_the_api_asks(
        self, mcp_mock_server_with_end_users: str
    ):
        """No GET /accounts was made, so the first request goes without; the 400 prompts one."""
        base_url = mcp_mock_server_with_end_users
        toolset = StackOneToolSet(api_key="test-key", base_url=base_url)
        _reset(base_url)

        assert toolset.fetch_tools(account_ids=["acc2"]).get_tool("acc2_tool_1") is not None
        assert _end_user_ids(base_url, "acc2") == {None}
        assert toolset.fetch_tools(account_ids=["acc1"]).get_tool("acc1_tool_1") is not None
        # Refused once without it, then listed with it.
        assert _end_user_ids(base_url, "acc1") == {None, "end-user-1"}
        assert toolset._end_user_id("acc1") == "end-user-1"

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

        failures: list[BaseException] = []
        completed: list[str] = []

        def run(name: str) -> None:
            # Thread swallows exceptions, so a failure in either call is collected and asserted.
            try:
                toolset.fetch_accounts()
                completed.append(name)
            except BaseException as exc:  # noqa: BLE001
                failures.append(exc)

        first = threading.Thread(target=run, args=("first",))
        first.start()
        assert first_started.wait(timeout=5)

        second = threading.Thread(target=run, args=("second",))
        second.start()
        assert second_started.wait(timeout=5)

        # Started second, finishes first.
        release_second.set()
        second.join(timeout=5)
        # Started first, finishes last: its stale response must not overwrite the
        # newer-started call's record.
        release_first.set()
        first.join(timeout=5)

        assert not first.is_alive() and not second.is_alive()
        assert failures == []
        # Both calls returned, so the stale first one really did finish last.
        assert completed == ["second", "first"]
        assert toolset._end_user_id("b") == "user-b"
        assert toolset._end_user_id("a") is None


class TestHeaders:
    def test_discovery_sends_it_on_listing_and_execution(self, monkeypatch):
        _accounts_response(
            monkeypatch,
            [_account("a", shared=False, origin_username="user-a"), _account("b", shared=True)],
        )
        mcp = _Mcp(monkeypatch)
        tools = StackOneToolSet(api_key="k", include_non_shared=True).fetch_tools()
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


class TestRecordOrdering:
    def test_an_older_call_records_when_a_newer_one_fails(self, monkeypatch):
        """Ordered against the last write, not the last start: a newer call that fails must
        not leave the older call's result unrecorded."""
        toolset = StackOneToolSet(api_key="k")
        first_started = threading.Event()
        release_first = threading.Event()

        def handle(_self: Any, request: httpx.Request) -> httpx.Response:
            if not first_started.is_set():
                first_started.set()
                assert release_first.wait(timeout=5)
                body = [_account("a", shared=False, origin_username="user-a", provider="hris")]
                return httpx.Response(200, json=body, request=request)
            return httpx.Response(500, text="nope", request=request)

        monkeypatch.setattr(httpx.HTTPTransport, "handle_request", handle)
        first = threading.Thread(target=toolset.fetch_accounts)
        first.start()
        assert first_started.wait(timeout=5)

        with pytest.raises(StackOneAPIError):
            toolset.fetch_accounts()
        release_first.set()
        first.join(timeout=5)

        assert not first.is_alive()
        assert toolset._end_user_id("a") == "user-a"
        assert toolset._providers == {"a": "hris"}


class TestDiscoverySkipsNonSharedAccounts:
    SKIPPED = (
        "Discovery skipped 2 non-shared account(s) (a, c): each belongs to a single end user. "
        "Pass their account ids, or opt in to non-shared accounts, to use them."
    )

    @pytest.fixture
    def accounts(self, monkeypatch: pytest.MonkeyPatch) -> list[int]:
        return _accounts_response(
            monkeypatch,
            *[
                [
                    _account("c", shared=False, origin_username="user-c"),
                    _account("b", shared=True),
                    _account("a", shared=False, origin_username="user-a"),
                    _account("d"),
                    _account("e", shared=False, status="error"),
                ]
            ]
            * 3,
        )

    def test_by_default_with_one_warning_per_discovery(self, accounts, monkeypatch, caplog):
        mcp = _Mcp(monkeypatch)
        toolset = StackOneToolSet(api_key="k")
        with caplog.at_level(logging.WARNING, logger="stackone.tools"):
            assert sorted(t.name for t in toolset.fetch_tools()) == ["tool_b", "tool_d"]
            toolset.fetch_tools()
        assert [r.getMessage() for r in caplog.records] == [self.SKIPPED]
        assert sorted(h["x-account-id"] for h in mcp.listed) == ["b", "d"]
        # Their end users are recorded all the same, for when their ids are passed.
        assert (toolset._end_user_id("a"), toolset._end_user_id("c")) == ("user-a", "user-c")

        caplog.clear()
        toolset.clear_catalog_cache()
        with caplog.at_level(logging.WARNING, logger="stackone.tools"):
            toolset.fetch_tools()
        assert [r.getMessage() for r in caplog.records] == [self.SKIPPED]
        assert len(accounts) == 2

    def test_passed_ids_are_used(self, accounts, monkeypatch):
        mcp = _Mcp(monkeypatch)
        toolset = StackOneToolSet(api_key="k")
        toolset.fetch_tools()
        toolset.fetch_tools(account_ids=["a"])
        assert mcp.listed[-1] == {**mcp.listed[-1], "x-account-id": "a", "x-end-user-id": "user-a"}

    def test_an_opt_in_includes_them(self, accounts, monkeypatch, caplog):
        _Mcp(monkeypatch)
        with caplog.at_level(logging.WARNING, logger="stackone.tools"):
            tools = StackOneToolSet(api_key="k", include_non_shared=True).fetch_tools()
        assert sorted(t.name for t in tools) == ["tool_a", "tool_b", "tool_c", "tool_d"]
        assert caplog.records == []

    def test_skipping_every_account_leaves_nothing_to_list(self, monkeypatch, caplog):
        _accounts_response(monkeypatch, [_account("a", shared=False, origin_username="user-a")])
        mcp = _Mcp(monkeypatch)
        toolset = StackOneToolSet(api_key="k")
        with caplog.at_level(logging.WARNING, logger="stackone.tools"):
            assert len(toolset.fetch_tools()) == 0
            assert toolset.search("anything") == []
            with pytest.raises(ToolsetLoadError, match="did not serve stackone_submit_feedback"):
                toolset.submit_feedback("positive", ["x"])
        assert mcp.listed == []
        assert [r.getMessage() for r in caplog.records] == [
            "Discovery skipped 1 non-shared account(s) (a): each belongs to a single end user. "
            "Pass their account ids, or opt in to non-shared accounts, to use them."
        ]


def _refusal(account: str) -> StackOneAPIError:
    body = json.dumps({"statusCode": 400, "message": f"{UCA_REFUSAL[: -len('acc1')]}{account}"})
    return StackOneAPIError(f"MCP request failed with 400 Bad Request: {body}", 400, body)


class _GuardedMcp:
    """A fake MCP transport that, like the API, refuses a non-shared account without its end user."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch, end_users: dict[str, str]) -> None:
        self.end_users = end_users
        self.listed: list[dict[str, str]] = []
        self.called: list[dict[str, str]] = []
        monkeypatch.setattr("stackone_ai.toolset.fetch_mcp_tools", self._list)
        monkeypatch.setattr("stackone_ai.tools.call_mcp_tool", self._call)

    def _guard(self, headers: dict[str, str]) -> None:
        account = headers["x-account-id"]
        expected = self.end_users.get(account)
        if expected is not None and headers.get("x-end-user-id") != expected:
            raise _refusal(account)

    def _list(self, _endpoint: str, headers: dict[str, str], **_kwargs: Any) -> list[McpToolDefinition]:
        self.listed.append(headers)
        self._guard(headers)
        return [McpToolDefinition(name=f"tool_{headers['x-account-id']}", description="", input_schema={})]

    def _call(self, _endpoint: str, headers: dict[str, str], *_args: Any, **_kwargs: Any) -> dict[str, Any]:
        self.called.append(headers)
        self._guard(headers)
        return {"isError": False}


class TestAPassedIdLooksItsEndUserUp:
    """An account passed by id has no end user recorded until something lists accounts; the
    API's 400 asking for one prompts that listing, and one retry with what it recorded."""

    def test_a_listing_is_retried_once_with_it(self, monkeypatch):
        calls = _accounts_response(monkeypatch, [_account("a", shared=False, origin_username="user-a")])
        mcp = _GuardedMcp(monkeypatch, {"a": "user-a"})
        tools = StackOneToolSet(api_key="k", account_id="a").fetch_tools()
        assert [t.name for t in tools] == ["tool_a"]
        assert [h.get("x-end-user-id") for h in mcp.listed] == [None, "user-a"]
        assert len(calls) == 1

    def test_a_call_is_retried_once_with_it(self, monkeypatch):
        calls = _accounts_response(monkeypatch, [_account("a", shared=False, origin_username="user-a")])
        mcp = _GuardedMcp(monkeypatch, {})
        tool = StackOneToolSet(api_key="k", account_id="a").fetch_tools()[0]
        mcp.end_users["a"] = "user-a"

        tool.execute({})
        assert [h.get("x-end-user-id") for h in mcp.called] == [None, "user-a"]
        assert len(calls) == 1

    @pytest.mark.parametrize(
        "accounts_response",
        [
            pytest.param([_account("a", shared=True)], id="no-end-user-listed"),
            pytest.param(500, id="listing-fails"),
        ],
    )
    def test_with_no_end_user_to_retry_with_the_400_is_raised(self, monkeypatch, accounts_response):
        calls = _accounts_response(monkeypatch, accounts_response)
        mcp = _GuardedMcp(monkeypatch, {"a": "user-a"})
        with pytest.raises(StackOneAPIError) as excinfo:
            StackOneToolSet(api_key="k", account_id="a").fetch_tools()
        assert excinfo.value.status_code == 400
        assert excinfo.value.response_body == _refusal("a").response_body
        assert len(mcp.listed) == 1
        assert len(calls) == 1

    def test_a_request_sent_with_a_recorded_end_user_is_not_retried(self, monkeypatch):
        calls = _accounts_response(monkeypatch, [_account("a", shared=False, origin_username="stale")])
        mcp = _GuardedMcp(monkeypatch, {"a": "user-a"})
        toolset = StackOneToolSet(api_key="k", account_id="a")
        toolset.fetch_accounts()
        with pytest.raises(StackOneAPIError):
            toolset.fetch_tools()
        assert len(mcp.listed) == 1
        assert len(calls) == 1

    @pytest.mark.parametrize(
        "error",
        [
            pytest.param(_refusal("a1"), id="another-account"),
            pytest.param(
                StackOneAPIError("400", 400, json.dumps({"message": "bad request"})), id="another-400"
            ),
            pytest.param(StackOneAPIError("401", 401, _refusal("a").response_body), id="not-a-400"),
        ],
    )
    def test_any_other_failure_is_not_retried(self, monkeypatch, error):
        calls = _accounts_response(monkeypatch)
        listed: list[int] = []

        def refuse(*_args: Any, **_kwargs: Any) -> list[McpToolDefinition]:
            listed.append(1)
            raise error

        monkeypatch.setattr("stackone_ai.toolset.fetch_mcp_tools", refuse)
        with pytest.raises(StackOneAPIError) as excinfo:
            StackOneToolSet(api_key="k", account_id="a").fetch_tools()
        assert excinfo.value is error
        assert (calls, listed) == ([], [1])

    def test_a_rate_limited_lookup_stays_fatal_to_a_fan_out(self, monkeypatch):
        """Skipping the account would hand back a partial catalog, so the 429 is raised."""
        monkeypatch.setattr("stackone_ai.tools._sleep", lambda _delay: None)
        _accounts_response(monkeypatch, 429, 429, 429, 429)
        _GuardedMcp(monkeypatch, {"a": "user-a"})
        with pytest.raises(StackOneAPIError) as excinfo:
            StackOneToolSet(api_key="k").fetch_tools(account_ids=["a", "b"])
        assert excinfo.value.status_code == 429

    def test_a_refusal_quoting_the_account_id_is_still_looked_up(self, monkeypatch):
        calls = _accounts_response(monkeypatch, [_account("a", shared=False, origin_username="user-a")])

        def guarded(_endpoint: str, headers: dict[str, str], **_kwargs: Any) -> list[McpToolDefinition]:
            if headers.get("x-end-user-id") != "user-a":
                body = json.dumps({"statusCode": 400, "message": f'{UCA_REFUSAL[: -len("acc1")]}"a"'})
                raise StackOneAPIError(f"MCP request failed with 400 Bad Request: {body}", 400, body)
            return [McpToolDefinition(name="tool_a", description="", input_schema={})]

        monkeypatch.setattr("stackone_ai.toolset.fetch_mcp_tools", guarded)
        tools = StackOneToolSet(api_key="k", account_id="a").fetch_tools()
        assert [tool.name for tool in tools] == ["tool_a"]
        assert len(calls) == 1

    def test_requests_refused_together_share_one_accounts_listing(self, monkeypatch):
        joined = threading.Event()

        class Listing(concurrent.futures.Future):  # type: ignore[type-arg]
            def result(self, timeout: float | None = None) -> Any:
                # The owner waits only once its own GET /accounts is done, so a wait while it
                # is still held is the other account's, joining it.
                joined.set()
                return super().result(timeout)

        monkeypatch.setattr(concurrent.futures, "Future", Listing)
        calls: list[int] = []

        def handle(_self: Any, request: httpx.Request) -> httpx.Response:
            calls.append(1)
            assert joined.wait(timeout=5)
            body = [
                _account("a", shared=False, origin_username="user-a"),
                _account("b", shared=False, origin_username="user-b"),
            ]
            return httpx.Response(200, json=body, request=request)

        monkeypatch.setattr(httpx.HTTPTransport, "handle_request", handle)
        mcp = _GuardedMcp(monkeypatch, {"a": "user-a", "b": "user-b"})
        tools = StackOneToolSet(api_key="k").fetch_tools(account_ids=["a", "b"])
        assert sorted(t.name for t in tools) == ["tool_a", "tool_b"]
        assert len(calls) == 1
        assert sorted((h["x-account-id"], h.get("x-end-user-id", "")) for h in mcp.listed) == [
            ("a", ""),
            ("a", "user-a"),
            ("b", ""),
            ("b", "user-b"),
        ]

    def test_a_lookup_joins_the_latest_accounts_listing_still_in_flight(self, monkeypatch):
        """An older listing finishing must not stop a lookup joining a newer one in flight."""
        joined = threading.Event()

        class Listing(concurrent.futures.Future):  # type: ignore[type-arg]
            def result(self, timeout: float | None = None) -> Any:
                joined.set()
                return super().result(timeout)

        monkeypatch.setattr(concurrent.futures, "Future", Listing)
        calls: list[int] = []
        started = [threading.Event(), threading.Event()]
        release = [threading.Event(), threading.Event()]

        def handle(_self: Any, request: httpx.Request) -> httpx.Response:
            index = len(calls)
            calls.append(index)
            if index < 2:
                started[index].set()
                assert release[index].wait(timeout=5)
            body = [_account("a", shared=False, origin_username="user-a")]
            return httpx.Response(200, json=body, request=request)

        monkeypatch.setattr(httpx.HTTPTransport, "handle_request", handle)
        toolset = StackOneToolSet(api_key="k")
        older = threading.Thread(target=toolset.fetch_accounts)
        older.start()
        assert started[0].wait(timeout=5)
        newer = threading.Thread(target=toolset.fetch_accounts)
        newer.start()
        assert started[1].wait(timeout=5)
        release[0].set()
        older.join(timeout=5)

        found: list[bool] = []
        lookup = threading.Thread(target=lambda: found.append(toolset._look_up_end_user("a", _refusal("a"))))
        lookup.start()
        assert joined.wait(timeout=5)
        try:
            assert len(calls) == 2
        finally:
            release[1].set()
            lookup.join(timeout=5)
            newer.join(timeout=5)
        assert found == [True]
