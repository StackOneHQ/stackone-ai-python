"""429 handling: every request the SDK makes retries a rate limit, and one that outlasts
its retries ends the whole call rather than being skipped like a per-account failure."""

from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime, timedelta
from email.utils import format_datetime
from typing import Any

import httpx
import pytest

from stackone_ai import tools as tools_module
from stackone_ai.tools import (
    RATE_LIMIT_BACKOFF_SECONDS,
    RATE_LIMIT_MAX_DELAY_SECONDS,
    RateLimitRetryingAsyncClient,
    RateLimitRetryingClient,
    _buffer_error_body,
    _rate_limit_delay,
    _retry_after_seconds,
    _Throttle,
    fetch_mcp_tools,
    is_rate_limited,
)
from stackone_ai.toolset import StackOneToolSet
from stackone_ai.types import StackOneAPIError, ToolsetLoadError

URL = "https://api.example.com/mcp"


@pytest.fixture
def sleeps(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """Record every retry delay, sync or async, instead of waiting it out.

    Each recorded delay still moves the retry clocks on, so a deadline sees the time a
    real wait would have taken.
    """
    recorded: list[float] = []
    real_clock, real_async_clock = tools_module._clock, tools_module._async_clock

    def fake_sleep(delay: float) -> None:
        recorded.append(delay)

    async def fake_async_sleep(delay: float) -> None:
        recorded.append(delay)

    monkeypatch.setattr(tools_module, "_sleep", fake_sleep)
    monkeypatch.setattr(tools_module, "_async_sleep", fake_async_sleep)
    monkeypatch.setattr(tools_module, "_clock", lambda: real_clock() + sum(recorded))
    monkeypatch.setattr(tools_module, "_async_clock", lambda: real_async_clock() + sum(recorded))
    return recorded


def _responder(*responses: httpx.Response) -> tuple[list[httpx.Request], Any]:
    """A handler that answers with ``responses`` in turn, repeating the last, and the requests it saw."""
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return responses[min(len(seen), len(responses)) - 1]

    return seen, handler


def _429(retry_after: str | None = None) -> httpx.Response:
    headers = {"Retry-After": retry_after} if retry_after is not None else {}
    return httpx.Response(429, headers=headers, json={"message": "Too many requests"})


def _recorded(base_url: str, account_id: str, method: str) -> list[dict[str, Any]]:
    response = httpx.get(f"{base_url}/__requests")
    response.raise_for_status()
    return [r for r in response.json() if r["accountId"] == account_id and r["method"] == method]


class TestRetryDelay:
    def test_retry_after_as_an_http_date(self):
        when = datetime.now(UTC) + timedelta(seconds=10)
        assert 8.0 <= (_retry_after_seconds(format_datetime(when, usegmt=True)) or 0) <= 10.0

    def test_retry_after_with_an_ordinary_leap_second(self):
        now = datetime(2015, 6, 30, 23, 59, 0, tzinfo=UTC)
        assert _retry_after_seconds("Tue, 30 Jun 2015 23:59:60 GMT", now) == 60.0

    def test_retry_after_with_a_leap_second_at_the_maximum_year_does_not_overflow(self):
        now = datetime(9999, 12, 31, 23, 59, 0, tzinfo=UTC)
        assert _retry_after_seconds("Fri, 31 Dec 9999 23:59:60 GMT", now) == 60.0

    @pytest.mark.parametrize(
        "retry_after", [format_datetime(datetime.now(UTC) + timedelta(hours=1), usegmt=True)]
    )
    def test_an_http_date_retry_after_is_capped(self, retry_after: str):
        assert _rate_limit_delay(_429(retry_after), 1) == RATE_LIMIT_MAX_DELAY_SECONDS == 30.0

    @pytest.mark.parametrize("retry", [1, 2, 3])
    def test_backoff_without_a_header_is_jittered_within_bounds(self, retry: int):
        base = RATE_LIMIT_BACKOFF_SECONDS[retry - 1]
        assert base == 2.0 ** (retry - 1)
        delays = [_rate_limit_delay(_429(), retry) for _ in range(200)]
        assert all(base * 0.5 <= delay <= base for delay in delays)
        assert len(set(delays)) > 1


class TestSyncClient:
    def test_a_429_then_success(self, sleeps: list[float]):
        seen, handler = _responder(_429("2"), httpx.Response(200, json={"ok": True}))
        with RateLimitRetryingClient(transport=httpx.MockTransport(handler)) as client:
            response = client.get(URL)
        assert response.json() == {"ok": True}
        assert len(seen) == 2
        assert sleeps == [2.0]

    def test_a_persistent_429_is_returned_after_four_attempts(self, sleeps: list[float]):
        seen, handler = _responder(_429())
        with RateLimitRetryingClient(transport=httpx.MockTransport(handler)) as client:
            response = client.get(URL)
        assert response.status_code == 429
        assert response.json() == {"message": "Too many requests"}
        assert len(seen) == 4
        assert len(sleeps) == 3

    @pytest.mark.parametrize("status", [400, 412, 500, 503])
    def test_no_retry_on_any_other_status(self, sleeps: list[float], status: int):
        seen, handler = _responder(httpx.Response(status, headers={"Retry-After": "1"}))
        with RateLimitRetryingClient(transport=httpx.MockTransport(handler)) as client:
            assert client.get(URL).status_code == status
        assert len(seen) == 1
        assert sleeps == []

    def test_each_retry_logs_one_warning(self, sleeps: list[float], caplog: pytest.LogCaptureFixture):
        _, handler = _responder(_429("3"))
        with caplog.at_level(logging.WARNING, logger="stackone.tools"):
            with RateLimitRetryingClient(transport=httpx.MockTransport(handler)) as client:
                client.get(URL)
        messages = [r.getMessage() for r in caplog.records]
        assert messages == [
            f"GET {URL} was rate limited (429) on attempt {n} of 4; retrying in 3s" for n in (1, 2, 3)
        ]


class TestAsyncClient:
    def _run(self, handler: Any, *, stream: bool) -> tuple[int, bytes]:
        async def go() -> tuple[int, bytes]:
            async with RateLimitRetryingAsyncClient(
                transport=httpx.MockTransport(handler),
                event_hooks={"response": [_buffer_error_body]},
            ) as client:
                if not stream:
                    response = await client.post(URL, json={"jsonrpc": "2.0"})
                    return response.status_code, response.content
                async with client.stream("POST", URL, json={"jsonrpc": "2.0"}) as response:
                    return response.status_code, await response.aread()

        return asyncio.run(go())

    @pytest.mark.parametrize("stream", [False, True])
    def test_a_429_then_success_resends_the_same_body(self, sleeps: list[float], stream: bool):
        seen, handler = _responder(_429("1"), _429(), httpx.Response(200, json={"ok": True}))
        assert self._run(handler, stream=stream) == (200, b'{"ok":true}')
        assert [r.content for r in seen] == [b'{"jsonrpc":"2.0"}'] * 3
        assert sleeps[0] == 1.0
        assert 1.0 <= sleeps[1] <= 2.0

    @pytest.mark.parametrize("stream", [False, True])
    def test_the_last_429_keeps_its_body(self, sleeps: list[float], stream: bool):
        seen, handler = _responder(_429("0"))
        assert self._run(handler, stream=stream) == (429, b'{"message":"Too many requests"}')
        assert len(seen) == 4
        assert sleeps == [0.0, 0.0, 0.0]

    @pytest.mark.parametrize("status", [400, 500])
    def test_no_retry_on_any_other_status(self, sleeps: list[float], status: int):
        seen, handler = _responder(httpx.Response(status))
        assert self._run(handler, stream=True)[0] == status
        assert len(seen) == 1
        assert sleeps == []


class TestFetchAccounts:
    @pytest.fixture
    def accounts_api(self, monkeypatch: pytest.MonkeyPatch) -> Any:
        def install(*responses: httpx.Response) -> list[httpx.Request]:
            seen, handler = _responder(*responses)
            monkeypatch.setattr(
                httpx.HTTPTransport, "handle_request", lambda _self, request: handler(request)
            )
            return seen

        return install

    def test_a_429_then_success(self, sleeps: list[float], accounts_api: Any):
        accounts = [{"id": "a", "provider": "p", "status": "active"}]
        seen = accounts_api(_429("4"), httpx.Response(200, json=accounts))
        assert StackOneToolSet(api_key="k").fetch_accounts() == accounts
        assert len(seen) == 2
        assert sleeps == [4.0]

    def test_a_persistent_429_raises_after_four_attempts(self, sleeps: list[float], accounts_api: Any):
        seen = accounts_api(_429())
        with pytest.raises(StackOneAPIError) as excinfo:
            StackOneToolSet(api_key="k").fetch_accounts()
        assert excinfo.value.status_code == 429
        assert "Too many requests" in excinfo.value.response_body
        assert len(seen) == 4
        assert len(sleeps) == 3

    def test_no_retry_on_500(self, sleeps: list[float], accounts_api: Any):
        seen = accounts_api(httpx.Response(500, text="boom"))
        with pytest.raises(StackOneAPIError) as excinfo:
            StackOneToolSet(api_key="k").fetch_accounts()
        assert excinfo.value.status_code == 500
        assert len(seen) == 1
        assert sleeps == []

    def test_a_wait_longer_than_the_timeout_is_not_started(self, sleeps: list[float], accounts_api: Any):
        """Retry-After 3 against timeout=1 raises the 429 at once, rather than after 9s."""
        seen = accounts_api(_429("3"))
        with pytest.raises(StackOneAPIError) as excinfo:
            StackOneToolSet(api_key="k", timeout=1).fetch_accounts()
        assert excinfo.value.status_code == 429
        assert len(seen) == 1
        assert sleeps == []

    def test_the_timeout_counts_from_the_first_attempt(self, sleeps: list[float], accounts_api: Any):
        # 2s fits in 5s; a second 2s would end at 4s and fits; a third would end at 6s.
        seen = accounts_api(_429("2"))
        with pytest.raises(StackOneAPIError) as excinfo:
            StackOneToolSet(api_key="k", timeout=5).fetch_accounts()
        assert excinfo.value.status_code == 429
        assert len(seen) == 3
        assert sleeps == [2.0, 2.0]


class TestATimeoutAfterARetried429IsThe429:
    """The retry's wait ends before the deadline, but the retried request can still run past it.

    As a timeout, a multi-account call skipped the account like any other failure and
    returned a partial catalog that looked complete.
    """

    def test_fetch_accounts(self, sleeps: list[float], monkeypatch: pytest.MonkeyPatch):
        seen: list[httpx.Request] = []

        def handler(_self: Any, request: httpx.Request) -> httpx.Response:
            seen.append(request)
            if len(seen) == 1:
                return _429("0")
            raise httpx.ReadTimeout("timed out", request=request)

        monkeypatch.setattr(httpx.HTTPTransport, "handle_request", handler)
        with pytest.raises(StackOneAPIError) as excinfo:
            StackOneToolSet(api_key="k").fetch_accounts()
        assert excinfo.value.status_code == 429
        assert str(excinfo.value) == (
            "Listing accounts at https://api.stackone.com/accounts was rate limited (429) and "
            "timed out after 60s while retrying"
        )
        assert len(seen) == 2

    @staticmethod
    def _429_then_hang(monkeypatch: pytest.MonkeyPatch, account: str) -> None:
        """The MCP host answers ``account`` 429 once, then never answers its retry."""
        real = httpx.AsyncHTTPTransport.handle_async_request
        attempts: dict[str, int] = {}

        async def handler(self: Any, request: httpx.Request) -> httpx.Response:
            if request.headers.get("x-account-id") != account:
                return await real(self, request)
            attempts[account] = attempts.get(account, 0) + 1
            if attempts[account] == 1:
                return _429("0")
            await asyncio.sleep(60)
            raise AssertionError("the exchange's deadline should have cancelled this")

        monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", handler)

    def test_listing(self, sleeps: list[float], monkeypatch: pytest.MonkeyPatch):
        self._429_then_hang(monkeypatch, "throttled")
        with pytest.raises(StackOneAPIError) as excinfo:
            fetch_mcp_tools(URL, {"x-account-id": "throttled"}, timeout=0.5)
        assert excinfo.value.status_code == 429
        assert str(excinfo.value) == (
            f"MCP request to {URL} was rate limited (429) and timed out after 0.5s while retrying"
        )
        assert is_rate_limited(excinfo.value)

    def test_a_timeout_with_no_429_is_still_a_timeout(self, monkeypatch: pytest.MonkeyPatch):
        async def hang(_self: Any, _request: httpx.Request) -> httpx.Response:
            await asyncio.sleep(60)
            raise AssertionError("unreachable")

        monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", hang)
        with pytest.raises(ToolsetLoadError, match="timed out after 0.5s"):
            fetch_mcp_tools(URL, {"x-account-id": "a"}, timeout=0.5)

    def test_a_multi_account_listing_is_aborted(
        self, mcp_mock_server: str, sleeps: list[float], monkeypatch: pytest.MonkeyPatch
    ):
        self._429_then_hang(monkeypatch, "throttled")
        toolset = StackOneToolSet(api_key="test-key", base_url=mcp_mock_server, timeout=1)
        with pytest.raises(StackOneAPIError) as excinfo:
            toolset.fetch_tools(account_ids=["acc1", "throttled"])
        assert excinfo.value.status_code == 429

    def test_a_tool_call(self, mcp_mock_server: str, sleeps: list[float], monkeypatch: pytest.MonkeyPatch):
        from stackone_ai.tools import StackOneMcpTool
        from stackone_ai.types import ToolParameters

        self._429_then_hang(monkeypatch, "throttled")
        tool = StackOneMcpTool(
            name="default_tool_1",
            description="",
            parameters=ToolParameters(type="object", properties={}),
            api_key="test-key",
            endpoint=f"{mcp_mock_server}/mcp",
            account_id="throttled",
            timeout=0.5,
        )
        with pytest.raises(StackOneAPIError) as excinfo:
            tool.execute({})
        assert excinfo.value.status_code == 429
        assert is_rate_limited(excinfo.value)


class TestAnOverlappingRequestDoesNotClearTheThrottle:
    """Another request answered while a 429's retry hangs leaves that 429 on record.

    The MCP client's requests overlap, its event-stream GET with a ``tools/list`` say: if any
    answer cleared the marker, the retry's timeout would be reported as an ordinary timeout.
    """

    def test_the_throttle_survives_another_requests_answer(self, sleeps: list[float]):
        throttle = _Throttle()
        list_attempts = 0
        retry_started = asyncio.Event()

        async def handler(request: httpx.Request) -> httpx.Response:
            nonlocal list_attempts
            if request.method == "GET":
                await retry_started.wait()
                return httpx.Response(405)
            list_attempts += 1
            if list_attempts == 1:
                return _429("0")
            retry_started.set()
            await asyncio.sleep(60)
            raise AssertionError("cancelled before this")

        async def go() -> httpx.Response | None:
            async with RateLimitRetryingAsyncClient(
                transport=httpx.MockTransport(handler), throttle=throttle
            ) as client:
                listing = asyncio.create_task(client.post(URL, json={"method": "tools/list"}))
                await client.get(URL)
                listing.cancel()
                return throttle.response

        assert asyncio.run(go()) is not None


class TestAStaleThrottleMarkerDoesNotTaintALaterTimeout:
    """A retried 429 that then succeeds must not mark a later, unrelated timeout as a 429.

    ``initialize`` and ``tools/list`` share one ``_Throttle`` for the whole MCP exchange.
    Without clearing the marker after the retry succeeds, a plain timeout on ``tools/list``
    was reported as the ``initialize`` 429 instead, and a multi-account call aborted the
    whole listing rather than skipping the one account.
    """

    @staticmethod
    def _429_once_on_initialize_then_hang_on_list(monkeypatch: pytest.MonkeyPatch, account: str) -> None:
        import json as _json

        real = httpx.AsyncHTTPTransport.handle_async_request
        attempts: dict[str, int] = {}

        async def handler(self: Any, request: httpx.Request) -> httpx.Response:
            if request.headers.get("x-account-id") != account:
                return await real(self, request)
            method = _json.loads(request.content or b"{}").get("method")
            if method == "initialize":
                attempts["initialize"] = attempts.get("initialize", 0) + 1
                if attempts["initialize"] == 1:
                    return _429("0")
                return await real(self, request)
            if method == "tools/list":
                await asyncio.sleep(60)
                raise AssertionError("the exchange's deadline should have cancelled this")
            return await real(self, request)

        monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", handler)

    def test_fetch_mcp_tools_times_out_rather_than_reporting_a_stale_429(
        self, mcp_mock_server: str, sleeps: list[float], monkeypatch: pytest.MonkeyPatch
    ):
        self._429_once_on_initialize_then_hang_on_list(monkeypatch, "acc1")
        with pytest.raises(ToolsetLoadError, match="timed out after 0.5s") as excinfo:
            fetch_mcp_tools(
                f"{mcp_mock_server}/mcp",
                {"x-account-id": "acc1", "Authorization": "Basic dGVzdC1rZXk6"},
                timeout=0.5,
            )
        assert not is_rate_limited(excinfo.value)

    def test_a_two_account_listing_skips_only_the_timed_out_account(
        self, mcp_mock_server: str, sleeps: list[float], monkeypatch: pytest.MonkeyPatch
    ):
        self._429_once_on_initialize_then_hang_on_list(monkeypatch, "acc2")
        toolset = StackOneToolSet(api_key="test-key", base_url=mcp_mock_server, timeout=0.5)
        tools = toolset.fetch_tools(account_ids=["acc1", "acc2"])
        assert tools.get_tool("acc1_tool_1") is not None
        assert tools.get_tool("acc2_tool_1") is None


class TestA429ThenAStalledStreamIsATimeout:
    """The retry of a 429 is answered 200, and then that response's stream stalls: the 429
    cleared, so the deadline that ends it is an ordinary timeout, as in Node."""

    @staticmethod
    def _429_then_stall_on_list(monkeypatch: pytest.MonkeyPatch, account: str) -> None:
        import json as _json

        real = httpx.AsyncHTTPTransport.handle_async_request
        attempts: list[int] = []

        class Stalled(httpx.AsyncByteStream):
            async def __aiter__(self) -> Any:
                yield b"event: message\n"
                await asyncio.sleep(60)
                raise AssertionError("the exchange's deadline should have cancelled this")

        async def handler(self: Any, request: httpx.Request) -> httpx.Response:
            if request.headers.get("x-account-id") != account:
                return await real(self, request)
            if _json.loads(request.content or b"{}").get("method") != "tools/list":
                return await real(self, request)
            attempts.append(1)
            if len(attempts) == 1:
                return _429("0")
            return httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=Stalled())

        monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", handler)

    def test_listing(self, mcp_mock_server: str, sleeps: list[float], monkeypatch: pytest.MonkeyPatch):
        self._429_then_stall_on_list(monkeypatch, "acc1")
        with pytest.raises(ToolsetLoadError, match="timed out after 0.5s") as excinfo:
            fetch_mcp_tools(
                f"{mcp_mock_server}/mcp",
                {"x-account-id": "acc1", "Authorization": "Basic dGVzdC1rZXk6"},
                timeout=0.5,
            )
        assert not is_rate_limited(excinfo.value)
        assert len(sleeps) == 1

    def test_a_two_account_listing_skips_only_that_account(
        self, mcp_mock_server: str, sleeps: list[float], monkeypatch: pytest.MonkeyPatch
    ):
        self._429_then_stall_on_list(monkeypatch, "acc2")
        toolset = StackOneToolSet(api_key="test-key", base_url=mcp_mock_server, timeout=0.5)
        tools = toolset.fetch_tools(account_ids=["acc1", "acc2"])
        assert tools.get_tool("acc1_tool_1") is not None
        assert tools.get_tool("acc2_tool_1") is None


class TestMcpAgainstMockServer:
    """The mock answers 429 for `ratelimit-<all|call>-<n|always>-<tag>` account ids."""

    @pytest.mark.parametrize("account", ["ratelimit-list-2-listing", "ratelimit-all-2-initialize"])
    def test_listing_recovers_from_429s(self, mcp_mock_server: str, sleeps: list[float], account: str):
        """On tools/list itself, and on the initialize that precedes it."""
        toolset = StackOneToolSet(api_key="test-key", base_url=mcp_mock_server)
        tools = toolset.fetch_tools(account_ids=[account])
        assert tools.get_tool("default_tool_1") is not None
        assert sleeps == [0.0, 0.0]

    def test_tools_call_recovers_from_a_429(self, mcp_mock_server: str, sleeps: list[float]):
        account = "ratelimit-call-1-call"
        toolset = StackOneToolSet(api_key="test-key", base_url=mcp_mock_server)
        tool = toolset.fetch_tools(account_ids=[account]).get_tool("default_tool_1")
        assert tool is not None

        result = tool.execute({"fields": "id"})

        assert result["isError"] is False
        assert len(_recorded(mcp_mock_server, account, "tools/call")) == 2
        assert sleeps == [0.0]

    def test_a_persistent_429_on_tools_call_raises_after_four_attempts(
        self, mcp_mock_server: str, sleeps: list[float]
    ):
        account = "ratelimit-call-always-call"
        toolset = StackOneToolSet(api_key="test-key", base_url=mcp_mock_server)
        tool = toolset.fetch_tools(account_ids=[account]).get_tool("default_tool_1")
        assert tool is not None

        with pytest.raises(StackOneAPIError) as excinfo:
            tool.execute({"fields": "id"})

        assert excinfo.value.status_code == 429
        assert "Too many requests" in excinfo.value.response_body
        assert len(_recorded(mcp_mock_server, account, "tools/call")) == 4
        assert sleeps == [0.0, 0.0, 0.0]

    def test_a_persistent_429_on_listing_raises(self, mcp_mock_server: str, sleeps: list[float]):
        headers = {"Authorization": tools_module.build_auth_header("test-key")}
        with pytest.raises(StackOneAPIError) as excinfo:
            fetch_mcp_tools(
                f"{mcp_mock_server}/mcp", {**headers, "x-account-id": "ratelimit-all-always-listing"}
            )
        assert excinfo.value.status_code == 429
        assert "Too many requests" in excinfo.value.response_body
        assert sleeps == [0.0, 0.0, 0.0]

    def test_a_rate_limited_account_aborts_a_multi_account_listing(
        self, mcp_mock_server: str, sleeps: list[float], caplog: pytest.LogCaptureFixture
    ):
        toolset = StackOneToolSet(api_key="test-key", base_url=mcp_mock_server)
        with caplog.at_level(logging.WARNING, logger="stackone.tools"):
            with pytest.raises(StackOneAPIError) as excinfo:
                toolset.fetch_tools(account_ids=["acc1", "legacy-account", "ratelimit-all-always-multi"])
        assert excinfo.value.status_code == 429
        assert not any("Skipping account" in r.getMessage() for r in caplog.records)
        # Nothing was cached: the next call lists again rather than serving a partial catalog.
        assert toolset._catalog_cache == {}

    def test_other_failures_are_still_skipped(self, mcp_mock_server: str, caplog: pytest.LogCaptureFixture):
        toolset = StackOneToolSet(api_key="test-key", base_url=mcp_mock_server)
        with caplog.at_level(logging.WARNING, logger="stackone.tools"):
            tools = toolset.fetch_tools(account_ids=["acc1", "legacy-account"])
        assert tools.get_tool("acc1_tool_1") is not None
        assert any("Skipping account" in r.getMessage() for r in caplog.records)

    def test_a_rate_limited_account_aborts_discovery(
        self, mcp_mock_server: str, sleeps: list[float], monkeypatch: pytest.MonkeyPatch
    ):
        discovered = [
            {"id": "acc1", "provider": "p", "status": "active"},
            {"id": "ratelimit-all-always-discovery", "provider": "p", "status": "active"},
        ]
        monkeypatch.setattr(StackOneToolSet, "fetch_accounts", lambda _self: discovered)
        toolset = StackOneToolSet(api_key="test-key", base_url=mcp_mock_server)
        with pytest.raises(StackOneAPIError) as excinfo:
            toolset.fetch_tools()
        assert excinfo.value.status_code == 429

    def test_a_wait_past_the_deadline_raises_the_429(self, mcp_mock_server: str, sleeps: list[float]):
        """Retry-After 2 against timeout=3: one wait fits, a second would not.

        Sleeping into the deadline turned the 429 into a timeout, which was skipped like
        any per-account failure, and the call returned acc1's tools alone.
        """
        account = "ratelimit-all-always-after2-deadline"
        toolset = StackOneToolSet(api_key="test-key", base_url=mcp_mock_server, timeout=3)
        with pytest.raises(StackOneAPIError) as excinfo:
            toolset.fetch_tools(account_ids=["acc1", account])
        assert excinfo.value.status_code == 429
        assert sleeps == [2.0]

    def test_a_wait_longer_than_the_timeout_is_not_started(self, mcp_mock_server: str, sleeps: list[float]):
        account = "ratelimit-call-always-after5-deadline"
        toolset = StackOneToolSet(api_key="test-key", base_url=mcp_mock_server, timeout=3)
        tool = toolset.fetch_tools(account_ids=[account]).get_tool("default_tool_1")
        assert tool is not None

        with pytest.raises(StackOneAPIError) as excinfo:
            tool.execute({"fields": "id"})

        assert excinfo.value.status_code == 429
        assert len(_recorded(mcp_mock_server, account, "tools/call")) == 1
        assert sleeps == []

    def test_a_rate_limited_connector_aborts_search(self, mcp_mock_server: str, sleeps: list[float]):
        # Listing succeeds for both; only the rate-limited account's search_actions call 429s.
        toolset = StackOneToolSet(api_key="test-key", base_url=mcp_mock_server)
        with pytest.raises(StackOneAPIError) as excinfo:
            toolset.search("anything", account_ids=["acc1", "ratelimit-call-always-search"])
        assert excinfo.value.status_code == 429


class TestOnlyAnHttp429IsFatal:
    """A 429 in a tool result's payload was never retried, so it does not end the call."""

    def test_an_http_429_inside_a_task_group_is_fatal(self):
        request = httpx.Request("POST", "https://api.example.com/mcp")
        http = httpx.HTTPStatusError("429", request=request, response=httpx.Response(429, request=request))
        error = StackOneAPIError("rate limited", 429, None)
        error.__cause__ = ExceptionGroup("unhandled errors in a TaskGroup", [http])
        assert is_rate_limited(error)

    def test_a_payload_429_is_not(self):
        assert not is_rate_limited(StackOneAPIError("Tool failed", 429, {"isError": True}))
