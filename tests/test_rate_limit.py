"""429 handling: every request the SDK makes retries a rate limit."""

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
    fetch_mcp_tools,
)
from stackone_ai.toolset import StackOneToolSet
from stackone_ai.types import StackOneAPIError

URL = "https://api.example.com/mcp"


@pytest.fixture
def sleeps(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """Record every retry delay, sync or async, instead of waiting it out."""
    recorded: list[float] = []

    async def fake_async_sleep(delay: float) -> None:
        recorded.append(delay)

    monkeypatch.setattr(tools_module, "_sleep", recorded.append)
    monkeypatch.setattr(tools_module, "_async_sleep", fake_async_sleep)
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
    def test_retry_after_in_seconds(self):
        assert _retry_after_seconds("5") == 5.0
        assert _retry_after_seconds(" 0 ") == 0.0

    def test_retry_after_as_an_http_date(self):
        when = datetime.now(UTC) + timedelta(seconds=10)
        assert 8.0 <= (_retry_after_seconds(format_datetime(when, usegmt=True)) or 0) <= 10.0

    def test_a_past_http_date_means_now(self):
        assert _retry_after_seconds("Wed, 21 Oct 2015 07:28:00 GMT") == 0.0

    @pytest.mark.parametrize("value", [None, "", "soon", "-1", "1.5e3"])
    def test_an_absent_or_unreadable_header_falls_back(self, value: str | None):
        assert _retry_after_seconds(value) is None

    @pytest.mark.parametrize(
        "retry_after", ["120", format_datetime(datetime.now(UTC) + timedelta(hours=1), usegmt=True)]
    )
    def test_retry_after_is_capped(self, retry_after: str):
        assert _rate_limit_delay(_429(retry_after), 1) == RATE_LIMIT_MAX_DELAY_SECONDS == 30.0

    def test_retry_after_zero_is_immediate(self):
        assert _rate_limit_delay(_429("0"), 3) == 0.0

    @pytest.mark.parametrize("retry", [1, 2, 3])
    def test_backoff_without_a_header_is_jittered_within_bounds(self, retry: int):
        base = RATE_LIMIT_BACKOFF_SECONDS[retry - 1]
        assert base == 2.0 ** (retry - 1)
        delays = [_rate_limit_delay(_429(), retry) for _ in range(200)]
        assert all(base * 0.5 <= delay <= base for delay in delays)
        assert len(set(delays)) > 1

    def test_backoff_jitter_extremes(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(tools_module.random, "uniform", lambda low, high: low)
        assert [_rate_limit_delay(_429(), r) for r in (1, 2, 3)] == [0.5, 1.0, 2.0]
        monkeypatch.setattr(tools_module.random, "uniform", lambda low, high: high)
        assert [_rate_limit_delay(_429(), r) for r in (1, 2, 3)] == [1.0, 2.0, 4.0]


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
            f"GET {URL} was rate limited (429) on attempt {n} of 4; retrying in 3.00s" for n in (1, 2, 3)
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


class TestMcpAgainstMockServer:
    """The mock answers 429 for `ratelimit-<all|call>-<n|always>-<tag>` account ids."""

    def test_listing_recovers_from_429s(self, mcp_mock_server: str, sleeps: list[float]):
        toolset = StackOneToolSet(api_key="test-key", base_url=mcp_mock_server)
        tools = toolset.fetch_tools(account_ids=["ratelimit-all-2-listing"])
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
