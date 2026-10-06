"""sdk-conformance's shared unit-test vectors, run against this SDK's own code.

``tests/vectors/`` is a byte-identical copy of sdk-conformance's ``vectors/`` at the commit
CI pins; ``scripts/sync_vectors.sh`` refreshes it and CI fails if it drifts. The Node SDK
loads the same files, so both are checked against the same inputs and expected outputs.
Every case runs through the code path the SDK uses for real: a header case through
``StackOneMcpTool.execute``, an account-id case through the toolset, a message through
whatever raises or logs it. ``unresolved`` and ``known_differences`` are not graded.
"""

from __future__ import annotations

import json
import logging
import math
import re
from collections.abc import Callable, Iterator
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import pytest

from stackone_ai import tools as tools_module
from stackone_ai.tools import (
    McpToolDefinition,
    RateLimitRetryingClient,
    StackOneMcpTool,
    StackOneTool,
    Tools,
    _backoff_delay,
    _describe_mcp_failure,
    _outlasts_deadline,
    _raise_mcp_failure,
    _retry_after_seconds,
    _Throttle,
    parse_tool_result,
)
from stackone_ai.toolset import StackOneToolSet
from stackone_ai.types import (
    ExecuteConfig,
    StackOneAPIError,
    StackOneError,
    ToolArgumentsError,
    ToolParameters,
    ToolsetConfigError,
    ToolsetLoadError,
)

VECTORS = Path(__file__).parent / "vectors"

# Bumped by sdk-conformance when an existing case changes meaning. A bump fails here until
# someone has read what changed and updated the SDK to match.
VERSIONS = {
    "account-ids.json": 1,
    "argument-encoding.json": 1,
    "backoff.json": 1,
    "header-arguments.json": 1,
    "header-names.json": 1,
    "header-values.json": 1,
    "messages.json": 1,
    "retry-after.json": 1,
}

ENDPOINT = "https://api.example.com/mcp"
TOOL_NAME = "linear_acct_execute_action"


def _decode(value: Any) -> Any:
    """Replace every ``{"$number": "NaN" | "Infinity" | "-Infinity"}`` with its float."""
    if isinstance(value, dict):
        if value.keys() == {"$number"}:
            return float(value["$number"].replace("Infinity", "inf"))
        return {key: _decode(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_decode(item) for item in value]
    return value


def _load(name: str) -> dict[str, Any]:
    return json.loads((VECTORS / name).read_text(encoding="utf-8"))


def _cases(name: str, key: str = "cases") -> list[Any]:
    return [pytest.param(case, id=case["id"]) for case in _load(name)[key]]


def _seconds_close(actual: float | None, expected: float | None) -> bool:
    """Equal within the README's relative tolerance of 1e-9; ``None`` only equals ``None``."""
    if actual is None or expected is None:
        return actual is expected
    if math.isinf(expected):
        return actual == expected
    return math.isclose(actual, expected, rel_tol=1e-9, abs_tol=0.0)


# --- messages.json: rendering a template ------------------------------------------------

MESSAGES = _load("messages.json")
TEMPLATES = {message["id"]: message for message in MESSAGES["messages"]}


def _js_seconds(value: float) -> str:
    """A number as JavaScript's String(n) writes it, for the ordinary values the tests use."""
    return str(int(value)) if float(value).is_integer() else repr(float(value))


def _json_type(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int | float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    return "object"


FORMATS: dict[str, Callable[[Any], str]] = {
    "text": str,
    "json": lambda v: json.dumps(v, ensure_ascii=False, separators=(",", ":")),
    "integer": lambda v: str(int(v)),
    "seconds": _js_seconds,
    "seconds-2dp": lambda v: _js_seconds(math.floor(v * 100 + 0.5) / 100),
    "json-type": _json_type,
    "reason": lambda v: MESSAGES["reasons"][v],
}


def _render(message_id: str, **values: Any) -> str:
    """The canonical text of a message, its placeholders written in their formats."""
    message = TEMPLATES[message_id]
    placeholders = message["placeholders"]
    assert set(values) == set(placeholders), f"{message_id} takes {sorted(placeholders)}"
    return re.sub(
        r"\{(\w+)\}",
        lambda m: FORMATS[placeholders[m[1]]["format"]](values[m[1]]),
        message["template"],
    )


def _warnings(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [
        r.getMessage()
        for r in caplog.records
        if r.name.startswith("stackone") and r.levelno == logging.WARNING
    ]


# --- shared fixtures ---------------------------------------------------------------------


@pytest.fixture
def sent(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Replace the MCP transport and record the arguments a tools/call would have sent."""
    captured: dict[str, Any] = {}

    def fake_call(endpoint, headers, name, arguments, **_kwargs):
        captured["arguments"] = arguments
        return {"isError": False, "result": {}}

    monkeypatch.setattr(tools_module, "call_mcp_tool", fake_call)
    return captured


@pytest.fixture
def warnings_log(caplog: pytest.LogCaptureFixture) -> Iterator[pytest.LogCaptureFixture]:
    with caplog.at_level(logging.WARNING, logger="stackone"):
        yield caplog


def _served_tool(input_schema: dict[str, Any], name: str = TOOL_NAME) -> StackOneTool:
    """A tool built the way fetch_tools() builds one from a served inputSchema."""
    toolset = StackOneToolSet(api_key="test-key", account_id="acc-1")
    return toolset._create_tool(McpToolDefinition(name, "", input_schema), "acc-1", ENDPOINT)


OPEN_HEADERS = {"type": "object", "properties": {"headers": {"type": "object"}}}


def test_every_vector_file_is_graded():
    """A new vector file fails here until a test loads it."""
    assert sorted(p.name for p in VECTORS.glob("*.json")) == sorted(VERSIONS)


@pytest.mark.parametrize(("name", "version"), sorted(VERSIONS.items()))
def test_vector_file_versions(name: str, version: int):
    assert _load(name)["version"] == version, f"{name} changed meaning: read what changed, then bump"


# --- retry-after.json --------------------------------------------------------------------


@pytest.mark.parametrize("case", _cases("retry-after.json"))
def test_retry_after(case: dict[str, Any]):
    now = datetime.fromisoformat(case["now"].replace("Z", "+00:00"))
    expected = _decode(case["expected_seconds"])
    actual = _retry_after_seconds(case["header"], now)
    assert _seconds_close(actual, expected), f"{case['header']!r} read as {actual}, expected {expected}"


# --- backoff.json ------------------------------------------------------------------------

BACKOFF = _load("backoff.json")


def test_backoff_constants():
    constants = BACKOFF["constants"]
    assert tools_module.RATE_LIMIT_MAX_RETRIES == constants["max_retries"]
    assert list(tools_module.RATE_LIMIT_BACKOFF_SECONDS) == constants["backoff_seconds"]
    assert tools_module.RATE_LIMIT_JITTER == (constants["jitter"]["min"], constants["jitter"]["max"])
    assert tools_module.RATE_LIMIT_MAX_DELAY_SECONDS == constants["max_delay_seconds"]


@pytest.mark.parametrize("case", _cases("backoff.json"))
def test_backoff(case: dict[str, Any], monkeypatch: pytest.MonkeyPatch):
    # Through _rate_limit_delay, as the client calls it: the 429's Retry-After header, the
    # jitter drawn from the patched random source.
    monkeypatch.setattr(tools_module.random, "random", lambda: case["random"])
    retry_after = case["retry_after_seconds"]
    headers = {} if retry_after is None else {"Retry-After": str(retry_after)}
    actual = tools_module._rate_limit_delay(httpx.Response(429, headers=headers), case["retry"])
    assert _seconds_close(actual, case["expected_delay_seconds"])
    assert _seconds_close(
        _backoff_delay(case["retry"], retry_after, case["random"]), case["expected_delay_seconds"]
    )


@pytest.mark.parametrize("case", _cases("backoff.json", "deadline_cases"))
def test_backoff_deadline(case: dict[str, Any], warnings_log: pytest.LogCaptureFixture):
    request = httpx.Request("GET", ENDPOINT)
    gives_up = _outlasts_deadline(request, 1, case["delay_seconds"], case["remaining_seconds"])
    assert ("return-429" if gives_up else "wait") == case["expected"]
    expected_warnings = (
        [
            _render(
                "rate-limit-deadline",
                method="GET",
                url=ENDPOINT,
                attempt=1,
                max_attempts=BACKOFF["constants"]["max_attempts"],
                delay=case["delay_seconds"],
            )
        ]
        if gives_up
        else []
    )
    assert _warnings(warnings_log) == expected_warnings


# --- header-values.json ------------------------------------------------------------------


@pytest.mark.parametrize("case", _cases("header-values.json"))
def test_header_value(case: dict[str, Any], sent: dict[str, Any]):
    _served_tool(OPEN_HEADERS).execute({"headers": {"x-probe": _decode(case["value"])}})
    expected = case["expected"]
    forwarded = {} if expected == {"dropped": True} else {"x-probe": expected}
    assert sent["arguments"] == {"headers": forwarded}


# --- header-names.json -------------------------------------------------------------------


@pytest.mark.parametrize("case", _cases("header-names.json"))
def test_header_name(case: dict[str, Any], sent: dict[str, Any], warnings_log: pytest.LogCaptureFixture):
    _served_tool(OPEN_HEADERS).execute({"headers": {case["name"]: "v"}})
    expected = case["expected"]
    if "forwarded" in expected:
        assert sent["arguments"] == {"headers": {expected["forwarded"]: "v"}}
        assert _warnings(warnings_log) == []
    else:
        assert sent["arguments"] == {"headers": {}}
        header = tools_module._trim_header_name(case["name"])
        assert _warnings(warnings_log) == [
            _render("header-dropped", header=header, reason=expected["dropped"])
        ]


# --- header-arguments.json ---------------------------------------------------------------


def _render_header_warning(warning: dict[str, str]) -> str:
    if "header" in warning:
        return _render("header-dropped", header=warning["header"], reason=warning["reason"])
    return _render("header-argument-dropped", argument=warning["argument"], reason=warning["reason"])


@pytest.mark.parametrize("case", _cases("header-arguments.json"))
def test_header_arguments(case: dict[str, Any], sent: dict[str, Any], warnings_log: pytest.LogCaptureFixture):
    _served_tool(case["schema"]).execute(_decode(case["arguments"]))
    assert sent["arguments"] == _decode(case["expected_arguments"])
    assert _warnings(warnings_log) == [_render_header_warning(w) for w in case["expected_warnings"]]


# --- argument-encoding.json --------------------------------------------------------------


def _first_non_finite(value: Any) -> float | None:
    if isinstance(value, float) and not math.isfinite(value):
        return value
    children = value.values() if isinstance(value, dict) else value if isinstance(value, list) else []
    return next((found for child in children if (found := _first_non_finite(child)) is not None), None)


@pytest.mark.parametrize("case", _cases("argument-encoding.json"))
def test_argument_encoding(case: dict[str, Any], sent: dict[str, Any]):
    tool = _served_tool({"type": "object", "properties": {}})
    arguments = case["arguments_json"] if "arguments_json" in case else _decode(case["arguments"])
    expected = case["expected"]
    if expected == "ok":
        tool.execute(arguments)
        assert sent["arguments"] == (json.loads(arguments) if isinstance(arguments, str) else arguments)
        return

    assert expected["error"] == "ToolArgumentsError"
    with pytest.raises(ToolArgumentsError) as raised:
        tool.execute(arguments)
    message = str(raised.value)
    reason = expected["reason"]
    if reason == "not-finite":
        parsed = json.loads(arguments) if isinstance(arguments, str) else arguments
        value = {"nan": "NaN", "inf": "Infinity", "-inf": "-Infinity"}[repr(_first_non_finite(parsed))]
        assert message == _render("arguments-not-finite", tool=TOOL_NAME, value=value)
    elif reason == "not-an-object":
        assert message == _render("arguments-not-an-object", tool=TOOL_NAME)
    else:
        message_id = {"invalid-json": "arguments-invalid-json", "unencodable": "arguments-not-encodable"}[
            reason
        ]
        detail = str(raised.value.__cause__)
        assert message == _render(message_id, tool=TOOL_NAME, detail=detail)
        assert not message.endswith("is not a JSON number")


# --- account-ids.json --------------------------------------------------------------------

ACCOUNT_IDS = _load("account-ids.json")


def _account_id_messages(parameter: str, value: Any) -> set[str]:
    """Every canonical refusal of an account-id argument."""
    if parameter == "account_id":
        return {_render("empty-account-id", parameter=parameter)}
    rendered = {
        _render("account-ids-not-strings", parameter=parameter),
        _render("account-ids-empty-id", parameter=parameter),
    }
    if isinstance(value, str):
        rendered.add(_render("account-ids-not-a-list", parameter=parameter, value=value))
    return rendered


def _account_id_paths(parameter: str, value: Any) -> list[Callable[[], object]]:
    """Every way a caller hands the toolset this argument."""
    if parameter == "account_id":
        return [lambda: StackOneToolSet(api_key="test-key", account_id=value)]
    return [
        lambda: StackOneToolSet(api_key="test-key", execute={"account_ids": value}),
        lambda: StackOneToolSet(api_key="test-key").set_accounts(value),
        lambda: StackOneToolSet(api_key="test-key", account_id="acc-0")._resolve_account_ids(value),
    ]


@pytest.mark.parametrize("case", _cases("account-ids.json"))
def test_account_ids(case: dict[str, Any]):
    parameter, value, expected = case["parameter"], case["value"], case["expected"]
    for path in _account_id_paths(parameter, value):
        if expected == "ok":
            path()
            continue
        error_class = expected["error"]["python"]
        assert error_class == ToolsetConfigError.__name__ == ACCOUNT_IDS["error_classes"]["python"]
        with pytest.raises(ToolsetConfigError) as raised:
            path()
        assert str(raised.value) in _account_id_messages(parameter, value)


# --- messages.json: every message through the code that emits it ------------------------


def _listing(*names: str) -> Callable[..., list[McpToolDefinition]]:
    def fake_fetch(_endpoint: str, headers: dict[str, str], **_kwargs: object) -> list[McpToolDefinition]:
        account = headers.get("x-account-id", "")
        return [
            McpToolDefinition(name.format(account=account), "", {"type": "object", "properties": {}})
            for name in names
        ]

    return fake_fetch


def _failing_for(failing: set[str], error: Exception, *names: str) -> Callable[..., list[McpToolDefinition]]:
    listed = _listing(*names)

    def fake_fetch(endpoint: str, headers: dict[str, str], **kwargs: object) -> list[McpToolDefinition]:
        if headers.get("x-account-id") in failing:
            raise error
        return listed(endpoint, headers, **kwargs)

    return fake_fetch


def _accounts_answer(
    monkeypatch: pytest.MonkeyPatch, answer: Callable[[httpx.Request], httpx.Response]
) -> str:
    """Answer GET /accounts with ``answer``; the URL it is requested at."""
    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", lambda _self, request: answer(request))
    return "https://api.stackone.com/accounts"


def _raised(error_class: type[BaseException], call: Callable[[], object]) -> BaseException:
    with pytest.raises(error_class) as raised:
        call()
    return raised.value


# The warning or tool-result text, or the exception raised; and the placeholder values.
Emitted = tuple[str | BaseException, dict[str, Any]]


def _header_dropped(mp, log) -> Emitted:
    mp.setattr(tools_module, "call_mcp_tool", lambda *_args, **_kwargs: {})
    _served_tool(OPEN_HEADERS).execute({"headers": {"Authorization": "Bearer x"}})
    return _warnings(log)[0], {"header": "Authorization", "reason": "set-by-sdk"}


def _header_argument_dropped(mp, log) -> Emitted:
    mp.setattr(tools_module, "call_mcp_tool", lambda *_args, **_kwargs: {})
    _served_tool({"type": "object", "properties": {}}).execute({"headers_x-foo": "1"})
    return _warnings(log)[0], {"argument": "headers_x-foo", "reason": "not-declared"}


def _rate_limit_retry(mp, log) -> Emitted:
    mp.setattr(tools_module.random, "random", lambda: 0.5)
    mp.setattr(tools_module, "_sleep", lambda _delay: None)
    answers = iter([httpx.Response(429), httpx.Response(200)])
    with RateLimitRetryingClient(transport=httpx.MockTransport(lambda _r: next(answers))) as client:
        client.get(ENDPOINT)
    values = {"method": "GET", "url": ENDPOINT, "attempt": 1, "max_attempts": 4, "delay": 0.75}
    return _warnings(log)[0], values


def _rate_limit_deadline(mp, log) -> Emitted:
    answer = httpx.Response(429, headers={"Retry-After": "5"})
    with RateLimitRetryingClient(transport=httpx.MockTransport(lambda _r: answer), retry_within=1) as client:
        client.get(ENDPOINT)
    values = {"method": "GET", "url": ENDPOINT, "attempt": 1, "max_attempts": 4, "delay": 5}
    return _warnings(log)[0], values


def _skip_account(mp, log) -> Emitted:
    mp.setattr(
        "stackone_ai.toolset.fetch_mcp_tools",
        _failing_for({"acc-2"}, ToolsetLoadError("boom"), "t_{account}"),
    )
    StackOneToolSet(api_key="k").fetch_tools(account_ids=["acc-1", "acc-2"])
    return _warnings(log)[0], {"account_id": "acc-2", "error": "boom"}


def _skip_connector(mp, log) -> Emitted:
    mp.setattr("stackone_ai.toolset.fetch_mcp_tools", _listing("c_{account}_search_actions"))

    def search(self: StackOneTool, _arguments: Any = None) -> dict[str, Any]:
        if self.name == "c_acc-2_search_actions":
            raise StackOneAPIError("boom", 500, None)
        return {"actions": [{"action_id": "c_a"}]}

    mp.setattr(StackOneMcpTool, "execute", search)
    StackOneToolSet(api_key="k").search("q", account_ids=["acc-1", "acc-2"])
    return _warnings(log)[0], {"tool": "c_acc-2_search_actions", "error": "boom"}


def _duplicate_tool_names(mp, log) -> Emitted:
    tools = Tools([_served_tool({"type": "object", "properties": {}}, "t") for _ in range(2)])
    # Once by the collection, and again by each adapter that keeps the first of each name.
    tools.to_openai()
    collection, adapter = _warnings(log)
    assert collection == adapter
    return adapter, {"count": 1, "names": "t"}


def _ambiguous_connector(mp, log) -> Emitted:
    mp.setattr("stackone_ai.toolset.fetch_mcp_tools", _listing("linear_{account}_execute_action"))
    mp.setattr(StackOneMcpTool, "execute", lambda _self, _arguments=None: {})
    error = _raised(
        ToolsetConfigError,
        lambda: StackOneToolSet(api_key="k").execute("linear_list_issues", account_ids=["acc-2", "acc-1"]),
    )
    tools = "linear_acc-1_execute_action on acc-1, linear_acc-2_execute_action on acc-2"
    return error, {"action_id": "linear_list_issues", "count": 2, "tools": tools}


def _account_id_env_ignored(mp, log) -> Emitted:
    mp.setenv("STACKONE_ACCOUNT_ID", "acc-1")
    StackOneToolSet(api_key="k")
    [warning] = _warnings(log)
    return warning, {}


def _connector_account_unavailable(mp, log) -> Emitted:
    mp.setattr(
        "stackone_ai.toolset.fetch_mcp_tools",
        _failing_for({"acc-2"}, ToolsetLoadError("boom"), "linear_{account}_execute_action"),
    )
    # Explicit ids: no GET /accounts, so acc-2's provider is unknown and it blocks.
    _accounts_answer(mp, lambda _r: httpx.Response(200, json=[{"id": "acc-1", "provider": "linear"}]))
    error = _raised(
        ToolsetLoadError,
        lambda: StackOneToolSet(api_key="k").execute("linear_list_issues", account_ids=["acc-1", "acc-2"]),
    )
    return error, {"action_id": "linear_list_issues", "failures": "acc-2: boom"}


def _non_shared_accounts_skipped(mp, log) -> Emitted:
    accounts = [
        {"id": "acc-1", "provider": "linear", "status": "active", "shared": True},
        {"id": "acc-3", "provider": "linear", "status": "active", "shared": False, "origin_username": "u3"},
        {"id": "acc-2", "provider": "linear", "status": "active", "shared": False, "origin_username": "u2"},
    ]
    mp.setattr(StackOneToolSet, "fetch_accounts", lambda _self: accounts)
    mp.setattr("stackone_ai.toolset.fetch_mcp_tools", _listing("linear_{account}_execute_action"))
    StackOneToolSet(api_key="k").fetch_tools()
    [warning] = _warnings(log)
    return warning, {"count": 2, "accounts": "acc-2, acc-3"}


def _mcp_timeout(mp, log) -> Emitted:
    return _describe_mcp_failure(TimeoutError(), ENDPOINT, 0.5), {"endpoint": ENDPOINT, "timeout": 0.5}


def _mcp_http_failure(mp, log) -> Emitted:
    response = httpx.Response(503, text=" down ", request=httpx.Request("POST", ENDPOINT))
    failure = httpx.HTTPStatusError("503", request=response.request, response=response)
    values = {"endpoint": ENDPOINT, "status": 503, "reason_phrase": "Service Unavailable", "body": "down"}
    return _describe_mcp_failure(failure, ENDPOINT, 60), values


def _mcp_failure(mp, log) -> Emitted:
    failure = ConnectionRefusedError("refused")
    values = {"endpoint": ENDPOINT, "error_type": "ConnectionRefusedError", "error_message": "refused"}
    return _describe_mcp_failure(failure, ENDPOINT, 60), values


def _mcp_rate_limit_timeout(mp, log) -> Emitted:
    throttled = httpx.Response(429, request=httpx.Request("POST", ENDPOINT))
    throttle = _Throttle(response=throttled)
    error = _raised(StackOneAPIError, lambda: _raise_mcp_failure(TimeoutError(), ENDPOINT, 0.5, throttle))
    assert error.status_code == 429
    return error, {"endpoint": ENDPOINT, "timeout": 0.5}


def _accounts_http_failure(mp, log) -> Emitted:
    url = _accounts_answer(mp, lambda _r: httpx.Response(401, text="bad key\n"))
    error = _raised(StackOneAPIError, StackOneToolSet(api_key="k").fetch_accounts)
    return error, {"url": url, "status": 401, "reason_phrase": "Unauthorized", "body": "bad key"}


def _accounts_rate_limit_timeout(mp, log) -> Emitted:
    seen: list[httpx.Request] = []

    def answer(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if len(seen) == 1:
            return httpx.Response(429, headers={"Retry-After": "0"})
        raise httpx.ReadTimeout("timed out", request=request)

    url = _accounts_answer(mp, answer)
    error = _raised(StackOneAPIError, StackOneToolSet(api_key="k").fetch_accounts)
    assert error.status_code == 429
    return error, {"url": url, "timeout": 60.0}


def _accounts_unreachable(mp, log) -> Emitted:
    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    url = _accounts_answer(mp, refuse)
    return _raised(ToolsetLoadError, StackOneToolSet(api_key="k").fetch_accounts), {
        "url": url,
        "detail": "refused",
    }


def _accounts_invalid_json(mp, log) -> Emitted:
    url = _accounts_answer(mp, lambda _r: httpx.Response(200, content=b"{"))
    error = _raised(ToolsetLoadError, StackOneToolSet(api_key="k").fetch_accounts)
    return error, {"url": url, "detail": str(error.__cause__)}


def _accounts_unexpected_shape(mp, log) -> Emitted:
    _accounts_answer(mp, lambda _r: httpx.Response(200, json="accounts"))
    return _raised(ToolsetLoadError, StackOneToolSet(api_key="k").fetch_accounts), {"json_type": "accounts"}


def _no_linked_accounts(mp, log) -> Emitted:
    _accounts_answer(mp, lambda _r: httpx.Response(200, json=[]))
    return _raised(ToolsetConfigError, StackOneToolSet(api_key="k").fetch_tools), {}


def _no_active_accounts(mp, log) -> Emitted:
    accounts = [
        {"id": "a", "provider": "linear", "status": "inactive"},
        {"id": "b", "provider": "jira", "status": "error"},
    ]
    _accounts_answer(mp, lambda _r: httpx.Response(200, json=accounts))
    error = _raised(ToolsetConfigError, StackOneToolSet(api_key="k").fetch_tools)
    return error, {"count": 2, "accounts": "linear (inactive), jira (error)"}


def _no_shared_accounts(mp, log) -> Emitted:
    accounts = [
        {"id": "a", "provider": "linear", "status": "active", "shared": False, "origin_username": "u1"},
        {"id": "b", "provider": "jira", "status": "active", "shared": False, "origin_username": "u2"},
        {"id": "c", "provider": "jira", "status": "error", "shared": True},
    ]
    _accounts_answer(mp, lambda _r: httpx.Response(200, json=accounts))
    error = _raised(ToolsetConfigError, StackOneToolSet(api_key="k").fetch_tools)
    return error, {"count": 2}


def _all_accounts_failed(mp, log) -> Emitted:
    failures = {"acc-1": StackOneAPIError("gone", 412, None), "acc-2": ToolsetLoadError("boom")}

    def fail(_endpoint: str, headers: dict[str, str], **_kwargs: object) -> list[McpToolDefinition]:
        raise failures[headers["x-account-id"]]

    mp.setattr("stackone_ai.toolset.fetch_mcp_tools", fail)
    error = _raised(
        ToolsetLoadError, lambda: StackOneToolSet(api_key="k").fetch_tools(account_ids=["acc-2", "acc-1"])
    )
    return error, {"failures": "acc-1: gone; acc-2: boom"}


def _no_connector_returned_results(mp, log) -> Emitted:
    mp.setattr("stackone_ai.toolset.fetch_mcp_tools", _listing("c_{account}_search_actions"))

    def fail(self: StackOneTool, _arguments: Any = None) -> dict[str, Any]:
        raise StackOneAPIError("boom", 500, None)

    mp.setattr(StackOneMcpTool, "execute", fail)
    error = _raised(
        ToolsetLoadError, lambda: StackOneToolSet(api_key="k").search("q", account_ids=["acc-1", "acc-2"])
    )
    return error, {"failures": "c_acc-1_search_actions: boom | c_acc-2_search_actions: boom"}


def _fetch_tools_failed(mp, log) -> Emitted:
    def explode(*_args: object, **_kwargs: object) -> list[McpToolDefinition]:
        raise RuntimeError("kaput")

    mp.setattr("stackone_ai.toolset.fetch_mcp_tools", explode)
    error = _raised(ToolsetLoadError, StackOneToolSet(api_key="k", account_id="acc-1").fetch_tools)
    return error, {"detail": "kaput"}


def _missing_api_key(mp, log) -> Emitted:
    mp.delenv("STACKONE_API_KEY", raising=False)
    return _raised(ToolsetConfigError, StackOneToolSet), {}


def _empty_account_id(mp, log) -> Emitted:
    error = _raised(ToolsetConfigError, lambda: StackOneToolSet(api_key="k", account_id=""))
    return error, {"parameter": "account_id"}


def _account_ids_not_a_list(mp, log) -> Emitted:
    error = _raised(ToolsetConfigError, lambda: StackOneToolSet(api_key="k").set_accounts("acc-1"))
    return error, {"parameter": "account_ids", "value": "acc-1"}


def _account_ids_not_strings(mp, log) -> Emitted:
    error = _raised(ToolsetConfigError, lambda: StackOneToolSet(api_key="k").set_accounts([1]))
    return error, {"parameter": "account_ids"}


def _account_ids_empty_id(mp, log) -> Emitted:
    error = _raised(ToolsetConfigError, lambda: StackOneToolSet(api_key="k").set_accounts(["acc-1", ""]))
    return error, {"parameter": "account_ids"}


def _invalid_session_id(mp, log) -> Emitted:
    error = _raised(
        ToolsetConfigError, lambda: StackOneToolSet(api_key="k").execute("linear_x", session_id="")
    )
    return error, {"parameter": "session_id", "value": ""}


def _invalid_action_id(mp, log) -> Emitted:
    return _raised(ToolsetConfigError, lambda: StackOneToolSet(api_key="k").execute("")), {
        "parameter": "action_id",
        "value": "",
    }


def _execute_arguments_not_an_object(mp, log) -> Emitted:
    error = _raised(ToolsetConfigError, lambda: StackOneToolSet(api_key="k").execute("linear_x", [1]))
    return error, {"json_type": [1]}


def _top_k_out_of_range(mp, log) -> Emitted:
    error = _raised(ToolsetConfigError, lambda: StackOneToolSet(api_key="k").search("q", top_k=0))
    return error, {"parameter": "top_k", "max": 50, "value": 0}


def _tool_names_not_a_list(mp, log) -> Emitted:
    error = _raised(ToolsetConfigError, lambda: StackOneToolSet(api_key="k").submit_feedback("positive", "t"))
    return error, {"parameter": "tool_names", "value": "t"}


def _feedback_not_enabled(mp, log) -> Emitted:
    mp.setattr("stackone_ai.toolset.fetch_mcp_tools", _listing("c_{account}_search_actions"))
    toolset = StackOneToolSet(api_key="k", account_id="acc-1")
    return _raised(ToolsetLoadError, lambda: toolset.submit_feedback("positive", ["t"])), {}


def _no_connector_for_action(mp, log) -> Emitted:
    mp.setattr("stackone_ai.toolset.fetch_mcp_tools", _listing("jira_{account}_execute_action"))
    toolset = StackOneToolSet(api_key="k", account_id="acc-1")
    return _raised(ToolsetLoadError, lambda: toolset.execute("linear_list_issues")), {
        "action_id": "linear_list_issues"
    }


def _arguments_not_finite(mp, log) -> Emitted:
    tool = _served_tool({"type": "object", "properties": {}})
    error = _raised(ToolArgumentsError, lambda: tool.execute({"a": [1, float("-inf")]}))
    return error, {"tool": TOOL_NAME, "value": "-Infinity"}


def _arguments_not_encodable(mp, log) -> Emitted:
    tool = _served_tool({"type": "object", "properties": {}})
    error = _raised(ToolArgumentsError, lambda: tool.execute({"a": {1, 2}}))
    return error, {"tool": TOOL_NAME, "detail": str(error.__cause__)}


def _arguments_invalid_json(mp, log) -> Emitted:
    tool = _served_tool({"type": "object", "properties": {}})
    error = _raised(ToolArgumentsError, lambda: tool.execute("{"))
    return error, {"tool": TOOL_NAME, "detail": str(error.__cause__)}


def _arguments_not_an_object(mp, log) -> Emitted:
    tool = _served_tool({"type": "object", "properties": {}})
    return _raised(ToolArgumentsError, lambda: tool.execute("[1]")), {"tool": TOOL_NAME}


def _tool_call_failed(mp, log) -> Emitted:
    result = SimpleNamespace(content=[SimpleNamespace(text='{"error":"bad"}')], isError=True)
    error = _raised(StackOneAPIError, lambda: parse_tool_result(result, "t"))
    return error, {"tool": "t", "detail": '{"error":"bad"}'}


def _no_executor(mp, log) -> Emitted:
    tool = StackOneTool(
        description="",
        parameters=ToolParameters(type="object", properties={}),
        _execute_config=ExecuteConfig(headers={}, name="t"),
    )
    return _raised(StackOneError, lambda: tool.execute({})), {"tool": "t"}


def _toolset_headers_ignored(mp, log) -> Emitted:
    StackOneToolSet(api_key="k", headers={"Authorization": "x", " X-Account-Id ": "y", "x-trace": "z"})
    [warning] = _warnings(log)
    return warning, {"names": '"Authorization", " X-Account-Id "'}


def _unknown_tool(mp, log) -> Emitted:
    [message] = Tools([]).execute_openai_tool_calls(
        [{"id": "1", "function": {"name": "nope", "arguments": "{}"}}]
    )
    return json.loads(message["content"])["error"], {"name": "nope"}


EMITTERS: dict[str, Callable[[pytest.MonkeyPatch, pytest.LogCaptureFixture], Emitted]] = {
    "header-dropped": _header_dropped,
    "header-argument-dropped": _header_argument_dropped,
    "rate-limit-retry": _rate_limit_retry,
    "rate-limit-deadline": _rate_limit_deadline,
    "skip-account": _skip_account,
    "skip-connector": _skip_connector,
    "duplicate-tool-names": _duplicate_tool_names,
    "ambiguous-connector": _ambiguous_connector,
    "account-id-env-ignored": _account_id_env_ignored,
    "connector-account-unavailable": _connector_account_unavailable,
    "non-shared-accounts-skipped": _non_shared_accounts_skipped,
    "mcp-timeout": _mcp_timeout,
    "mcp-http-failure": _mcp_http_failure,
    "mcp-failure": _mcp_failure,
    "mcp-rate-limit-timeout": _mcp_rate_limit_timeout,
    "accounts-http-failure": _accounts_http_failure,
    "accounts-rate-limit-timeout": _accounts_rate_limit_timeout,
    "accounts-unreachable": _accounts_unreachable,
    "accounts-invalid-json": _accounts_invalid_json,
    "accounts-unexpected-shape": _accounts_unexpected_shape,
    "no-linked-accounts": _no_linked_accounts,
    "no-active-accounts": _no_active_accounts,
    "no-shared-accounts": _no_shared_accounts,
    "all-accounts-failed": _all_accounts_failed,
    "no-connector-returned-results": _no_connector_returned_results,
    "fetch-tools-failed": _fetch_tools_failed,
    "missing-api-key": _missing_api_key,
    "empty-account-id": _empty_account_id,
    "account-ids-not-a-list": _account_ids_not_a_list,
    "account-ids-not-strings": _account_ids_not_strings,
    "account-ids-empty-id": _account_ids_empty_id,
    "invalid-session-id": _invalid_session_id,
    "invalid-action-id": _invalid_action_id,
    "execute-arguments-not-an-object": _execute_arguments_not_an_object,
    "top-k-out-of-range": _top_k_out_of_range,
    "tool-names-not-a-list": _tool_names_not_a_list,
    "feedback-not-enabled": _feedback_not_enabled,
    "no-connector-for-action": _no_connector_for_action,
    "arguments-not-finite": _arguments_not_finite,
    "arguments-not-encodable": _arguments_not_encodable,
    "arguments-invalid-json": _arguments_invalid_json,
    "arguments-not-an-object": _arguments_not_an_object,
    "tool-call-failed": _tool_call_failed,
    "no-executor": _no_executor,
    "unknown-tool": _unknown_tool,
    "toolset-headers-ignored": _toolset_headers_ignored,
}


def test_every_message_has_an_emitter():
    assert sorted(EMITTERS) == sorted(TEMPLATES)


@pytest.mark.parametrize("message", [pytest.param(m, id=m["id"]) for m in MESSAGES["messages"]])
def test_message(
    message: dict[str, Any], monkeypatch: pytest.MonkeyPatch, warnings_log: pytest.LogCaptureFixture
):
    emitted, values = EMITTERS[message["id"]](monkeypatch, warnings_log)
    if message["kind"] == "error":
        assert isinstance(emitted, BaseException)
        assert message["error"]["python"] in {cls.__name__ for cls in type(emitted).__mro__}
    assert str(emitted) == _render(message["id"], **values)
