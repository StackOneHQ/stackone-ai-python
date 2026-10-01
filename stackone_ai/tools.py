"""Tools served by the StackOne MCP endpoint, and their execution over MCP ``tools/call``.

The guiding property of this module is that a tool is the served catalog entry:
the schema listed to a model is the schema the MCP server sent, and a call sends the
model's arguments back to the endpoint that listed it, as given. Nothing is invented,
nothing is dropped, nothing is rebuilt.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import math
import random
import re
import threading
import time
from collections import Counter
from collections.abc import AsyncIterator, Coroutine, Iterable, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from email.utils import parsedate_to_datetime
from importlib import metadata
from typing import Any, TypeVar

import anyio
import httpx
from pydantic import BaseModel, Field, PrivateAttr

from stackone_ai.types import (
    ExecuteConfig,
    Headers,
    JsonDict,
    StackOneAPIError,
    StackOneError,
    ToolParameters,
    ToolsetConfigError,
    ToolsetError,
    ToolsetLoadError,
)

logger = logging.getLogger("stackone.tools")

T = TypeVar("T")

try:
    _SDK_VERSION = metadata.version("stackone-ai")
except metadata.PackageNotFoundError:  # pragma: no cover - best-effort fallback when running from source
    _SDK_VERSION = "dev"

USER_AGENT = f"stackone-ai-python/{_SDK_VERSION}"

_HEADER_NAME_PATTERN = re.compile(r"[A-Za-z0-9!#$%&'*+.^_`|~-]+")
_HEADER_VALUE_PATTERN = re.compile(r"[\x20-\x7e\t\x80-\xff]*")

# Header names the SDK sets itself, after every other header. A tool call may not supply
# them even when a served schema declares them: they are the credential, the tenant
# selector and the client identity.
_SDK_OWNED_HEADERS = frozenset({"authorization", "x-account-id", "user-agent"})

# A top-level argument with this prefix is a header argument, as is an entry of a nested
# `headers` object argument.
_FLAT_HEADER_PREFIX = "headers_"

# How a 429 is retried, on every request the SDK makes: up to three more attempts, each
# after the server's Retry-After (capped) or else an exponential backoff with jitter. A
# wait that would not end before the request's deadline is not started: the 429 is
# returned at once instead.
RATE_LIMIT_MAX_RETRIES = 3
RATE_LIMIT_BACKOFF_SECONDS = (1.0, 2.0, 4.0)
RATE_LIMIT_JITTER = (0.5, 1.0)
RATE_LIMIT_MAX_DELAY_SECONDS = 30.0

# Looked up at call time, so tests can replace them and not actually wait.
_sleep = time.sleep
_async_sleep = anyio.sleep
_clock = time.monotonic
_async_clock = anyio.current_time


@dataclass
class McpToolDefinition:
    """A tool exactly as the MCP server listed it."""

    name: str
    description: str | None
    input_schema: dict[str, Any]


def run_async(awaitable: Coroutine[Any, Any, T]) -> T:
    """Run a coroutine, even when called from an existing event loop."""

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        in_a_loop = False
    else:
        in_a_loop = True
    # Run outside the except block, or every failure would carry the probe's
    # "no running event loop" as its __context__.
    if not in_a_loop:
        return asyncio.run(awaitable)

    result: dict[str, T] = {}
    error: dict[str, BaseException] = {}

    def runner() -> None:
        try:
            result["value"] = asyncio.run(awaitable)
        except BaseException as exc:  # pragma: no cover - surfaced in caller context
            error["error"] = exc

    thread = threading.Thread(target=runner, daemon=True)
    thread.start()
    thread.join()

    if "error" in error:
        raise error["error"]

    return result["value"]


def _js_number(value: float) -> str:
    """A float as JavaScript's ``String(number)`` writes it: ``1.0`` is "1", ``1e-7`` is "1e-7".

    Both languages print the shortest digits that round-trip, so only the layout differs:
    the cut-offs for exponent notation, and the exponent's sign and padding.
    """
    if math.isnan(value):
        return "NaN"
    if math.isinf(value):
        return "Infinity" if value > 0 else "-Infinity"
    if value == 0:
        return "0"
    sign = "-" if value < 0 else ""
    parsed = Decimal(repr(abs(value))).normalize().as_tuple()
    digits = "".join(map(str, parsed.digits))
    k = len(digits)
    n = int(parsed.exponent) + k
    if k <= n <= 21:
        return sign + digits + "0" * (n - k)
    if 0 < n <= 21:
        return sign + digits[:n] + "." + digits[n:]
    if -6 < n <= 0:
        return sign + "0." + "0" * -n + digits
    mantissa = digits if k == 1 else f"{digits[0]}.{digits[1:]}"
    return f"{sign}{mantissa}e{'+' if n > 0 else '-'}{abs(n - 1)}"


def _js_json(value: Any) -> str:
    """``value`` as JavaScript's ``JSON.stringify`` writes it: compact, numbers as JS prints them.

    Raises:
        TypeError: If ``value`` holds something JSON cannot represent, such as a set.
    """
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return _js_number(value) if math.isfinite(value) else "null"
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, dict):
        entries = (
            f"{json.dumps(key if isinstance(key, str) else _header_text(key) or 'null', ensure_ascii=False)}:"
            f"{_js_json(item)}"
            for key, item in value.items()
        )
        return "{" + ",".join(entries) + "}"
    if isinstance(value, list | tuple):
        return "[" + ",".join(_js_json(item) for item in value) + "]"
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _header_text(value: Any) -> str | None:
    """A header argument's value as text, as Node's ``headerText`` writes it; ``None`` if null.

    Strings as they are, booleans as ``true``/``false``, numbers as JavaScript prints them
    and anything else as compact JSON. ``str()`` sent Python's spelling instead: ``True``,
    ``1.0``, ``['a', 'b']``.
    """
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, bool | int):
        return _js_json(value)
    if isinstance(value, float):
        return _js_number(value)
    return _js_json(value)


def build_auth_header(api_key: str) -> str:
    token = base64.b64encode(f"{api_key}:".encode()).decode()
    return f"Basic {token}"


async def _buffer_error_body(response: httpx.Response) -> None:
    """Read an error response's body while its stream is still open.

    The ``mcp`` client raises on the status after the streamed response is closed, so
    reading the body afterwards got nothing and every failure said only "400 Bad
    Request". The body is where the server explains itself: "Legacy accounts cannot be
    used with MCP", or a 412 asking for the account to be re-linked. Node already
    carries it.
    """
    if response.is_error:
        await response.aread()


_HTTP_DATE_START = re.compile(r"(Mon|Tue|Wed|Thu|Fri|Sat|Sun)[a-z]*[, ]")


def _retry_after_seconds(value: str | None) -> float | None:
    """A Retry-After header as seconds from now: delta-seconds or an HTTP-date.

    ``None`` when the header is absent or unreadable, so the caller falls back to its
    own backoff. A date in the past means retry now.
    """
    if value is None:
        return None
    value = value.strip()
    # ASCII digits only: str.isdigit() also accepts "²", which float() then refuses.
    if re.fullmatch(r"[0-9]+", value):
        return float(value)
    # All three HTTP-date forms start with a weekday ("Sun,", "Sunday,", "Sun "); anything
    # else is unreadable, as in Node, and falls back to the backoff.
    if not _HTTP_DATE_START.match(value):
        return None
    try:
        when = parsedate_to_datetime(value)
    except (TypeError, ValueError, IndexError, OverflowError):
        # OverflowError: a year too large for a C long, e.g. 99999999999999999999.
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=UTC)
    return max(0.0, (when - datetime.now(UTC)).total_seconds())


def _rate_limit_delay(response: httpx.Response, retry: int) -> float:
    """How long to wait before retry number ``retry`` (1-based) of a 429."""
    delay = _retry_after_seconds(response.headers.get("retry-after"))
    if delay is None:
        delay = RATE_LIMIT_BACKOFF_SECONDS[retry - 1] * random.uniform(*RATE_LIMIT_JITTER)
    return min(delay, RATE_LIMIT_MAX_DELAY_SECONDS)


def _log_rate_limit_retry(request: httpx.Request, attempt: int, delay: float) -> None:
    logger.warning(
        "%s %s was rate limited (429) on attempt %d of %d; retrying in %.2fs",
        request.method,
        request.url,
        attempt,
        RATE_LIMIT_MAX_RETRIES + 1,
        delay,
    )


def _outlasts_deadline(request: httpx.Request, attempt: int, delay: float, remaining: float) -> bool:
    """Whether waiting ``delay`` would not end before the deadline, logging when so.

    Such a wait is not started. Sleeping into the deadline would turn the 429 into a
    timeout, which a multi-account call skips like any per-account failure, and hand back
    a partial catalog that looks complete. The 429 itself ends the call.
    """
    if delay < remaining:
        return False
    logger.warning(
        "%s %s was rate limited (429) on attempt %d of %d; not retrying, because waiting "
        "%.2fs would outlast the request's deadline (%.2fs left)",
        request.method,
        request.url,
        attempt,
        RATE_LIMIT_MAX_RETRIES + 1,
        delay,
        max(remaining, 0.0),
    )
    return True


class RateLimitRetryingClient(httpx.Client):
    """An ``httpx.Client`` that retries a 429 before the caller ever sees it.

    Retried in ``send`` rather than in a wrapping transport: passing a custom transport
    makes httpx ignore the proxy environment variables, which would quietly break the SDK
    behind a corporate proxy. The last 429 is returned as it came, so the caller reports
    the server's body.

    ``retry_within`` is the deadline in seconds, measured from the first attempt: a retry
    whose wait would not end before it is not made, and the 429 is returned instead.
    """

    def __init__(self, *args: Any, retry_within: float | None = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._retry_within = retry_within

    def send(self, request: httpx.Request, **kwargs: Any) -> httpx.Response:
        deadline = _clock() + self._retry_within if self._retry_within is not None else None
        attempt = 1
        response = super().send(request, **kwargs)
        while response.status_code == 429 and attempt <= RATE_LIMIT_MAX_RETRIES:
            delay = _rate_limit_delay(response, attempt)
            if deadline is not None and _outlasts_deadline(request, attempt, delay, deadline - _clock()):
                break
            # Read before closing, so the connection goes back to the pool.
            response.read()
            response.close()
            _log_rate_limit_retry(request, attempt, delay)
            _sleep(delay)
            attempt += 1
            response = super().send(request, **kwargs)
        return response


class RateLimitRetryingAsyncClient(httpx.AsyncClient):
    """The async counterpart of :class:`RateLimitRetryingClient`, used for every MCP request.

    The MCP client streams its responses; a discarded 429 is read and closed here, and
    the final one still passes through the response hooks, so its body is buffered.

    The deadline is the enclosing cancel scope's: the ``anyio.fail_after(timeout)`` that
    bounds the whole MCP exchange. A retry whose wait would not end before it is not made.
    """

    async def send(self, request: httpx.Request, **kwargs: Any) -> httpx.Response:
        attempt = 1
        response = await super().send(request, **kwargs)
        while response.status_code == 429 and attempt <= RATE_LIMIT_MAX_RETRIES:
            delay = _rate_limit_delay(response, attempt)
            remaining = anyio.current_effective_deadline() - _async_clock()
            if _outlasts_deadline(request, attempt, delay, remaining):
                break
            await response.aread()
            await response.aclose()
            _log_rate_limit_retry(request, attempt, delay)
            await _async_sleep(delay)
            attempt += 1
            response = await super().send(request, **kwargs)
        return response


def is_rate_limited(exc: BaseException) -> bool:
    """Whether a failure is a 429 that outlasted every retry.

    Such a failure ends the whole call: skipping the account and carrying on would hand
    back a partial catalog that looks complete. Only an HTTP 429 counts, as in Node: a
    tool result whose payload says 429 was never retried, so it is that connector's
    failure, skipped like any other.
    """
    if not (isinstance(exc, StackOneAPIError) and exc.status_code == 429):
        return False
    seen: set[int] = set()
    stack: list[BaseException] = [exc]
    while stack:
        current = stack.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        if isinstance(current, httpx.HTTPStatusError) and current.response.status_code == 429:
            return True
        stack.extend(getattr(current, "exceptions", None) or [])
        stack.extend(e for e in (current.__cause__, current.__context__) if e is not None)
    return False


@asynccontextmanager
async def _mcp_transport(
    endpoint: str, headers: dict[str, str], timeout: float
) -> AsyncIterator[tuple[Any, Any, Any]]:
    """Open the streamable-HTTP transport with the caller's timeout on every leg.

    The client's own defaults are a 30s connect and a 300s SSE read, and neither was
    overridden — so ``StackOneToolSet(timeout=2)`` against a host that accepts and never
    answers hung for five minutes. The execution path honoured ``timeout``; the MCP
    path, which search() and execute() and every listing use, did not.

    The HTTP client is built here with the settings the ``mcp`` package's own factory
    uses (redirects followed), because ``streamable_http_client`` takes a client rather
    than headers and timeouts. It retries a 429, so the MCP client never sees one that
    a retry got past.
    """
    from mcp.client.streamable_http import streamable_http_client  # ty: ignore[unresolved-import]

    async with RateLimitRetryingAsyncClient(
        headers=headers,
        timeout=httpx.Timeout(timeout),
        follow_redirects=True,
        event_hooks={"response": [_buffer_error_body]},
    ) as client:
        async with streamable_http_client(endpoint, http_client=client) as streams:
            yield streams


def fetch_mcp_tools(
    endpoint: str, headers: dict[str, str], *, timeout: float = 60.0
) -> list[McpToolDefinition]:
    """List every tool the MCP endpoint serves, following pagination.

    ``timeout`` bounds the whole exchange, handshake included.
    """
    try:
        from mcp import types as mcp_types  # ty: ignore[unresolved-import]
        from mcp.client.session import ClientSession  # ty: ignore[unresolved-import]
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise ToolsetConfigError(
            "mcp is a core dependency of stackone-ai but could not be imported — reinstall the package."
        ) from exc

    async def _list() -> list[McpToolDefinition]:
        with anyio.fail_after(timeout):
            return await _list_within_deadline()

    async def _list_within_deadline() -> list[McpToolDefinition]:
        async with _mcp_transport(endpoint, headers, timeout) as (
            read_stream,
            write_stream,
            _,
        ):
            session = ClientSession(
                read_stream,
                write_stream,
                client_info=mcp_types.Implementation(name="stackone-ai-python", version=_SDK_VERSION),
            )
            async with session:
                await session.initialize()
                cursor: str | None = None
                collected: list[McpToolDefinition] = []
                while True:
                    result = await session.list_tools(cursor)
                    for tool in result.tools:
                        input_schema = tool.inputSchema or {}
                        collected.append(
                            McpToolDefinition(
                                name=tool.name,
                                description=tool.description,
                                input_schema=dict(input_schema),
                            )
                        )
                    cursor = result.nextCursor
                    if cursor is None:
                        break
                return collected

    try:
        return run_async(_list())
    except BaseException as exc:
        raise _describe_mcp_failure(exc, endpoint, timeout) from exc


def _response_body(response: httpx.Response) -> str:
    """Best-effort read of an error body.

    The MCP transport streams, so `.text` raises until the stream is read, and by
    the time the error surfaces the stream may already be closed. The body is the
    only place the server explains itself, so it is worth trying; an empty string
    just means the message falls back to the status code.
    """
    try:
        return response.text.strip()
    except Exception:
        pass
    try:
        response.read()
        return response.text.strip()
    except Exception:
        return ""


def _describe_mcp_failure(exc: BaseException, endpoint: str, timeout: float) -> Exception:
    """Turn the MCP client's nested failure into something a caller can act on.

    The listing runs inside a TaskGroup, so any HTTP error arrives wrapped in an
    ExceptionGroup whose str() is "unhandled errors in a TaskGroup (1 sub-exception)".
    Unwrap to the underlying error and carry the status code and response body,
    which is where the server explains itself — a dead account, for example,
    answers 412 with "re-link the account to resume". A timeout says so, as in Node.
    """
    seen: set[int] = set()
    stack: list[BaseException] = [exc]
    while stack:
        current = stack.pop()
        if current is None or id(current) in seen:
            continue
        seen.add(id(current))
        # Our own errors are already descriptive. They reach here wrapped in an
        # ExceptionGroup because they are raised inside the session's TaskGroup.
        if isinstance(current, StackOneError | ToolsetError):
            return current
        if isinstance(current, httpx.HTTPStatusError):
            body = _response_body(current.response)
            detail = f": {body}" if body else ""
            # StackOneAPIError carries the status and body as attributes, so a caller
            # can branch on 412 rather than pattern-matching the message.
            return StackOneAPIError(
                f"MCP request to {endpoint} failed with "
                f"{current.response.status_code} {current.response.reason_phrase}{detail}",
                current.response.status_code,
                body or None,
            )
        stack.extend(getattr(current, "exceptions", None) or [])
        stack.extend(e for e in (current.__cause__, current.__context__) if e is not None)

    # The innermost cause, through group members and __cause__ only. An ExceptionGroup's
    # own str() is the "unhandled errors in a TaskGroup" boilerplate that hides the real
    # cause, and a __context__ is only what was being handled when the error was raised:
    # for the deadline's TimeoutError that is the cancellation, or whatever the
    # cancelled task happened to be catching.
    leaf: BaseException = exc
    while True:
        if isinstance(leaf, TimeoutError | httpx.TimeoutException):
            return ToolsetLoadError(f"MCP request to {endpoint} timed out after {timeout:g}s")
        members = getattr(leaf, "exceptions", None)
        if members:
            leaf = members[0]
        elif leaf.__cause__ is not None:
            leaf = leaf.__cause__
        else:
            break
    return ToolsetLoadError(f"MCP request to {endpoint} failed: {type(leaf).__name__}: {leaf}")


def _status_of(parsed: JsonDict) -> int:
    """Dig the HTTP status out of an MCP error payload.

    The transport succeeded, so there is no status on the response itself — but the
    payload carries one, and a caller cannot branch on 0. Both spellings and all the
    wrapper keys the API actually uses are checked: `statusCode` is what this repo's
    own mock emits, and `error`/`data` are real wrappers, so looking only for
    `status_code` under `result` reported 0 for the most likely shapes.
    """
    for candidate in (parsed, parsed.get("result"), parsed.get("error"), parsed.get("data")):
        if not isinstance(candidate, dict):
            continue
        for key in ("status_code", "statusCode"):
            status = candidate.get(key)
            # bool is an int subclass; True is not a status code.
            if isinstance(status, int) and not isinstance(status, bool):
                return status
    return 0


def _reject_json_constant(constant: str) -> Any:
    raise ValueError(f"{constant} is not JSON")


def parse_tool_result(result: Any, name: str) -> JsonDict:
    """Turn an MCP ``CallToolResult`` into a plain dict.

    Text parts are joined and parsed as JSON; a non-object is wrapped as
    ``{"result": ...}``. With no text at all, ``structuredContent`` is used, and text
    wins when both are present. Parts that are not text (images, embedded resources)
    are kept under ``content_parts``.

    The result is returned as the server wrote it. For an action tool, ``*_execute_action``
    and feedback that is ``{"isError": false, "result": ..., "defenderMetadata"?: ...,
    "policyMetadata"?: ...}``; a search result is bare JSON.

    Raises:
        StackOneAPIError: If the result carries ``isError``. A failed tool call comes
            back as an ordinary response with that flag set, so without this check the
            error body is handed to the caller as though it were a success.
    """
    texts = [getattr(part, "text", "") for part in result.content]
    payload = "".join(t for t in texts if t)
    # Parts that are not text (images, embedded resources) have no `.text`; keep them
    # rather than silently returning an empty dict.
    non_text = [part for part in result.content if not getattr(part, "text", "")]

    parsed: JsonDict = {}
    structured = getattr(result, "structuredContent", None)
    if payload:
        try:
            # NaN and Infinity are not JSON; json.loads accepts them, JSON.parse does not,
            # and the Node SDK hands such a payload back as text.
            loaded = json.loads(payload, parse_constant=_reject_json_constant)
        except ValueError:
            loaded = payload
        parsed = loaded if isinstance(loaded, dict) else {"result": loaded}
    elif isinstance(structured, dict):
        # A result may carry only structuredContent. Without this it came back as `{}`,
        # a success with the whole payload missing. A copy, so content_parts below does
        # not write into the caller's result.
        parsed = dict(structured)

    if getattr(result, "isError", False):
        # The transport succeeded, so there is no HTTP status here — but the payload
        # carries the real one, and a caller cannot branch on 0.
        detail = payload or json.dumps(parsed, ensure_ascii=False, separators=(",", ":"), default=str)
        raise StackOneAPIError(f'Tool "{name}" failed: {detail}', _status_of(parsed), parsed)

    if non_text:
        parsed["content_parts"] = non_text
    return parsed


def call_mcp_tool(
    endpoint: str, headers: dict[str, str], name: str, arguments: JsonDict, *, timeout: float = 60.0
) -> JsonDict:
    """Invoke a tool over MCP ``tools/call``, the one way every tool is executed.

    ``arguments`` are sent exactly as given; the server maps them onto the action.
    """
    from mcp import types as mcp_types  # ty: ignore[unresolved-import]
    from mcp.client.session import ClientSession  # ty: ignore[unresolved-import]

    async def _call() -> JsonDict:
        with anyio.fail_after(timeout):
            return await _call_within_deadline()

    async def _call_within_deadline() -> JsonDict:
        async with _mcp_transport(endpoint, headers, timeout) as (
            read_stream,
            write_stream,
            _,
        ):
            session = ClientSession(
                read_stream,
                write_stream,
                client_info=mcp_types.Implementation(name="stackone-ai-python", version=_SDK_VERSION),
            )
            async with session:
                await session.initialize()
                # send_request rather than call_tool: call_tool validates structuredContent
                # against the tool's output schema, and on a fresh session that means a
                # tools/list first — relisting the whole catalog on every call, which the
                # Node SDK never does. The result is parsed below either way.
                request = mcp_types.ClientRequest(
                    mcp_types.CallToolRequest(
                        params=mcp_types.CallToolRequestParams(name=name, arguments=arguments),
                    )
                )
                result = await session.send_request(request, mcp_types.CallToolResult)
                return parse_tool_result(result, name)

    try:
        return run_async(_call())
    except BaseException as exc:
        raise _describe_mcp_failure(exc, endpoint, timeout) from exc


def _strip_internal_keys(schema: Any) -> Any:
    """Recursively drop the SDK's internal markers from a property schema.

    Everything else is preserved verbatim: ``format``, ``pattern``, ``default``,
    ``minimum``/``maximum``, ``oneOf``/``anyOf``, nested ``required`` and any
    keyword the server sends that this SDK has never heard of.
    """
    if isinstance(schema, dict):
        return {
            key: _strip_internal_keys(value)
            for key, value in schema.items()
            if not (key == "nullable" and isinstance(value, bool))
        }
    if isinstance(schema, list):
        return [_strip_internal_keys(item) for item in schema]
    return schema


class StackOneTool(BaseModel):
    """A single tool: its served schema, and how to call it.

    The base class describes a tool and converts it for agent frameworks, but cannot run
    it: :meth:`execute` raises. Tools from :meth:`StackOneToolSet.fetch_tools` are
    :class:`StackOneMcpTool` instances, which execute over MCP ``tools/call``. A hand-built
    tool must override :meth:`execute`.
    """

    name: str = Field(description="Tool name")
    description: str = Field(description="Tool description")
    parameters: ToolParameters = Field(description="Tool parameters")
    _execute_config: ExecuteConfig = PrivateAttr()
    _api_key: str | None = PrivateAttr(default=None)
    _account_id: str | None = PrivateAttr(default=None)

    def __init__(
        self,
        description: str,
        parameters: ToolParameters,
        _execute_config: ExecuteConfig,
        # Optional: the base class cannot execute, so it has nothing to authenticate. Kept
        # as a parameter so existing hand-built tools and overrides that read it still work.
        _api_key: str | None = None,
        _account_id: str | None = None,
    ) -> None:
        super().__init__(
            name=_execute_config.name,
            description=description,
            parameters=parameters,
        )
        self._execute_config = _execute_config
        self._api_key = _api_key
        self._account_id = _account_id

    def _parse_arguments(self, arguments: str | JsonDict | None) -> JsonDict:
        """Arguments as a dict, from a dict, a JSON string, or nothing.

        Raises:
            ValueError: If the string is not JSON, or the value is not a JSON object.
        """
        if arguments is None:
            return {}
        if isinstance(arguments, str):
            try:
                parsed = json.loads(arguments)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in arguments for {self.name!r}: {exc}") from exc
        else:
            parsed = arguments
        if not isinstance(parsed, dict):
            raise ValueError("Tool arguments must be a JSON object")
        return dict(parsed)

    def _declared_headers(self) -> tuple[set[str] | None, set[str]]:
        """The header names this tool's served schema declares, casefolded.

        Returns ``(nested, flat)``: the names under a nested ``headers`` object, casefolded,
        and each flat ``headers_<name>`` property exactly as served — a flat header is a
        top-level argument, so it is declared only under its own key, as in Node.
        ``nested`` is ``None`` when the ``headers`` object is open — ``type: "object"`` with
        no ``properties`` and ``additionalProperties`` not ``false``, as ``*_execute_action``
        serves it — which declares every name. Any other schema with no ``properties``
        declares none.
        """
        properties = self.parameters.properties or {}
        flat = {prop for prop in properties if prop.startswith(_FLAT_HEADER_PREFIX)}
        nested: set[str] | None = set()
        schema = properties.get("headers")
        if isinstance(schema, dict):
            if "properties" not in schema:
                if schema.get("type") == "object" and schema.get("additionalProperties") is not False:
                    nested = None
            elif isinstance(schema["properties"], dict):
                nested = {str(name).casefold() for name in schema["properties"]}
        return nested, flat

    @staticmethod
    def _header_refusal(name: str, declared: bool) -> str | None:
        """Why a header argument may not be forwarded, or ``None`` if it may."""
        # Normalise before comparing: " x-foo" and "X-FOO\t" are the same header to any
        # server, and casefold() closes the non-ASCII folding holes lower() leaves.
        if name.strip().casefold() in _SDK_OWNED_HEADERS:
            return "it is set by the SDK"
        if not declared:
            return "it is not declared by the schema"
        return None

    @staticmethod
    def _is_well_formed_header(name: str, text: str) -> bool:
        # Defence in depth on a declared header's model-supplied value. fullmatch, not
        # match: `$` also matches just before a trailing newline, so `match` let "value\n"
        # — the one character class this rejects — straight through.
        return bool(_HEADER_NAME_PATTERN.fullmatch(name) and _HEADER_VALUE_PATTERN.fullmatch(text))

    def _sanitise_headers(self, supplied: dict[str, Any] | None) -> dict[str, str]:
        """Keep only the entries of a nested ``headers`` argument the served schema declares.

        An allowlist, not a denylist. Tool arguments are model-controlled, so a
        prompt-injected call reaches this dict directly — and a denylist has to
        enumerate every synonym of "credential" and "tenant selector" in every
        provider's vocabulary (Proxy-Authorization, x-stackone-account-id, Cookie,
        X-Api-Key, ...) and is wrong the moment one is missed.

        The allowlist is the served schema itself, so this needs no maintenance. An open
        ``headers`` object declares every name. Authorization, x-account-id and User-Agent
        are refused even when declared: the SDK sets them itself.
        """
        declared, _ = self._declared_headers()

        clean: dict[str, str] = {}
        for key, value in (supplied or {}).items():
            text = _header_text(value)
            if text is None or not isinstance(key, str):
                continue
            name = key.strip()
            reason = self._header_refusal(name, declared is None or name.casefold() in declared)
            if reason:
                logger.warning("Dropping header %r from a tool call: %s", name, reason)
                continue
            if not self._is_well_formed_header(name, text):
                logger.warning("Dropping malformed header %r from a tool call", name)
                continue
            clean[name] = text
        return clean

    def _sanitise_header_arguments(self, arguments: JsonDict) -> JsonDict:
        """Filter the header arguments in a call; every other argument is kept unchanged.

        A header argument is an entry of a nested ``headers`` object, or a top-level
        ``headers_<name>``. Each is forwarded only if the served schema declares it in the
        same form, and never if it is one the SDK sets itself. A declared ``headers_<name>``
        keeps its value as given, as every other top-level argument does; nested entries
        are written as text, as Node writes them.

        Raises:
            TypeError: If a header value holds something JSON cannot represent.
        """
        _, declared_flat = self._declared_headers()
        clean: JsonDict = {}
        for key, value in arguments.items():
            if key == "headers" and isinstance(value, dict):
                clean[key] = self._sanitise_headers(value)
                continue
            if not key.startswith(_FLAT_HEADER_PREFIX):
                clean[key] = value
                continue
            text = _header_text(value)
            if text is None:
                continue
            name = key[len(_FLAT_HEADER_PREFIX) :]
            reason = self._header_refusal(name, key in declared_flat)
            if reason:
                logger.warning("Dropping header argument %r from a tool call: %s", key, reason)
                continue
            if not self._is_well_formed_header(name, text):
                logger.warning("Dropping malformed header argument %r from a tool call", key)
                continue
            clean[key] = value
        return clean

    def execute(self, arguments: str | JsonDict | None = None) -> JsonDict:
        """Execute the tool. Not implemented on the base class.

        Tools from :meth:`StackOneToolSet.fetch_tools` execute over MCP ``tools/call``.
        A hand-built ``StackOneTool`` has nothing to call, so override this to run one.

        Raises:
            StackOneError: Always, on the base class.
        """
        raise StackOneError(
            f'Tool "{self.name}" has no executor. Override execute() to run a hand-built tool.'
        )

    def call(self, *args: Any, **kwargs: Any) -> JsonDict:
        """Call the tool with the given arguments

        Examples:
            >>> tool.call({"name": "John", "email": "john@example.com"})
            >>> tool.call(name="John", email="john@example.com")
        """
        if args and kwargs:
            raise ValueError("Cannot provide both positional and keyword arguments")

        if args:
            if len(args) > 1:
                raise ValueError("Only one positional argument is allowed")
            return self.execute(args[0])

        return self.execute(kwargs if kwargs else None)

    def __call__(self, *args: Any, **kwargs: Any) -> JsonDict:
        """Make the tool directly callable. Alias for :meth:`call`."""
        return self.call(*args, **kwargs)

    def to_openai_function(self) -> JsonDict:
        """Convert this tool to OpenAI's function format.

        The served schema is passed through verbatim apart from the SDK's internal
        ``nullable`` marker, which is stripped. Constraints such as ``format``,
        ``pattern``, ``default``, ``minimum``/``maximum`` and ``oneOf``/``anyOf``
        reach the model intact — without them a model cannot generate valid arguments
        for a constrained field. ``required`` is the served list, in the served order,
        and is omitted when the server sent none or an empty one.
        """
        properties: JsonDict = {}
        for name, prop in self.parameters.properties.items():
            properties[name] = _strip_internal_keys(prop) if isinstance(prop, dict) else {"type": "string"}

        parameters: JsonDict = self.parameters.model_dump()
        parameters["properties"] = properties

        # Taken from the served root, not rebuilt from the per-property markers: that
        # rebuild re-sorted `required` into property order, so the model saw a list the
        # server never sent. A non-list `required`, or a non-string entry in one, is
        # malformed and dropped rather than iterated as characters or sent on.
        required = parameters.pop("required", None)
        names = [name for name in required if isinstance(name, str)] if isinstance(required, list) else []
        if names:
            parameters["required"] = names

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": parameters,
            },
        }

    def to_langchain(self) -> Any:
        """Convert this tool to LangChain format.

        Returns a ``langchain_core.tools.BaseTool``, typed as ``Any`` because
        ``langchain-core`` is an optional dependency and must not be imported at
        module level.

        Requires ``stackone-ai[langchain]``.
        """
        try:
            from langchain_core.tools import BaseTool, ToolException
        except ImportError as e:
            raise ImportError(
                "Install `langchain-core` (or `stackone-ai[langchain]`) to use the LangChain integration."
            ) from e

        # Hand the served JSON Schema over as-is. The previous version rebuilt a
        # pydantic model from the top-level `type` of each property, which threw away
        # every nested object's fields, every enum, format, bound, item type and union —
        # roughly half the schema, so the model was told "pass an object" with no field
        # names. langchain-core accepts a JSON Schema dict directly, which makes this
        # surface byte-equivalent to to_openai_function() for free.
        args_json_schema = self.to_openai_function()["function"]["parameters"]

        parent_tool = self

        class StackOneLangChainTool(BaseTool):
            name: str = parent_tool.name
            description: str = parent_tool.description
            args_schema: dict[str, Any] = args_json_schema  # ty: ignore[invalid-assignment]

            def _run(self, **kwargs: Any) -> Any:
                # Drop unsupplied optionals. pydantic materialises every optional field
                # as None and BaseTool passes them all through, and the API reads an
                # explicit null as "required field missing" — which 400'd every single
                # tool call made through this adapter.
                supplied = {key: value for key, value in kwargs.items() if value is not None}
                try:
                    return parent_tool.execute(supplied)
                except (StackOneError, ValueError) as exc:
                    # LangChain's handle_tool_error only catches ToolException, so a
                    # StackOneError would kill the graph rather than reaching the
                    # agent. Models guess arguments wrong and StackOne's 400 names the
                    # offending field — re-raising in LangChain's own vocabulary lets
                    # a caller opt into feeding that back and retrying.
                    # Carry the structured fields across: without them handle_tool_error
                    # leaves a caller with only str(exc), unable to tell 401 from 429.
                    # The field that is actually wrong is often only in response_body.
                    # Without it the agent retries blind, which is the whole point of
                    # handing the error back.
                    body = getattr(exc, "response_body", None)
                    failure = ToolException(f"{exc}: {body}" if body else str(exc))
                    failure.status_code = getattr(exc, "status_code", None)  # type: ignore[attr-defined]
                    failure.response_body = getattr(exc, "response_body", None)  # type: ignore[attr-defined]
                    raise failure from exc

        return StackOneLangChainTool()

    def to_pydantic_ai_tool(self) -> Any:
        """Convert this tool to a Pydantic AI ``Tool``.

        Returns ``pydantic_ai.tools.Tool``, typed as ``Any`` because ``pydantic-ai``
        is an optional dependency and must not be imported at module level.

        Requires ``stackone-ai[pydantic-ai]`` (installs ``pydantic-ai-slim``).
        """
        try:
            from pydantic_ai.exceptions import ModelRetry
            from pydantic_ai.tools import Tool
        except ImportError as e:
            raise ImportError(
                "Install `pydantic-ai-slim` (or `stackone-ai[pydantic-ai]`) "
                "to use the Pydantic AI integration."
            ) from e

        openai_function = self.to_openai_function()
        json_schema = openai_function["function"]["parameters"]
        parent_tool = self

        def implementation(**kwargs: Any) -> Any:
            try:
                return parent_tool.execute(kwargs)
            except (StackOneError, ValueError) as exc:
                # Tool.from_schema skips argument validation entirely, so every wrong
                # guess the model makes reaches the API — and a raised StackOneError
                # escaped the agent loop and ended the run. ModelRetry hands the
                # server's explanation back to the model so it can correct itself,
                # which is what the LangChain adapter already does via ToolException.
                body = getattr(exc, "response_body", None)
                raise ModelRetry(f"{exc}: {body}" if body else str(exc)) from exc

        return Tool.from_schema(
            function=implementation,
            name=self.name,
            description=self.description,
            json_schema=json_schema,
        )

    def set_account_id(self, account_id: str | None) -> None:
        """Set the account ID for this tool"""
        self._account_id = account_id

    def get_account_id(self) -> str | None:
        """Get the current account ID for this tool"""
        return self._account_id


class StackOneMcpTool(StackOneTool):
    """A tool executed over MCP ``tools/call`` on the endpoint that listed it.

    Every tool the toolset builds is one of these: per-action tools, the
    ``*_search_actions`` / ``*_execute_action`` meta tools and ``stackone_submit_feedback``.
    """

    _endpoint: str = PrivateAttr()

    def __init__(
        self,
        *,
        name: str,
        description: str,
        parameters: ToolParameters,
        api_key: str,
        endpoint: str,
        account_id: str | None,
        headers: Headers | None = None,
        timeout: float = 60.0,
    ) -> None:
        super().__init__(
            description=description,
            parameters=parameters,
            _execute_config=ExecuteConfig(name=name, headers=dict(headers or {}), timeout=timeout),
            _api_key=api_key,
            _account_id=account_id,
        )
        self._endpoint = endpoint

    def _prepare_headers(self) -> Headers:
        """The request headers: the configured extras first, then the SDK's own.

        Authorization, x-account-id and User-Agent are set last, and any case variant of
        them among the extras is dropped first, so neither can replace the credential or
        retarget the call at another account.
        """
        headers: Headers = {
            name: value
            for name, value in self._execute_config.headers.items()
            if name.strip().casefold() not in _SDK_OWNED_HEADERS
        }
        # The constructor requires the key; only code that clears it afterwards gets here.
        if not self._api_key:
            raise StackOneError(f'Tool "{self.name}" has no API key to authenticate with.')
        headers["User-Agent"] = USER_AGENT
        headers["Authorization"] = build_auth_header(self._api_key)
        if self._account_id:
            headers["x-account-id"] = self._account_id
        return headers

    def execute(self, arguments: str | JsonDict | None = None) -> JsonDict:
        """Call the tool over MCP ``tools/call``.

        Arguments are sent as given; the server maps them onto the action. Header arguments
        — the entries of a ``headers`` object and any ``headers_<name>`` — are the exception:
        these arguments are model-controlled, so each is kept only if this tool's own schema
        declares it, and never if it is Authorization, x-account-id or User-Agent.

        Returns:
            The tool's result as the server wrote it (see :func:`parse_tool_result`): for an
            action, ``{"isError": false, "result": ..., ...}``. A file action's ``result`` is a
            download link, not the file.

        Raises:
            StackOneAPIError: If the result carries ``isError``, with the status from its
                payload, or the endpoint answers with an HTTP error.
            ValueError: If the arguments are not a JSON object or cannot be encoded.
        """
        parsed = self._parse_arguments(arguments)

        try:
            # Without this a prompt-injected call could put its own Authorization or
            # x-account-id in a header argument the server unpacks.
            parsed = self._sanitise_header_arguments(parsed)
            json.dumps(parsed, ensure_ascii=False, allow_nan=False).encode("utf-8")
        except (UnicodeEncodeError, TypeError, ValueError) as exc:
            # A lone surrogate — what a model emits when a token boundary splits an emoji —
            # or a value JSON cannot encode (a set, bytes) would otherwise fail deep inside
            # the MCP client and surface as a transport error. It is an argument problem.
            # NaN and Infinity are not JSON either; the MCP client would send them as null.
            raise ValueError(f"Arguments for {self.name!r} could not be encoded as JSON: {exc}") from exc

        return call_mcp_tool(
            self._endpoint, self._prepare_headers(), self.name, parsed, timeout=self._execute_config.timeout
        )


def _read_openai_tool_call(call: Any) -> tuple[str, str, str | JsonDict]:
    """Pull (id, name, arguments) from an openai ToolCall object or its dict form."""
    if isinstance(call, dict):
        function = call.get("function") or {}
        return str(call.get("id", "")), str(function.get("name", "")), function.get("arguments") or {}
    function = call.function
    return str(call.id), str(function.name), function.arguments or {}


class Tools:
    """Container for Tool instances with lookup capabilities"""

    def __init__(self, tools: list[StackOneTool]) -> None:
        self.tools = tools
        # First listing wins, as Node's getTool() finds the first match.
        self._tool_map: dict[str, StackOneTool] = {}
        for tool in tools:
            self._tool_map.setdefault(tool.name, tool)

        # Two accounts on one provider serve identically named tools, and get_tool()
        # returns only one of them. OpenAI accepts the duplicate function names without
        # complaint, so nothing downstream surfaces it either — the only symptom is an
        # action running against an account the caller never chose.
        if len(self._tool_map) != len(tools):
            counts = Counter(tool.name for tool in tools)
            clashing = sorted(name for name, count in counts.items() if count > 1)
            logger.warning(
                "%d tool name(s) are served by more than one account (%s). get_tool() will "
                "return the first one listed — pass account_ids to choose.",
                len(clashing),
                ", ".join(clashing[:5]),
            )

    def __getitem__(self, index: int) -> StackOneTool:
        return self.tools[index]

    def __len__(self) -> int:
        return len(self.tools)

    def __iter__(self) -> Any:
        """Make Tools iterable"""
        return iter(self.tools)

    def to_list(self) -> list[StackOneTool]:
        """Convert to list of tools"""
        return list(self.tools)

    def get_tool(self, name: str) -> StackOneTool | None:
        """Get a tool by its name, or None if absent.

        When more than one account serves the name, this is the first one listed.
        """
        return self._tool_map.get(name)

    def set_account_id(self, account_id: str | None) -> None:
        """Set the account ID for all tools in this collection"""
        for tool in self.tools:
            tool.set_account_id(account_id)

    def get_account_id(self) -> str | None:
        """Get the first non-None account ID found, or None if none set"""
        for tool in self.tools:
            account_id = tool.get_account_id()
            if isinstance(account_id, str):
                return account_id
        return None

    def to_openai(self) -> list[JsonDict]:
        """Convert all tools to OpenAI function format"""
        return [tool.to_openai_function() for tool in self.tools]

    def execute_openai_tool_calls(self, tool_calls: Iterable[Any] | None) -> list[JsonDict]:
        """Run a Chat Completions response's tool calls and return the ``tool`` messages.

        The counterpart to :meth:`to_openai`: that turns these tools into what OpenAI
        accepts, this turns what OpenAI returns back into messages to send it. Append
        the assistant message first, then these, in order::

            message = response.choices[0].message
            messages.append(message.model_dump(exclude_none=True))
            messages.extend(tools.execute_openai_tool_calls(message.tool_calls))

        A failed call does not raise. Its error becomes the tool message's content, so
        the model can read why and retry — the same thing the LangChain and Pydantic AI
        adapters do. A call to a tool that is not in this collection is reported the
        same way.

        Accepts the ``openai`` package's objects or plain dicts, so it needs no extra.
        """
        messages: list[JsonDict] = []
        for call in tool_calls or []:
            call_id, name, arguments = _read_openai_tool_call(call)
            tool = self.get_tool(name)
            if tool is None:
                result: Any = {"error": f"Unknown tool {name!r}"}
            else:
                try:
                    result = tool.execute(arguments)
                except (StackOneError, ValueError) as exc:
                    body = getattr(exc, "response_body", None)
                    result = {"error": str(exc), **({"response_body": body} if body else {})}
            # default=str: non-text content parts (images, embedded resources) are objects.
            content = json.dumps(result, default=str)
            messages.append({"role": "tool", "tool_call_id": call_id, "content": content})
        return messages

    def to_langchain(self) -> Sequence[Any]:
        """Convert all tools to LangChain format.

        Requires ``stackone-ai[langchain]``.
        """
        return [tool.to_langchain() for tool in self.tools]

    def to_pydantic_ai(self) -> list[Any]:
        """Convert all tools to Pydantic AI ``Tool`` instances.

        Requires ``stackone-ai[pydantic-ai]`` (installs ``pydantic-ai-slim``).
        """
        return [tool.to_pydantic_ai_tool() for tool in self.tools]
