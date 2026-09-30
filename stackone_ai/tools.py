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
import re
import threading
from collections import Counter
from collections.abc import Coroutine, Iterable, Sequence
from dataclasses import dataclass
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


# Added per property by the toolset when normalising a served schema; it records
# whether the field was absent from the schema's `required` list. It is an internal
# marker, not part of the served schema, so it is stripped before a schema is
# handed to a model.
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


def build_auth_header(api_key: str) -> str:
    token = base64.b64encode(f"{api_key}:".encode()).decode()
    return f"Basic {token}"


def _mcp_transport(client: Any, endpoint: str, headers: dict[str, str], timeout: float) -> Any:
    """Open the streamable-HTTP transport with the caller's timeout on every leg.

    The client's own defaults are a 30s connect and a 300s SSE read, and neither was
    overridden — so ``StackOneToolSet(timeout=2)`` against a host that accepts and never
    answers hung for five minutes. The execution path honoured ``timeout``; the MCP
    path, which search() and execute() and every listing use, did not.
    """
    return client(endpoint, headers=headers, timeout=timeout, sse_read_timeout=timeout)


def fetch_mcp_tools(
    endpoint: str, headers: dict[str, str], *, timeout: float = 60.0
) -> list[McpToolDefinition]:
    """List every tool the MCP endpoint serves, following pagination.

    ``timeout`` bounds the whole exchange, handshake included.
    """
    try:
        from mcp import types as mcp_types  # ty: ignore[unresolved-import]
        from mcp.client.session import ClientSession  # ty: ignore[unresolved-import]
        from mcp.client.streamable_http import streamablehttp_client  # ty: ignore[unresolved-import]
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise ToolsetConfigError(
            "mcp is a core dependency of stackone-ai but could not be imported — reinstall the package."
        ) from exc

    async def _list() -> list[McpToolDefinition]:
        with anyio.fail_after(timeout):
            return await _list_within_deadline()

    async def _list_within_deadline() -> list[McpToolDefinition]:
        async with _mcp_transport(streamablehttp_client, endpoint, headers, timeout) as (
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
        raise _describe_mcp_failure(exc, endpoint) from exc


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


def _describe_mcp_failure(exc: BaseException, endpoint: str) -> Exception:
    """Turn the MCP client's nested failure into something a caller can act on.

    The listing runs inside a TaskGroup, so any HTTP error arrives wrapped in an
    ExceptionGroup whose str() is "unhandled errors in a TaskGroup (1 sub-exception)".
    Unwrap to the underlying error and carry the status code and response body,
    which is where the server explains itself — a dead account, for example,
    answers 412 with "re-link the account to resume".
    """
    seen: set[int] = set()
    stack: list[BaseException] = [exc]
    leaf: BaseException = exc
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
        # Track the innermost non-group exception: an ExceptionGroup's own str() is
        # the "unhandled errors in a TaskGroup" boilerplate that hides the real cause.
        if not getattr(current, "exceptions", None):
            leaf = current
        stack.extend(getattr(current, "exceptions", None) or [])
        stack.extend(e for e in (current.__cause__, current.__context__) if e is not None)
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
    from mcp.client.streamable_http import streamablehttp_client  # ty: ignore[unresolved-import]

    async def _call() -> JsonDict:
        with anyio.fail_after(timeout):
            return await _call_within_deadline()

    async def _call_within_deadline() -> JsonDict:
        async with _mcp_transport(streamablehttp_client, endpoint, headers, timeout) as (
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
        raise _describe_mcp_failure(exc, endpoint) from exc


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
    _api_key: str = PrivateAttr()
    _account_id: str | None = PrivateAttr(default=None)

    def __init__(
        self,
        description: str,
        parameters: ToolParameters,
        _execute_config: ExecuteConfig,
        _api_key: str,
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
        headers["User-Agent"] = USER_AGENT
        headers["Authorization"] = build_auth_header(self._api_key)
        if self._account_id:
            headers["x-account-id"] = self._account_id
        return headers

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

        Returns ``(nested, flat)``: the names under a nested ``headers`` object, and the
        ``<name>`` of each flat ``headers_<name>`` property. ``nested`` is ``None`` when the
        ``headers`` object is open — an object schema with no ``properties``, as
        ``*_execute_action`` serves it — which declares every name.
        """
        properties = self.parameters.properties or {}
        flat = {
            prop[len(_FLAT_HEADER_PREFIX) :].casefold()
            for prop in properties
            if prop.startswith(_FLAT_HEADER_PREFIX)
        }
        nested: set[str] | None = set()
        schema = properties.get("headers")
        if isinstance(schema, dict):
            if "properties" not in schema:
                nested = None
            elif isinstance(schema["properties"], dict):
                nested = {str(name).casefold() for name in schema["properties"]}
        return nested, flat

    @staticmethod
    def _header_refusal(name: str, declared: set[str] | None) -> str | None:
        """Why a header argument may not be forwarded, or ``None`` if it may.

        ``declared`` is ``None`` for an open ``headers`` object, which declares every name.
        """
        # Normalise before comparing: " x-foo" and "X-FOO\t" are the same header to any
        # server, and casefold() closes the non-ASCII folding holes lower() leaves.
        folded = name.strip().casefold()
        if folded in _SDK_OWNED_HEADERS:
            return "it is set by the SDK"
        if declared is not None and folded not in declared:
            return "it is not declared by the schema"
        return None

    @staticmethod
    def _is_well_formed_header(name: str, value: Any) -> bool:
        # Defence in depth on a declared header's model-supplied value. fullmatch, not
        # match: `$` also matches just before a trailing newline, so `match` let "value\n"
        # — the one character class this rejects — straight through.
        return bool(_HEADER_NAME_PATTERN.fullmatch(name) and _HEADER_VALUE_PATTERN.fullmatch(str(value)))

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
            if value is None or not isinstance(key, str):
                continue
            name = key.strip()
            reason = self._header_refusal(name, declared)
            if reason:
                logger.warning("Dropping header %r from a tool call: %s", name, reason)
                continue
            if not self._is_well_formed_header(name, value):
                logger.warning("Dropping malformed header %r from a tool call", name)
                continue
            clean[name] = str(value)
        return clean

    def _sanitise_header_arguments(self, arguments: JsonDict) -> JsonDict:
        """Filter the header arguments in a call; every other argument is kept unchanged.

        A header argument is an entry of a nested ``headers`` object, or a top-level
        ``headers_<name>``. Each is forwarded only if the served schema declares it in the
        same form, and never if it is one the SDK sets itself.
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
            name = key[len(_FLAT_HEADER_PREFIX) :]
            reason = self._header_refusal(name, declared_flat)
            if reason:
                logger.warning("Dropping header argument %r from a tool call: %s", key, reason)
                continue
            if value is None:
                continue
            if not self._is_well_formed_header(name, value):
                logger.warning("Dropping malformed header argument %r from a tool call", key)
                continue
            clean[key] = str(value)
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

        # Without this a prompt-injected call could put its own Authorization or
        # x-account-id in a header argument the server unpacks.
        parsed = self._sanitise_header_arguments(parsed)

        try:
            json.dumps(parsed, ensure_ascii=False).encode("utf-8")
        except (UnicodeEncodeError, TypeError, ValueError) as exc:
            # A lone surrogate — what a model emits when a token boundary splits an emoji —
            # or a value JSON cannot encode (a set, bytes) would otherwise fail deep inside
            # the MCP client and surface as a transport error. It is an argument problem.
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
        self._tool_map = {tool.name: tool for tool in tools}

        # Two accounts on one provider serve identically named tools, so this dict
        # silently kept the last one and get_tool() routed every call to whichever
        # account happened to list last. OpenAI accepts the duplicate function names
        # without complaint, so nothing downstream surfaces it either — the only
        # symptom is an action running against an account the caller never chose.
        if len(self._tool_map) != len(tools):
            counts = Counter(tool.name for tool in tools)
            clashing = sorted(name for name, count in counts.items() if count > 1)
            logger.warning(
                "%d tool name(s) are served by more than one account (%s). get_tool() will "
                "return the last one listed — pass account_ids to choose.",
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
        """Get a tool by its name, or None if absent"""
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
