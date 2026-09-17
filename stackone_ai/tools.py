"""Tools served by the StackOne MCP endpoint, and their execution over the RPC endpoint.

The guiding property of this module is that a tool is the served catalog entry:
the schema listed to a model is the schema the MCP server sent, and the request
sent to ``/actions/rpc`` matches that schema. Nothing is invented, nothing is
dropped, nothing is rebuilt.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import re
import threading
from collections import Counter
from collections.abc import Coroutine, Sequence
from dataclasses import dataclass
from importlib import metadata
from typing import Any, TypeVar, cast

import anyio
import httpx
from pydantic import BaseModel, Field, PrivateAttr

from stackone_ai.types import (
    ExecuteConfig,
    Headers,
    JsonDict,
    ParameterLocation,
    StackOneAPIError,
    StackOneError,
    ToolParameters,
    ToolsetConfigError,
    ToolsetError,
    ToolsetLoadError,
    filename_from_content_disposition,
    is_json_content_type,
)

logger = logging.getLogger("stackone.tools")

T = TypeVar("T")

try:
    _SDK_VERSION = metadata.version("stackone-ai")
except metadata.PackageNotFoundError:  # pragma: no cover - best-effort fallback when running from source
    _SDK_VERSION = "dev"

USER_AGENT = f"stackone-ai-python/{_SDK_VERSION}"

_RPC_PARAMETER_LOCATIONS = {
    "action": ParameterLocation.BODY,
    "body": ParameterLocation.BODY,
    "headers": ParameterLocation.BODY,
    "path": ParameterLocation.BODY,
    "query": ParameterLocation.BODY,
}

# Param-style pinned on the /mcp tool-listing URL. The MCP schema and the RPC-execution unwrap
# (_split_envelope_params) must agree on this, so it is pinned rather than following the server
# default — the server default is free to change without breaking the SDK.
MCP_PARAM_STYLE = "flat_prefixed"

# Matches a flat_prefixed envelope key: `<location>_<field>` (e.g. `path_id`, `query_limit`).
_HEADER_NAME_PATTERN = re.compile(r"[A-Za-z0-9!#$%&'*+.^_`|~-]+")
_HEADER_VALUE_PATTERN = re.compile(r"[\x20-\x7e\t\x80-\xff]*")

_FLAT_ENVELOPE_KEY_PATTERN = re.compile(r"^(path|query|body|headers)_(.+)$")

# Added per property by the toolset when normalising a served schema; it records
# whether the field was absent from the schema's `required` list. It is an internal
# marker, not part of the served schema, so it is stripped before a schema is
# handed to a model.
_INTERNAL_SCHEMA_KEYS = frozenset({"nullable"})


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


def _describe_api_failure(exc: httpx.HTTPStatusError, body: Any) -> str:
    """Lead with the server's own explanation of what went wrong.

    ``str(httpx.HTTPStatusError)`` is "Client error '400 Bad Request' for url ..." plus a
    link to MDN's generic status page — it never says which field was wrong, even though
    the answer is already in hand. This is the error a user hits on every bad tool call,
    so it is the one worth making actionable.
    """
    detail: Any = None
    if isinstance(body, dict):
        for key in ("message", "error", "detail"):
            if isinstance(body.get(key), str):
                detail = body[key]
                break
        else:
            detail = body
    elif isinstance(body, str) and body.strip():
        detail = body.strip()[:500]

    status = f"{exc.response.status_code} {exc.response.reason_phrase}".strip()
    return f"{status}: {detail}" if detail else f"{status} from {exc.request.url}"


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


def parse_tool_result(result: Any, name: str) -> JsonDict:
    """Turn an MCP ``CallToolResult`` into a plain dict.

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
    if payload:
        try:
            loaded = json.loads(payload)
        except json.JSONDecodeError:
            loaded = payload
        parsed = loaded if isinstance(loaded, dict) else {"result": loaded}

    if getattr(result, "isError", False):
        # The transport succeeded, so there is no HTTP status here — but the payload
        # carries the real one, and a caller cannot branch on 0.
        raise StackOneAPIError(f"Tool {name!r} failed: {payload or parsed}", _status_of(parsed), parsed)

    if non_text:
        parsed["content_parts"] = non_text
    return parsed


def call_mcp_tool(
    endpoint: str, headers: dict[str, str], name: str, arguments: JsonDict, *, timeout: float = 60.0
) -> JsonDict:
    """Invoke a tool over MCP ``tools/call``.

    The search/execute meta tools exist only on the MCP endpoint — they have no
    ``/actions/rpc`` action behind them — so they must be called this way.
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
                return parse_tool_result(await session.call_tool(name, arguments), name)

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
            if key not in _INTERNAL_SCHEMA_KEYS
        }
    if isinstance(schema, list):
        return [_strip_internal_keys(item) for item in schema]
    return schema


class StackOneTool(BaseModel):
    """A single tool: its served schema plus the request needed to execute it."""

    name: str = Field(description="Tool name")
    description: str = Field(description="Tool description")
    parameters: ToolParameters = Field(description="Tool parameters")
    _execute_config: ExecuteConfig = PrivateAttr()
    _api_key: str = PrivateAttr()
    _account_id: str | None = PrivateAttr(default=None)

    @property
    def connector(self) -> str:
        """Extract connector from tool name.

        Tool names follow the format: {connector}_{action}_{entity}
        e.g., 'bamboohr_create_employee' -> 'bamboohr'
        """
        return self.name.split("_")[0].lower()

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
        """Prepare headers for the API request"""
        headers: Headers = {
            "Authorization": build_auth_header(self._api_key),
            "User-Agent": USER_AGENT,
        }

        if self._account_id:
            headers["x-account-id"] = self._account_id

        headers.update(self._execute_config.headers)
        return headers

    def _prepare_request_params(self, kwargs: JsonDict) -> tuple[str, JsonDict, JsonDict]:
        """Prepare URL and parameters for the API request

        Returns:
            Tuple of (url, body_params, query_params)
        """
        from urllib.parse import quote

        url = self._execute_config.url
        body_params: JsonDict = {}
        query_params: JsonDict = {}

        for key, value in kwargs.items():
            param_location = self._execute_config.parameter_locations.get(key)

            if param_location == ParameterLocation.PATH:
                # Safely encode path parameters to prevent SSRF attacks
                encoded_value = quote(str(value), safe="")
                url = url.replace(f"{{{key}}}", encoded_value)
            elif param_location == ParameterLocation.QUERY:
                query_params[key] = value
            elif param_location in (ParameterLocation.BODY, ParameterLocation.FILE):
                body_params[key] = value
            else:
                if f"{{{key}}}" in url:
                    encoded_value = quote(str(value), safe="")
                    url = url.replace(f"{{{key}}}", encoded_value)
                elif self._execute_config.method in {"GET", "DELETE"}:
                    query_params[key] = value
                else:
                    body_params[key] = value

        return url, body_params, query_params

    def _sanitise_headers(self, supplied: dict[str, Any] | None) -> dict[str, str]:
        """Keep only headers the served schema declared; drop everything else.

        An allowlist, not a denylist. Tool arguments are model-controlled, so a
        prompt-injected call reaches this dict directly — and a denylist has to
        enumerate every synonym of "credential" and "tenant selector" in every
        provider's vocabulary (Proxy-Authorization, x-stackone-account-id, Cookie,
        X-Api-Key, ...) and is wrong the moment one is missed. The previous two-name
        list let all of those through.

        The allowlist is the served schema itself, so this needs no maintenance: today
        zero of the served actions declare a ``headers_*`` property, and the RPC server
        ignores the envelope's ``headers`` object outright — but the day an action
        genuinely needs a header, it works with no SDK release.

        Lives on the base class because both execution paths need it: the RPC tool
        builds the envelope's headers, and the MCP tool — what search_execute mode
        returns, and so what the documented search()/execute() flow uses — forwards the
        model's headers object into tools/call.
        """
        allowed = {
            prop[len("headers_") :].casefold()
            for prop in (self.parameters.properties or {})
            if prop.startswith("headers_")
        }

        clean: dict[str, str] = {}
        for key, value in (supplied or {}).items():
            if value is None or not isinstance(key, str):
                continue
            # Normalise before comparing: " x-foo" and "X-FOO\t" are the same header to
            # any server, and casefold() closes the non-ASCII folding holes lower() leaves.
            name = key.strip()
            if name.casefold() not in allowed:
                logger.warning("Dropping header %r from a tool call: no served schema declares it", name)
                continue
            # Defence in depth on a declared header's model-supplied value. fullmatch,
            # not match: `$` also matches just before a trailing newline, so `match` let
            # "value\n" — the one character class this rejects — straight through.
            if not _HEADER_NAME_PATTERN.fullmatch(name) or not _HEADER_VALUE_PATTERN.fullmatch(str(value)):
                logger.warning("Dropping malformed header %r from a tool call", name)
                continue
            clean[name] = str(value)
        return clean

    def execute(self, arguments: str | JsonDict | None = None) -> JsonDict:
        """Execute the tool with the given parameters

        Returns:
            For JSON responses, the parsed API response as a dict.

            For file downloads (any non-JSON Content-Type, e.g. a
            ``documents_download_file`` action), a dict describing the file:
            ``{"content": <bytes>, "content_type": str, "status_code": int,
            "headers": dict, "file_name": str | None}``. Note ``content`` holds
            the raw bytes and is therefore not JSON-serializable - callers that
            re-serialize tool results (e.g. for an LLM) should handle this key.

        Raises:
            StackOneAPIError: If the API request fails
            ValueError: If the arguments are invalid
        """
        try:
            if isinstance(arguments, str):
                parsed_arguments = json.loads(arguments)
            else:
                parsed_arguments = arguments or {}

            if not isinstance(parsed_arguments, dict):
                raise ValueError("Tool arguments must be a JSON object")

            headers = self._prepare_headers()
            url_used, body_params, query_params = self._prepare_request_params(parsed_arguments)

            request_kwargs: dict[str, Any] = {
                "method": self._execute_config.method,
                "url": url_used,
                "headers": headers,
            }

            if body_params:
                body_type = self._execute_config.body_type or "json"
                if body_type == "json":
                    request_kwargs["json"] = body_params
                elif body_type == "form":
                    request_kwargs["data"] = body_params

            if query_params:
                request_kwargs["params"] = query_params

            response = httpx.request(**request_kwargs, timeout=self._execute_config.timeout)
            response.raise_for_status()

            content_type = response.headers.get("content-type", "")
            if response.status_code in (204, 205) or (
                not response.content and is_json_content_type(content_type)
            ):
                # A bodyless JSON success is not a file download. Falling through would
                # return `content: b""` with a made-up octet-stream type, breaking any
                # caller that re-serialises the result for a model. A zero-byte body with
                # a download content type IS a download, though — an empty file, with a
                # filename the caller still needs — so it must not be caught here.
                return {"status_code": response.status_code}
            if is_json_content_type(content_type):
                try:
                    result = response.json()
                except json.JSONDecodeError as exc:
                    # Not the caller's arguments — the server sent a JSON content type
                    # with a body that is not JSON. Saying "invalid JSON in arguments"
                    # here sends people to debug the wrong end of the call.
                    raise StackOneAPIError(
                        f"Server sent malformed JSON for {self.name!r}: {exc}",
                        response.status_code,
                        response.text[:500],
                    ) from exc
                return cast(JsonDict, result) if isinstance(result, dict) else {"result": result}

            # Non-JSON bodies are file downloads (e.g. documents_download_file), which the
            # API serves as raw binary with the file's own MIME type and a Content-Disposition
            # header. Return the bytes plus metadata rather than forcing a JSON/UTF-8 decode.
            return {
                "content": response.content,
                "content_type": content_type or "application/octet-stream",
                "status_code": response.status_code,
                "headers": dict(response.headers),
                "file_name": filename_from_content_disposition(response.headers.get("content-disposition")),
            }

        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in arguments: {exc}") from exc
        except httpx.HTTPStatusError as exc:
            response_body = None
            if exc.response.text:
                try:
                    response_body = exc.response.json()
                except json.JSONDecodeError:
                    response_body = exc.response.text
            raise StackOneAPIError(
                _describe_api_failure(exc, response_body), exc.response.status_code, response_body
            ) from exc
        except httpx.RequestError as exc:
            raise StackOneError(f"Request failed: {exc}") from exc

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
        ``nullable`` marker, which is translated into the JSON Schema ``required``
        list. Constraints such as ``format``, ``pattern``, ``default``,
        ``minimum``/``maximum`` and ``oneOf``/``anyOf`` reach the model intact —
        without them a model cannot generate valid arguments for a constrained field.
        """
        properties: JsonDict = {}
        required: list[str] = []

        for name, prop in self.parameters.properties.items():
            if isinstance(prop, dict):
                properties[name] = _strip_internal_keys(prop)
                if not prop.get("nullable", False):
                    required.append(name)
            else:
                properties[name] = {"type": "string"}
                required.append(name)

        parameters: JsonDict = {
            "type": "object",
            "properties": properties,
        }

        if required:
            parameters["required"] = required

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
                    # str(exc) on an httpx error is just "Client error '400 Bad Request'
                    # for url ..." — the field that is actually wrong is in response_body.
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


class StackOneRpcTool(StackOneTool):
    """RPC-backed tool wired to the StackOne actions RPC endpoint."""

    def __init__(
        self,
        *,
        name: str,
        description: str,
        parameters: ToolParameters,
        api_key: str,
        base_url: str,
        account_id: str | None,
        timeout: float = 60.0,
    ) -> None:
        execute_config = ExecuteConfig(
            method="POST",
            url=f"{base_url.rstrip('/')}/actions/rpc",
            name=name,
            headers={},
            body_type="json",
            parameter_locations=dict(_RPC_PARAMETER_LOCATIONS),
            timeout=timeout,
        )
        super().__init__(
            description=description,
            parameters=parameters,
            _execute_config=execute_config,
            _api_key=api_key,
            _account_id=account_id,
        )

    def execute(self, arguments: str | dict[str, Any] | None = None) -> dict[str, Any]:
        parsed_arguments = self._parse_arguments(arguments)
        envelope = self._split_envelope_params(parsed_arguments, set(self.parameters.properties) or None)

        payload: dict[str, Any] = {
            "action": self.name,
            "body": envelope["body"],
            "headers": self._build_action_headers(envelope["headers"] or None),
        }
        if envelope["path"]:
            payload["path"] = envelope["path"]
        if envelope["query"]:
            payload["query"] = envelope["query"]

        return super().execute(payload)

    def _parse_arguments(self, arguments: str | dict[str, Any] | None) -> dict[str, Any]:
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

    @staticmethod
    def _split_envelope_params(
        params: dict[str, Any], declared: set[str] | None = None
    ) -> dict[str, dict[str, Any]]:
        """Split LLM-supplied tool arguments into the RPC envelope (path/query/headers/body).

        Tools are listed with ``?param-style=flat_prefixed``, so keys arrive as
        ``<location>_<field>`` (for example ``path_id``, ``query_limit``) and the prefix
        carries the parameter location. A bare dict-valued ``path``/``query``/``headers``/
        ``body`` key is still bucketed for clients holding a cached nested schema, and any
        other key falls through to the body.

        ``declared`` is the served schema's property names. The prefix is stripped only from
        a key the server actually declared, because the pattern alone cannot tell
        ``path_id`` (a path param) from ``path_to_file`` (a body field that merely starts
        with "path_"). Splitting the latter would drop the argument from the body and send
        the server a path component it has no use for — silently, with the model none the
        wiser.

        ``None`` or an EMPTY set both mean "no schema to consult" and trust every match:
        a served schema with no usable ``properties`` must not route every ``path_*``
        key into the body and lose every path parameter.
        """
        buckets: dict[str, dict[str, Any]] = {"path": {}, "query": {}, "headers": {}, "body": {}}
        reserved = ("path", "query", "headers", "body")
        named = declared or set()

        # Whether to read a `<location>_<field>` key as located, decided ONCE from the
        # schema rather than per key. Under flat_prefixed every parameter is prefixed,
        # so `path_to_file` means path.to_file; under a bare schema it is a body field
        # that merely starts with "path_". Asking "is THIS key declared?" got that
        # wrong in both directions: it demoted undeclared `query_offset` to a body
        # field silently, and it would still have mis-split a declared bare name.
        # ALL, not any: under flat_prefixed every parameter is prefixed, so one bare
        # name is proof the schema is not. `any` would be satisfied by the very key
        # this exists to protect — a declared body field called `path_to_file`.
        prefixed = not named or all(_FLAT_ENVELOPE_KEY_PATTERN.match(k) for k in named)

        # Two passes so precedence is deterministic rather than following the caller's
        # dict order: an explicit flat_prefixed key always wins over a nested one.
        nested: list[tuple[str, dict[str, Any]]] = []
        bare: list[tuple[str, Any]] = []
        for key, value in params.items():
            match = _FLAT_ENVELOPE_KEY_PATTERN.match(key)
            if match and prefixed:
                buckets[match.group(1)][match.group(2)] = value
                continue
            # A reserved word the schema declares as a property is a field, not a
            # container — refusing it would reject a schema-valid call.
            if key in reserved and key not in named:
                # Reserved keys are containers. A scalar here is malformed input, not a
                # body field — putting it in the body would smuggle `path` into the payload.
                if not isinstance(value, dict):
                    raise ValueError(
                        f"{key!r} is an envelope container and must be an object, "
                        f"got {type(value).__name__}. Did you mean {key}_<field>?"
                    )
                nested.append((key, value))
                continue
            bare.append((key, value))

        # Deferred so precedence is a property of the KIND of key, not of the caller's
        # dict order: flat_prefixed beats nested beats bare, always. Assigning bare keys
        # in the first pass made {"body_foo": 1, "foo": 2} and {"foo": 2, "body_foo": 1}
        # produce different wire bodies — and the Node SDK a third, breaking the
        # cross-language byte-equality the conformance suite asserts.
        for key, value in nested:
            for field, field_value in value.items():
                buckets[key].setdefault(field, field_value)
        for key, value in bare:
            buckets["body"].setdefault(key, value)
        return buckets

    def _build_action_headers(self, additional_headers: dict[str, Any] | None) -> dict[str, str]:
        headers = self._sanitise_headers(additional_headers)

        account_id = self.get_account_id()
        if account_id:
            headers["x-account-id"] = account_id

        return headers


class StackOneMcpTool(StackOneTool):
    """A tool executed over MCP ``tools/call`` rather than the RPC endpoint."""

    _endpoint: str = PrivateAttr()
    _mcp_headers: Headers = PrivateAttr()

    def __init__(
        self,
        *,
        name: str,
        description: str,
        parameters: ToolParameters,
        api_key: str,
        endpoint: str,
        headers: Headers,
        account_id: str | None,
        timeout: float = 60.0,
    ) -> None:
        super().__init__(
            description=description,
            parameters=parameters,
            _execute_config=ExecuteConfig(
                method="POST", url=endpoint, name=name, headers={}, timeout=timeout
            ),
            _api_key=api_key,
            _account_id=account_id,
        )
        self._endpoint = endpoint
        self._mcp_headers = headers

    def execute(self, arguments: str | JsonDict | None = None) -> JsonDict:
        if isinstance(arguments, str):
            try:
                parsed = json.loads(arguments)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in arguments for {self.name!r}: {exc}") from exc
        else:
            parsed = arguments or {}
        if not isinstance(parsed, dict):
            raise ValueError("Tool arguments must be a JSON object")

        # The meta tools take a `headers` object in the envelope, and these arguments
        # are model-controlled. Without this, a prompt-injected call could put its own
        # Authorization or x-account-id in the envelope the server unpacks — the guard
        # the RPC path has had all along, on the path search()/execute() actually use.
        supplied_headers = parsed.get("headers")
        if isinstance(supplied_headers, dict):
            parsed = {**parsed, "headers": self._sanitise_headers(supplied_headers)}

        return call_mcp_tool(
            self._endpoint, self._mcp_headers, self.name, parsed, timeout=self._execute_config.timeout
        )


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

    def get_connectors(self) -> set[str]:
        """Get unique connector names from all tools (lowercase)"""
        return {tool.connector for tool in self.tools}

    def to_openai(self) -> list[JsonDict]:
        """Convert all tools to OpenAI function format"""
        return [tool.to_openai_function() for tool in self.tools]

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
