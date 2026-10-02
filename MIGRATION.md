# Migrating from 2.x to 3.0

3.0 is a thin client over StackOne's MCP endpoint. Every tool call goes over MCP
`tools/call`, and the only other request the SDK makes is `GET /accounts`, to find
your linked accounts. Search runs on the server. Anything MCP does not support has
been removed rather than kept on a second transport.

Work through the sections that apply to you. Each one gives the 2.x code and its 3.0
replacement.

- [Installation](#installation)
- [Imports](#imports)
- [Search and execute](#search-and-execute)
- [Results](#results)
- [Tool arguments and headers](#tool-arguments-and-headers)
- [Base URL](#base-url)
- [Accounts](#accounts)
- [Feedback](#feedback)
- [Schemas given to a model](#schemas-given-to-a-model)
- [Hand-built tools and `ExecuteConfig`](#hand-built-tools-and-executeconfig)
- [LangGraph helpers](#langgraph-helpers)

## Installation

- **Python 3.11 or later** is required. 3.10 is no longer supported.
- **`mcp` is a core dependency.** The `stackone-ai[mcp]` extra no longer exists.
- **`langchain-core` is now optional.** `to_langchain()` and `toolset.langchain()`
  need the `langchain` extra and raise an `ImportError` that says so without it.
- `bm25s` and `numpy` are no longer dependencies, because search now runs on the server.

```bash
# 2.x
uv add 'stackone-ai[mcp]'

# 3.0: add the extras for the frameworks you use
uv add stackone-ai
uv add 'stackone-ai[langchain]'    # LangChain and LangGraph
uv add 'stackone-ai[pydantic-ai]'  # Pydantic AI
```

## Imports

`stackone_ai.models` is split into `stackone_ai.tools` and `stackone_ai.types`. Anything
public can also be imported from the package root.

```python
# 2.x
from stackone_ai.models import ExecuteConfig, StackOneTool, ToolParameters, Tools
from stackone_ai.toolset import ToolsetConfigError

# 3.0
from stackone_ai import ExecuteConfig, StackOneTool, ToolParameters, Tools, ToolsetConfigError
```

`ToolsetError`, `ToolsetConfigError` and `ToolsetLoadError` now subclass `StackOneError`,
so `except StackOneError` catches everything the SDK raises. Existing
`except ToolsetError` clauses behave as before.

Unusable tool arguments raise the new `ToolArgumentsError`, before any request: a JSON
string that does not parse or does not parse to an object, an argument that cannot be
encoded as JSON, or `tool.call()` given both positional and keyword arguments, or more
than one positional argument. Raised by `tool.execute()`, `tool.call()` and
`toolset.execute()` alike. It subclasses both `StackOneError` and `ValueError`, which is
what 2.x raised, so existing `except ValueError` clauses still catch it.

These names have been removed and have no direct replacement:

| Removed | Use instead |
|---|---|
| `SearchConfig`, `SearchMode`, `SearchTool`, `SemanticSearchClient`, `SemanticSearchResult`, `SemanticSearchResponse`, `SemanticSearchError` | `toolset.search()`. See [Search and execute](#search-and-execute) |
| `stackone_ai.feedback`, `create_feedback_tool` | `toolset.submit_feedback()`. See [Feedback](#feedback) |
| `stackone_ai.integrations` | LangGraph's own APIs. See [LangGraph helpers](#langgraph-helpers) |
| `ToolDefinition` | `StackOneTool` fields: `name`, `description`, `parameters` |
| `ParameterLocation`, `validate_method` | Nothing. `ExecuteConfig` no longer has a method or parameter locations |
| `StackOneTool.connector`, `Tools.get_connectors()` | `fetch_tools(providers=[...])` to filter by provider. A provider name can contain `_` (`browser_linkedin`), so do not split the tool name |

Some names were public on `main` for a while before 3.0 but never in a release:
`StackOneRpcTool`, `MCP_PARAM_STYLE`, `is_json_content_type` and
`filename_from_content_disposition`. They are gone as well.

## Search and execute

Client-side search is gone: the BM25/TF-IDF index, the semantic search client, the
`search=` constructor argument, and the `tool_search`/`tool_execute` meta tools. The
server's `*_search_actions` tool now does the search, and `toolset.execute()` runs an
action by id through the server's `*_execute_action` tool.

```python
# 2.x
toolset = StackOneToolSet(search={"method": "auto"})
tools = toolset.search_tools("list employees", top_k=5)
result = tools[0].execute({"query_page_size": 25})

names = toolset.search_action_names("time off requests", top_k=5)

# 3.0
toolset = StackOneToolSet()
hits = toolset.search("list employees", top_k=5)   # a list of action dicts, best first
hit = hits[0]
result = toolset.execute(
    hit["action_id"],
    {"query": {"page_size": 25}},     # the nested form in hit["input_schema"]
    account_ids=[hit["account_id"]],
    session_id=hit.get("session_id"),
)
```

`search()` returns dicts carrying `action_id` and the `account_id` that found it, plus
`description`, `similarity_score`, `input_schema`, `example_request` and `session_id`
when the server sends them. `top_k` caps the whole result, as in 2.x, after ranking
across every connector; the same action linked on two accounts is two hits. When an
action's connector is linked on more than one account, `execute()` raises
`ToolsetConfigError` unless `account_ids` picks one, such as `[hit["account_id"]]`.

`toolset.execute()` has a new signature. It used to take a meta tool name, such as
`"tool_execute"`, with arguments as a JSON string or a dict. It now takes
`(action_id, arguments=None, *, account_ids=None, session_id=None)`, and `arguments`
must be a dict. It raises instead of returning `{"error": ...}`: `ToolsetConfigError`
before any request when `action_id`, `arguments` or `session_id` is malformed,
`ToolArgumentsError` when the arguments cannot be encoded as JSON, `ToolsetLoadError` when
no linked connector matches the action, and `StackOneAPIError` when the action fails,
including when the server rejects the arguments.

To give a model the search and execute tools, set the tool mode on the toolset.
`openai()`, `langchain()` and `pydantic_ai()` no longer take `mode=`.

```python
# 2.x
toolset = StackOneToolSet(search={"method": "auto"})
openai_tools = toolset.openai(mode="search_and_execute")
# ...for each tool call the model makes:
toolset.execute(call.function.name, call.function.arguments)

# 3.0: two tools per connector, served by the server
toolset = StackOneToolSet(tool_mode="search_execute")
tools = toolset.fetch_tools()
openai_tools = tools.to_openai()
# ...then run the model's tool calls and get the messages to send back:
messages.extend(tools.execute_openai_tool_calls(message.tool_calls))
```

`get_search_tool()` and `semantic_client` are removed. Call `toolset.search()` instead.

## Results

**Every tool returns the server's result as the server wrote it.** For an action,
that is `{"isError": False, "result": ..., "defenderMetadata"?, "policyMetadata"?}`.
Search results are bare JSON. This applies to `tool.execute()`, `tool.call()` and
`toolset.execute()`.

```python
tool = toolset.fetch_tools(account_ids=[account_id]).get_tool("hibob_list_employees")

# 2.x
employees = tool.execute({})["data"]

# 3.0
employees = tool.execute({})["result"]["data"]
```

A result with `isError` set raises `StackOneAPIError`, which carries the status from the
payload in `status_code` and the body in `response_body`.

**Integers above 2^53 keep their exact value.** This SDK parses the server's JSON
itself, so it is not limited to the integer precision of IEEE 754 doubles; the Node
SDK's result can differ for a field that size, because its JSON parser has already
rounded it by the time the SDK sees it.

**File actions return a download link, not bytes.** The SDK does not follow the link.
When no link can be issued, the call raises `StackOneAPIError` with `status_code` 501.

```python
# 2.x
result = download.execute({"id": "file-id"})
with open(result["file_name"] or "download.bin", "wb") as f:
    f.write(result["content"])

# 3.0
from pathlib import Path

import httpx

link = download.execute({"id": "file-id"})["result"]
# {"download_url": ..., "expires_at": ..., "file": {"name", "content_type", "content_length"}}
name = Path(link["file"]["name"] or "download.bin").name   # chosen by the provider: keep only the basename
Path(name).write_bytes(httpx.get(link["download_url"]).content)
```

## Tool arguments and headers

**Arguments are sent exactly as given**, as `tools/call` arguments. The SDK no longer
splits flat `path_`/`query_`/`body_` keys into a request envelope, because the server
maps them itself.

**`fetch_tools()` tools have the server's own argument shape.** In 2.x the SDK asked
the server for the flat, prefixed style (`?param-style=flat_prefixed`). That request is
gone, so the argument names are whatever the server serves. Read them from
`tool.parameters.properties` rather than hard-coding them:

```python
# 2.x
tool.execute({"body_variables": {"first": 25}})

# 3.0: use the keys the schema names, for example
print(tool.parameters.properties)
tool.execute({"body": {"variables": {"first": 25}}})
```

**Header arguments are allowlisted.** A header argument is an entry of a `headers`
object argument, or a top-level `headers_<name>` argument. Each one is forwarded only
if the tool's schema declares it in the same form: under `headers.properties`, or as a
`headers_<name>` property. An open `headers` object, `"type": "object"` with no
`properties` and `additionalProperties` not `false`, declares every name; without
`"type": "object"`, or with `additionalProperties: false` and no `properties`, it
declares none. `Authorization`, `x-account-id` and `User-Agent` are never forwarded,
even when declared, because the SDK sets them itself. A dropped header argument is
logged as a warning, except a null value, which is omitted silently. Every other
argument is sent unchanged.

A top-level `headers` argument that isn't a plain object is dropped with a warning,
unless the schema declares `headers` itself as a non-object field, in which case it's
an ordinary argument that happens to be named `headers`: it skips header filtering and is
sent as given, subject to the same JSON-value check as every other argument. A
`headers_<name>` argument is dropped with a warning when its value is a list or dict.

`*_execute_action` serves an open `headers` object, so `toolset.execute()` passes your
own headers on to the action, with the exception of those three:

```python
toolset.execute("linear_list_comments", {"headers": {"x-request-id": "abc"}})
```

## Base URL

**`STACKONE_BASE_URL` is now read.** `base_url` still takes precedence when given;
otherwise the SDK now falls back to `STACKONE_BASE_URL` before the default,
`https://api.stackone.com`. An empty `base_url` argument or an empty
`STACKONE_BASE_URL` is treated as not set, not as a literal empty host.

## Accounts

**With no account id, the SDK discovers your accounts.** In 2.x, calling
`fetch_tools()` with no account listed tools without an `x-account-id`. In 3.0 it asks
`GET /accounts` and lists the catalog of every active account. If you have many
accounts, pass `account_id=`, `account_ids=` or call `set_accounts()` so the SDK does
not fetch every catalog.

**`STACKONE_ACCOUNT_ID` is not read.** Pass the account id as `account_id=` or
`execute={"account_ids": [...]}`. A toolset constructed with neither while
`STACKONE_ACCOUNT_ID` is set to a non-empty value logs a warning that it is ignored.

**An empty account id raises.** `StackOneToolSet(account_id="")` raises
`ToolsetConfigError`, where it used to be treated as no account at all. An account id
read from an environment variable that is set but empty now raises too, rather than
quietly reaching every active account:

```python
# Raises when STACKONE_ACCOUNT_ID is set to ""
toolset = StackOneToolSet(account_id=os.getenv("STACKONE_ACCOUNT_ID"))

# Falls back to discovery only when the variable is unset or empty, deliberately
toolset = StackOneToolSet(account_id=os.getenv("STACKONE_ACCOUNT_ID") or None)
```

An empty or non-string entry in `account_ids`, `set_accounts()` or
`execute={"account_ids": [...]}` raises `ToolsetConfigError` as well, and so does
passing a tuple (or any other non-list) there: pass a list, `list(ids)`, instead. A falsy entry
(`None`, `0`, `""`) used to produce an unscoped request with no `x-account-id`; a
truthy non-string entry used to be sent as the header's value.

**`get_tool()` returns the first of any duplicates.** When two accounts serve the same
tool name, `Tools.get_tool()` now returns the first one listed, where 2.x returned the
last. Listings are merged in sorted account order, and a warning names the clashing
tools. `execute_openai_tool_calls()`, `to_openai()`, `to_langchain()` and
`to_pydantic_ai()` use the same one: each builds one tool per name. Pass `account_ids`
to choose the account yourself.

**A failing account is skipped, and left out for 30 seconds.** In a multi-account scope, an
account whose listing fails with a non-429 error is skipped with a warning and left out of
the cached catalog for 30 seconds, unless every account fails: then the accounts' shared
`StackOneAPIError` is raised when they all failed with one status, and otherwise a
`ToolsetLoadError` with each account's error in `failures`. A 429 that outlasts its retries
is never skipped — it aborts the whole call, even when other accounts are healthy, since
it is the key's rate limit rather than that account's failure. A single-account scope
raises its failure rather than skipping or caching it.

## Feedback

The client-side `tool_feedback` tool, `create_feedback_tool()` and the implicit
`feedback_*` execution options have been removed. `execute()` and `call()` no longer
take `options=`.

```python
# 2.x
feedback_tool = toolset.fetch_tools(actions=["tool_*"]).get_tool("tool_feedback")
feedback_tool.call(feedback="Worked well", account_id="acc_123", tool_names=["workday_list_workers"])
tool.execute({"id": "1"}, options={"feedback_session_id": "chat-42"})

# 3.0
hit = toolset.search("list workers")[0]
toolset.execute(hit["action_id"], account_ids=[hit["account_id"]], session_id=hit.get("session_id"))
toolset.submit_feedback(
    "positive",
    [hit["action_id"]],
    feedback="Worked well",
    session_id=hit.get("session_id"),
)
```

When feedback is enabled for your project, the server also serves a
`stackone_submit_feedback` tool. **`fetch_tools()`, `openai()`, `langchain()` and
`pydantic_ai()` include it**, once, however many accounts are linked. If you don't want
a model to call it, filter it out:

```python
from stackone_ai import Tools

tools = Tools([t for t in toolset.fetch_tools() if t.name != "stackone_submit_feedback"])
```

`submit_feedback()` raises `ToolsetLoadError` when feedback is not enabled. It calls the
tool once, on the lowest account id in scope, and sends `action_run_id` when given.

## Schemas given to a model

**`to_openai_function()` passes the served schema through.** 2.x kept only `type`,
`description`, `enum` and a shallow copy of `items`/`properties`. Now `format`,
`pattern`, `default`, bounds, `oneOf`/`anyOf` and nested `required` reach the model as
the server served them. `to_langchain()` and `to_pydantic_ai_tool()` hand over the
same schema.

**`required` is the served list.** 2.x derived `required` from each property's
internal `nullable` marker, which made every property of a hand-built tool with no
markers required. 3.0 emits the schema's own `required` list, in the order the server
sent it, and leaves it out when that list is missing or empty. Give a hand-built tool
its `required` list directly:

```python
# 2.x: required, because "nullable" is absent
ToolParameters(type="object", properties={"id": {"type": "string"}})

# 3.0
ToolParameters(type="object", properties={"id": {"type": "string"}}, required=["id"])
```

## Hand-built tools and `ExecuteConfig`

**`StackOneTool` no longer makes HTTP requests.** Its `execute()` raises `StackOneError`
on the base class, so a hand-built tool must override it. Tools from `fetch_tools()` are
`StackOneMcpTool` instances, which execute over MCP. `_api_key` is now optional on the
base class.

```python
# 2.x
tool = StackOneTool(
    description="Get an employee",
    parameters=ToolParameters(type="object", properties={"id": {"type": "string"}}),
    _execute_config=ExecuteConfig(
        name="get_employee", method="GET", url="https://api.example.com/employees/{id}",
        parameter_locations={"id": ParameterLocation.PATH},
    ),
    _api_key="...",
)

# 3.0
class GetEmployee(StackOneTool):
    def __init__(self) -> None:
        super().__init__(
            description="Get an employee",
            parameters=ToolParameters(type="object", properties={"id": {"type": "string"}}, required=["id"]),
            _execute_config=ExecuteConfig(name="get_employee"),
        )

    def execute(self, arguments=None):
        args = json.loads(arguments) if isinstance(arguments, str) else dict(arguments or {})
        return httpx.get(f"https://api.example.com/employees/{args['id']}").json()

tool = GetEmployee()
```

Define `__init__` on the subclass, as above. `StackOneTool` is a pydantic model, so a
type checker gives a subclass without its own `__init__` one built from the model's
fields, which rejects `_execute_config` even though the call works at runtime.

**`ExecuteConfig` keeps only `name`, `headers` and `timeout`**, and rejects any other
field. Passing `method`, `url`, `body_type` or `parameter_locations` raises a pydantic
`ValidationError` rather than being silently ignored.

**`StackOneMcpTool(headers=...)` adds headers; it does not replace them.** The SDK
always sets `Authorization`, `x-account-id` and `User-Agent` itself, after your headers,
and drops any case variant of those names you pass. `headers` is optional.

## LangGraph helpers

`stackone_ai.integrations` has been removed. Pass the LangChain tools to LangGraph
directly:

```python
# 2.x
from stackone_ai.integrations import to_tool_node
node = to_tool_node(tools)

# 3.0
from langgraph.prebuilt import ToolNode
node = ToolNode(tools.to_langchain())
```

See [examples/langgraph_integration.py](examples/langgraph_integration.py) for a
complete agent.
