# Changelog

## [3.0.0](https://github.com/StackOneHQ/stackone-ai-python/compare/stackone-ai-v2.10.1...stackone-ai-v3.0.0) (2026-10-09)


### ⚠ BREAKING CHANGES

* StackOneRpcTool, MCP_PARAM_STYLE, ParameterLocation, validate_method, is_json_content_type and filename_from_content_disposition are removed. Every tool fetch_tools() returns is a StackOneMcpTool that executes over MCP tools/call on the endpoint and account that listed it, with its arguments sent unchanged; nothing posts to /actions/rpc.
* **tools:** StackOneTool.execute() no longer makes HTTP requests. On the base class it raises StackOneError, so a hand-built tool must override execute().
* **types:** ExecuteConfig is extra="forbid" and keeps only name, headers and timeout. Passing method, url, body_type or parameter_locations raises a pydantic ValidationError.
* **toolset:** StackOneToolSet.execute() and tool.execute() return the server's {isError, result, defenderMetadata?, policyMetadata?} object rather than the unwrapped payload or the /actions/rpc body, so result["data"] becomes result["result"]["data"].
* **tools:** a file action returns {download_url, expires_at, file} as `result` instead of {content, content_type, status_code, headers, file_name}, and raises StackOneAPIError with status 501 when no link can be issued. Tools.execute_openai_tool_calls() no longer base64-encodes bytes.
* **toolset:** the /mcp URL no longer carries ?param-style=flat_prefixed, so per-action tools from fetch_tools() take whichever argument style the server serves by default instead of flat, prefixed keys such as body_variables.
* **tools:** StackOneMcpTool(headers=...) is now optional and means extra request headers, not the full header set. The SDK always sets Authorization, x-account-id and User-Agent itself, after them.
* **tools:** to_openai_function() no longer derives `required` from the per-property nullable markers. It emits the served root `required` list verbatim, in the served order, and omits it when absent or empty, so a hand-built ToolParameters without `required` now has no required fields.
* **toolset:** fetch_tools(), openai(), langchain() and pydantic_ai() include the stackone_submit_feedback tool, once, whenever the server serves it.
* **tools:** a top-level `headers_<name>` argument is dropped unless the tool's schema declares that property, and always for Authorization, x-account-id and User-Agent. A declared flat `headers_<name>` property no longer admits the same name inside a nested `headers` object. An open `headers` object, as `*_execute_action` serves it, forwards any other header, so StackOneToolSet.execute(..., {"headers": {...}}) passes host-set headers through where it previously dropped them all. A top-level `headers` argument that isn't a plain object is dropped too, unless the schema declares `headers` itself as a non-object field, in which case it is an ordinary argument sent as given. A declared flat `headers_<name>` argument whose value is a list or dict is dropped as well; only a string, number or boolean can be a header value.
* **tools:** when more than one account serves a tool name, Tools.get_tool() returns the first one listed (listings merge in sorted account order) rather than the last.
* Python 3.10 is no longer supported. stackone-ai requires Python 3.11 or later.
* the `stackone-ai[mcp]` extra no longer exists. mcp is a core dependency, so install plain `stackone-ai`.
* langchain-core is no longer installed with stackone-ai. StackOneTool.to_langchain(), Tools.to_langchain() and StackOneToolSet.langchain() need the `stackone-ai[langchain]` extra, which requires langchain-core 0.3.36 or later, and raise an ImportError that names it without it.
* stackone_ai.models, stackone_ai.constants and stackone_ai.utils no longer exist. Import StackOneTool and Tools from stackone_ai.tools; ToolParameters, ExecuteConfig, DEFAULT_BASE_URL and the errors from stackone_ai.types; or any public name from stackone_ai. ToolDefinition and DEFAULT_HYBRID_ALPHA are removed.
* **toolset:** client-side search is removed: SearchConfig, SearchMode, SearchTool, SemanticSearchClient, SemanticSearchResult, SemanticSearchResponse and SemanticSearchError, the `search=` constructor argument, search_tools(), search_action_names(), get_search_tool(), semantic_client, the tool_search and tool_execute meta tools, and the bm25s and numpy dependencies. StackOneToolSet.search() runs the server's *_search_actions tools instead.
* stackone_ai.integrations, with to_tool_node(), to_tool_executor(), bind_model_with_tools() and create_react_agent(), is removed. Pass tools.to_langchain() to LangGraph's own ToolNode and bind_tools().
* **tools:** the tool_feedback tool, stackone_ai.feedback, create_feedback_tool() and the implicit feedback_* execution options are removed, and StackOneTool.execute() and call() no longer take options=. Use StackOneToolSet.submit_feedback() and a search hit's session_id instead.
* **toolset:** StackOneToolSet.execute() takes (action_id, arguments=None, *, account_ids=None, session_id=None) instead of a meta tool name with arguments as a JSON string or dict. arguments must be a dict in the nested form an action's example_request shows, and a failure raises (ToolsetConfigError, ToolArgumentsError, ToolsetLoadError or StackOneAPIError) instead of returning {"error": ...}.
* **tools:** to_openai_function(), to_langchain() and to_pydantic_ai_tool() give the model the served JSON Schema instead of only type, description, enum and a shallow copy of items and properties, so format, pattern, default, bounds, oneOf/anyOf and nested required reach it. to_langchain() sets that schema as args_schema instead of a pydantic model built from it.
* **toolset:** openai(), langchain() and pydantic_ai() no longer take mode=. Set tool_mode="search_execute" on StackOneToolSet to give a model the server's search and execute tools.
* **toolset:** StackOneToolSet(account_id="") raises ToolsetConfigError instead of being treated as no account, which now lists every active account, so an account id read from an environment variable that is set but empty raises. An empty or non-string entry in account_ids, set_accounts() or execute={"account_ids": [...]} raises too, instead of being sent with no x-account-id.
* **toolset:** account_ids, set_accounts() and execute={"account_ids": ...} accept only a list of strings. A tuple or any other non-list, which 2.10.1 accepted, raises ToolsetConfigError; pass list(ids) instead.
* removed StackOneTool.connector property and Tools.get_connectors().

### Features

* export the feedback types and StackOneMcpTool, and take any sequence of tool names ([049abf5](https://github.com/StackOneHQ/stackone-ai-python/commit/049abf548c7f5448a89e7999e95336980d23afc3))
* retry HTTP 429 on every request, honouring Retry-After, and raise a 429 that outlasts the retries or the timeout instead of skipping the account ([049abf5](https://github.com/StackOneHQ/stackone-ai-python/commit/049abf548c7f5448a89e7999e95336980d23afc3))
* **toolset:** add a headers option sent on every request ([049abf5](https://github.com/StackOneHQ/stackone-ai-python/commit/049abf548c7f5448a89e7999e95336980d23afc3))
* **toolset:** add action_run_id to submit_feedback(), and send feedback through the lowest account id ([049abf5](https://github.com/StackOneHQ/stackone-ai-python/commit/049abf548c7f5448a89e7999e95336980d23afc3))
* **toolset:** link calls with session_id and add submit_feedback() ([049abf5](https://github.com/StackOneHQ/stackone-ai-python/commit/049abf548c7f5448a89e7999e95336980d23afc3))
* **toolset:** read STACKONE_BASE_URL ([049abf5](https://github.com/StackOneHQ/stackone-ai-python/commit/049abf548c7f5448a89e7999e95336980d23afc3))
* **toolset:** send x-end-user-id for non-shared accounts ([049abf5](https://github.com/StackOneHQ/stackone-ai-python/commit/049abf548c7f5448a89e7999e95336980d23afc3))
* **toolset:** skip non-shared accounts during discovery unless include_non_shared ([049abf5](https://github.com/StackOneHQ/stackone-ai-python/commit/049abf548c7f5448a89e7999e95336980d23afc3))
* **toolset:** tag each search hit with its account_id and apply top_k to the merged ranking ([049abf5](https://github.com/StackOneHQ/stackone-ai-python/commit/049abf548c7f5448a89e7999e95336980d23afc3))
* **toolset:** warn when STACKONE_ACCOUNT_ID is set but ignored ([049abf5](https://github.com/StackOneHQ/stackone-ai-python/commit/049abf548c7f5448a89e7999e95336980d23afc3))
* **tools:** raise ToolArgumentsError, a StackOneError and a ValueError, for unusable tool arguments ([049abf5](https://github.com/StackOneHQ/stackone-ai-python/commit/049abf548c7f5448a89e7999e95336980d23afc3))


### Bug Fixes

* **tools:** build every adapter from the first tool of each name ([049abf5](https://github.com/StackOneHQ/stackone-ai-python/commit/049abf548c7f5448a89e7999e95336980d23afc3))
* **tools:** emit the served required list in served order ([049abf5](https://github.com/StackOneHQ/stackone-ai-python/commit/049abf548c7f5448a89e7999e95336980d23afc3))
* **toolset:** accept account ids only as a list ([049abf5](https://github.com/StackOneHQ/stackone-ai-python/commit/049abf548c7f5448a89e7999e95336980d23afc3))
* **toolset:** cache the healthy accounts when one fails, and retry a failed one after 30 seconds ([049abf5](https://github.com/StackOneHQ/stackone-ai-python/commit/049abf548c7f5448a89e7999e95336980d23afc3))
* **toolset:** chain a failed end-user lookup onto the 400 it was looking up for ([049abf5](https://github.com/StackOneHQ/stackone-ai-python/commit/049abf548c7f5448a89e7999e95336980d23afc3))
* **toolset:** hold a provider lookup's miss for the failed-account window in execute() ([049abf5](https://github.com/StackOneHQ/stackone-ai-python/commit/049abf548c7f5448a89e7999e95336980d23afc3))
* **toolset:** learn a failed account's provider with one GET /accounts before execute() refuses ([049abf5](https://github.com/StackOneHQ/stackone-ai-python/commit/049abf548c7f5448a89e7999e95336980d23afc3))
* **toolset:** look up a non-shared account's end user when the API asks for it ([049abf5](https://github.com/StackOneHQ/stackone-ai-python/commit/049abf548c7f5448a89e7999e95336980d23afc3))
* **toolset:** move a pinned action_id and session_id to the end of the arguments ([049abf5](https://github.com/StackOneHQ/stackone-ai-python/commit/049abf548c7f5448a89e7999e95336980d23afc3))
* **toolset:** name shared accounts in the STACKONE_ACCOUNT_ID warning ([049abf5](https://github.com/StackOneHQ/stackone-ai-python/commit/049abf548c7f5448a89e7999e95336980d23afc3))
* **toolset:** raise the shared error when every account fails, and keep each failure ([049abf5](https://github.com/StackOneHQ/stackone-ai-python/commit/049abf548c7f5448a89e7999e95336980d23afc3))
* **toolset:** raise when every active account on the key is non-shared and not opted in ([049abf5](https://github.com/StackOneHQ/stackone-ai-python/commit/049abf548c7f5448a89e7999e95336980d23afc3))
* **toolset:** re-list only the failed accounts that could serve the action in execute() ([049abf5](https://github.com/StackOneHQ/stackone-ai-python/commit/049abf548c7f5448a89e7999e95336980d23afc3))
* **toolset:** refuse an action whose connector is linked on more than one account ([049abf5](https://github.com/StackOneHQ/stackone-ai-python/commit/049abf548c7f5448a89e7999e95336980d23afc3))
* **toolset:** refuse to execute while a matching account failed to list ([049abf5](https://github.com/StackOneHQ/stackone-ai-python/commit/049abf548c7f5448a89e7999e95336980d23afc3))
* **toolset:** reject an empty or non-string account id ([049abf5](https://github.com/StackOneHQ/stackone-ai-python/commit/049abf548c7f5448a89e7999e95336980d23afc3))
* **toolset:** route stackone_submit_feedback over MCP and list it once ([049abf5](https://github.com/StackOneHQ/stackone-ai-python/commit/049abf548c7f5448a89e7999e95336980d23afc3))
* **toolset:** send feedback once, through the first account ([049abf5](https://github.com/StackOneHQ/stackone-ai-python/commit/049abf548c7f5448a89e7999e95336980d23afc3))
* **tools:** guard every header argument by its own declaration ([049abf5](https://github.com/StackOneHQ/stackone-ai-python/commit/049abf548c7f5448a89e7999e95336980d23afc3))
* **tools:** keep a retried 429 on record while another request in the exchange is answered ([049abf5](https://github.com/StackOneHQ/stackone-ai-python/commit/049abf548c7f5448a89e7999e95336980d23afc3))
* **tools:** report an MCP timeout as a timeout ([049abf5](https://github.com/StackOneHQ/stackone-ai-python/commit/049abf548c7f5448a89e7999e95336980d23afc3))
* **tools:** return structuredContent from a tools/call result with no text ([049abf5](https://github.com/StackOneHQ/stackone-ai-python/commit/049abf548c7f5448a89e7999e95336980d23afc3))
* **tools:** return the first listing from get_tool(), as Node does ([049abf5](https://github.com/StackOneHQ/stackone-ai-python/commit/049abf548c7f5448a89e7999e95336980d23afc3))
* **tools:** send tools/call without relisting the catalog first ([049abf5](https://github.com/StackOneHQ/stackone-ai-python/commit/049abf548c7f5448a89e7999e95336980d23afc3))
* **tools:** write nested header values as Node writes them ([049abf5](https://github.com/StackOneHQ/stackone-ai-python/commit/049abf548c7f5448a89e7999e95336980d23afc3))
* **types:** declare ToolParameters.required so a hand-built tool type-checks ([049abf5](https://github.com/StackOneHQ/stackone-ai-python/commit/049abf548c7f5448a89e7999e95336980d23afc3))


### Code Refactoring

* execute every tool over MCP tools/call ([049abf5](https://github.com/StackOneHQ/stackone-ai-python/commit/049abf548c7f5448a89e7999e95336980d23afc3))
* MCP-only toolset with account discovery, search/execute, and hardened security ([#199](https://github.com/StackOneHQ/stackone-ai-python/issues/199)) ([e53edf0](https://github.com/StackOneHQ/stackone-ai-python/commit/e53edf0c7cc1da6cc5376df9ca27c6b845597e22))
* remove stackone_ai.integrations ([049abf5](https://github.com/StackOneHQ/stackone-ai-python/commit/049abf548c7f5448a89e7999e95336980d23afc3))
* split stackone_ai.models into stackone_ai.tools and stackone_ai.types ([049abf5](https://github.com/StackOneHQ/stackone-ai-python/commit/049abf548c7f5448a89e7999e95336980d23afc3))
* **toolset:** remove client-side search ([049abf5](https://github.com/StackOneHQ/stackone-ai-python/commit/049abf548c7f5448a89e7999e95336980d23afc3))
* **toolset:** remove mode= from openai(), langchain() and pydantic_ai() ([049abf5](https://github.com/StackOneHQ/stackone-ai-python/commit/049abf548c7f5448a89e7999e95336980d23afc3))
* **toolset:** return every result exactly as the server wrote it ([049abf5](https://github.com/StackOneHQ/stackone-ai-python/commit/049abf548c7f5448a89e7999e95336980d23afc3))
* **toolset:** run an action by id with execute() ([049abf5](https://github.com/StackOneHQ/stackone-ai-python/commit/049abf548c7f5448a89e7999e95336980d23afc3))
* **toolset:** stop pinning param-style=flat_prefixed on the MCP endpoint ([049abf5](https://github.com/StackOneHQ/stackone-ai-python/commit/049abf548c7f5448a89e7999e95336980d23afc3))
* **tools:** pass the served schema through to every adapter ([049abf5](https://github.com/StackOneHQ/stackone-ai-python/commit/049abf548c7f5448a89e7999e95336980d23afc3))
* **tools:** remove the client-side feedback tool ([049abf5](https://github.com/StackOneHQ/stackone-ai-python/commit/049abf548c7f5448a89e7999e95336980d23afc3))
* **tools:** remove the HTTP executor from the base StackOneTool ([049abf5](https://github.com/StackOneHQ/stackone-ai-python/commit/049abf548c7f5448a89e7999e95336980d23afc3))
* **tools:** return a download link from file actions ([049abf5](https://github.com/StackOneHQ/stackone-ai-python/commit/049abf548c7f5448a89e7999e95336980d23afc3))
* **tools:** treat StackOneMcpTool headers as extra headers ([049abf5](https://github.com/StackOneHQ/stackone-ai-python/commit/049abf548c7f5448a89e7999e95336980d23afc3))
* **types:** reduce ExecuteConfig to name, headers and timeout ([049abf5](https://github.com/StackOneHQ/stackone-ai-python/commit/049abf548c7f5448a89e7999e95336980d23afc3))


### Build System

* make langchain-core an optional dependency ([049abf5](https://github.com/StackOneHQ/stackone-ai-python/commit/049abf548c7f5448a89e7999e95336980d23afc3))
* make mcp a core dependency and remove the mcp extra ([049abf5](https://github.com/StackOneHQ/stackone-ai-python/commit/049abf548c7f5448a89e7999e95336980d23afc3))
* require Python 3.11 or later ([049abf5](https://github.com/StackOneHQ/stackone-ai-python/commit/049abf548c7f5448a89e7999e95336980d23afc3))

## [2.10.1](https://github.com/StackOneHQ/stackone-ai-python/compare/stackone-ai-v2.10.0...stackone-ai-v2.10.1) (2026-07-28)


### Bug Fixes

* **deps:** cap pydantic-ai in examples extra below 2.0 ([#195](https://github.com/StackOneHQ/stackone-ai-python/issues/195)) ([66f3f2b](https://github.com/StackOneHQ/stackone-ai-python/commit/66f3f2b8d2d41989852b850a0555c94c977e6afc))

## [2.10.0](https://github.com/StackOneHQ/stackone-ai-python/compare/stackone-ai-v2.9.1...stackone-ai-v2.10.0) (2026-07-27)


### Features

* **ENG-733:** support flat_prefixed MCP param style ([#192](https://github.com/StackOneHQ/stackone-ai-python/issues/192)) ([ff8fb9a](https://github.com/StackOneHQ/stackone-ai-python/commit/ff8fb9a41add55e9c31bd84e04447bb3685ab024))

## [2.9.1](https://github.com/StackOneHQ/stackone-ai-python/compare/stackone-ai-v2.9.0...stackone-ai-v2.9.1) (2026-06-03)


### Bug Fixes

* **models:** handle binary file downloads in tool execution ([#190](https://github.com/StackOneHQ/stackone-ai-python/issues/190)) ([8983c68](https://github.com/StackOneHQ/stackone-ai-python/commit/8983c68dd2fdbdbdd52af531a9a9f1de6d72e1b8))
* **search:** make ToolsetConfigError actionable and fix misleading docs ([#188](https://github.com/StackOneHQ/stackone-ai-python/issues/188)) ([d332c27](https://github.com/StackOneHQ/stackone-ai-python/commit/d332c27d2f7b827e7ca043f60c37cb0831a92361))

## [2.9.0](https://github.com/StackOneHQ/stackone-ai-python/compare/stackone-ai-v2.8.0...stackone-ai-v2.9.0) (2026-04-30)


### Features

* **integrations:** add native Pydantic AI support ([#184](https://github.com/StackOneHQ/stackone-ai-python/issues/184)) ([8216ab6](https://github.com/StackOneHQ/stackone-ai-python/commit/8216ab6c0776b0753cf3e68461bf5e8a9b044b06))

## [2.8.0](https://github.com/StackOneHQ/stackone-ai-python/compare/stackone-ai-v2.7.0...stackone-ai-v2.8.0) (2026-04-20)


### Features

* **examples:** streamline examples, standardize auth patterns and migrate examples to Workday ([#179](https://github.com/StackOneHQ/stackone-ai-python/issues/179)) ([0bdc939](https://github.com/StackOneHQ/stackone-ai-python/commit/0bdc939e2844a0dae6bf527c8c8209df0a90d1f7))

## [2.7.0](https://github.com/StackOneHQ/stackone-ai-python/compare/stackone-ai-v2.6.0...stackone-ai-v2.7.0) (2026-04-14)


### Features

* **search-optimization:** cache tool catalog and parallelize per-account MCP fetches ([#173](https://github.com/StackOneHQ/stackone-ai-python/issues/173)) ([cd635e6](https://github.com/StackOneHQ/stackone-ai-python/commit/cd635e65621e4e84da270a57e4e1453a3734ad95))

## [2.6.0](https://github.com/StackOneHQ/stackone-ai-python/compare/stackone-ai-v2.5.1...stackone-ai-v2.6.0) (2026-04-07)


### Features

* include available connectors in search/execute tool descriptions ([#165](https://github.com/StackOneHQ/stackone-ai-python/issues/165)) ([544f41e](https://github.com/StackOneHQ/stackone-ai-python/commit/544f41ef1340d11f58be1a34e627aa8e81f1102d))

## [2.5.1](https://github.com/StackOneHQ/stackone-ai-python/compare/stackone-ai-v2.5.0...stackone-ai-v2.5.1) (2026-03-26)


### Bug Fixes

* **search:** fall back to local search when semantic results don't match MCP tools ([#159](https://github.com/StackOneHQ/stackone-ai-python/issues/159)) ([2c86475](https://github.com/StackOneHQ/stackone-ai-python/commit/2c864759f43dc701d1bfa8407badf4a10f608332))

## [2.5.0](https://github.com/StackOneHQ/stackone-ai-python/compare/stackone-ai-v2.4.0...stackone-ai-v2.5.0) (2026-03-25)


### Features

* **search-tools:** LLM-driven search and execute and new API ([#151](https://github.com/StackOneHQ/stackone-ai-python/issues/151)) ([a5e5723](https://github.com/StackOneHQ/stackone-ai-python/commit/a5e5723689d702ce4d176194a0d6a43486bcdff7))

## [2.4.0](https://github.com/StackOneHQ/stackone-ai-python/compare/stackone-ai-v2.3.1...stackone-ai-v2.4.0) (2026-03-06)


### Features

* **search:** Semantic Tool Search ([#149](https://github.com/StackOneHQ/stackone-ai-python/issues/149)) ([ac76d1b](https://github.com/StackOneHQ/stackone-ai-python/commit/ac76d1b12734f81ad29c2e30cf968ff2f1a1326c))
* **skills:** add just-commands skill with dynamic context injection ([#133](https://github.com/StackOneHQ/stackone-ai-python/issues/133)) ([bf9f3fb](https://github.com/StackOneHQ/stackone-ai-python/commit/bf9f3fb46c4a6dbb55e76c83ceef0eb465b481fb))

## [2.3.1](https://github.com/StackOneHQ/stackone-ai-python/compare/stackone-ai-v2.3.0...stackone-ai-v2.3.1) (2026-01-29)


### Documentation

* fix Python version requirement to 3.10+ ([#131](https://github.com/StackOneHQ/stackone-ai-python/issues/131)) ([ef2b4e3](https://github.com/StackOneHQ/stackone-ai-python/commit/ef2b4e3d06290d4fdc638bc66eec0c48b177c4f0))

## [2.3.0](https://github.com/StackOneHQ/stackone-ai-python/compare/stackone-ai-v2.1.1...stackone-ai-v2.3.0) (2026-01-29)


### Bug Fixes

* **nix:** replace deprecated nixfmt-rfc-style with nixfmt ([#114](https://github.com/StackOneHQ/stackone-ai-python/issues/114)) ([10627b4](https://github.com/StackOneHQ/stackone-ai-python/commit/10627b441745806f3b57a7b1cdba296ef722b00f))


### Miscellaneous Chores

* trigger release 2.3.0 ([#130](https://github.com/StackOneHQ/stackone-ai-python/issues/130)) ([a28d0a6](https://github.com/StackOneHQ/stackone-ai-python/commit/a28d0a6fbcf703dd640d3255fa4171046ea225c7))

## [2.1.1](https://github.com/StackOneHQ/stackone-ai-python/compare/stackone-ai-v2.1.0...stackone-ai-v2.1.1) (2026-01-22)


### Bug Fixes

* **ci:** skip mock server install in release workflow [ENG-11910] ([#111](https://github.com/StackOneHQ/stackone-ai-python/issues/111)) ([377d766](https://github.com/StackOneHQ/stackone-ai-python/commit/377d766a276b444e84fee5af95f3d56db7e0b89b))

## [2.1.0](https://github.com/StackOneHQ/stackone-ai-python/compare/stackone-ai-v2.0.0...stackone-ai-v2.1.0) (2026-01-22)


### Features

* **nix:** integrate uv2nix for Python dependency management ([#88](https://github.com/StackOneHQ/stackone-ai-python/issues/88)) ([ee67062](https://github.com/StackOneHQ/stackone-ai-python/commit/ee67062d1a6628c9b549f2ad69c0c1d7bdde6d97))


### Bug Fixes

* **ci:** add submodules checkout to coverage job ([#85](https://github.com/StackOneHQ/stackone-ai-python/issues/85)) ([4cf907a](https://github.com/StackOneHQ/stackone-ai-python/commit/4cf907a49ac288f368ebbfa3a5dc59a0194bf54a))


### Documentation

* deepwiki badge ([#97](https://github.com/StackOneHQ/stackone-ai-python/issues/97)) ([d8b0234](https://github.com/StackOneHQ/stackone-ai-python/commit/d8b02346b6a77377b7c0356e636edcf2eac47096))
* **readme:** improve Nix development environment setup instructions ([#94](https://github.com/StackOneHQ/stackone-ai-python/issues/94)) ([2d6f6c2](https://github.com/StackOneHQ/stackone-ai-python/commit/2d6f6c224d1119a5a0254934ff542a0572f44c06))
* **readme:** reorganise installation section ([#72](https://github.com/StackOneHQ/stackone-ai-python/issues/72)) ([3cde479](https://github.com/StackOneHQ/stackone-ai-python/commit/3cde4794739e95479409396adc3b6e3b01eb3d33))
* **rules:** add nix-workflow rule ([#106](https://github.com/StackOneHQ/stackone-ai-python/issues/106)) ([b10c164](https://github.com/StackOneHQ/stackone-ai-python/commit/b10c164142ede3ce37b96a27b0e73452d6de50e6))

## [2.0.0](https://github.com/StackOneHQ/stackone-ai-python/compare/stackone-ai-v0.3.4...stackone-ai-v2.0.0) (2025-12-29)


### ⚠ BREAKING CHANGES

* Drop support for Python 3.9 and 3.10.
* Error handling now uses httpx exceptions instead of requests exceptions. Code catching RequestException should be updated to catch httpx.HTTPStatusError or httpx.RequestError.
* migrate examples and tests to connector-based tool naming ([#51](https://github.com/StackOneHQ/stackone-ai-python/issues/51))
* The `docs` optional dependency group and related commands (`make docs-serve`, `make docs-build`) are no longer available.
* remove MCP server implementation ([#45](https://github.com/StackOneHQ/stackone-ai-python/issues/45))
* remove deprecated OAS-based getTools, migrate to fetchTools only ([#42](https://github.com/StackOneHQ/stackone-ai-python/issues/42))

### Features

* add test coverage reporting with GitHub Pages badge ([#62](https://github.com/StackOneHQ/stackone-ai-python/issues/62)) ([0ef05cf](https://github.com/StackOneHQ/stackone-ai-python/commit/0ef05cf8746e3ca113e6fd6d33bc62fc91711f77))
* **security:** add gitleaks for secret detection ([#63](https://github.com/StackOneHQ/stackone-ai-python/issues/63)) ([1a31baa](https://github.com/StackOneHQ/stackone-ai-python/commit/1a31baa489882da9fc12684d1a060e48928288a9))


### Bug Fixes

* **ci:** use just commands in CI and release workflows ([#57](https://github.com/StackOneHQ/stackone-ai-python/issues/57)) ([38a9dd6](https://github.com/StackOneHQ/stackone-ai-python/commit/38a9dd6cc0b0deea53d1dfb6472686e151df53e4))
* migrate examples and tests to connector-based tool naming ([#51](https://github.com/StackOneHQ/stackone-ai-python/issues/51)) ([c365dbd](https://github.com/StackOneHQ/stackone-ai-python/commit/c365dbd98e8084eea45857292eee90a1798bce16))
* migrate HTTP client from requests to httpx ([#52](https://github.com/StackOneHQ/stackone-ai-python/issues/52)) ([9d180ef](https://github.com/StackOneHQ/stackone-ai-python/commit/9d180efb42647a573e3055ee47ea361dad2dec07))
* remove MCP server implementation ([#45](https://github.com/StackOneHQ/stackone-ai-python/issues/45)) ([bcb12b4](https://github.com/StackOneHQ/stackone-ai-python/commit/bcb12b4ee50e055c4cb29f3aa9baf81352683415))
* **scripts:** add uv lock refresh to version update script ([#50](https://github.com/StackOneHQ/stackone-ai-python/issues/50)) ([bde6d88](https://github.com/StackOneHQ/stackone-ai-python/commit/bde6d88a5688790ada366ed76563092aba0effe4))


### Documentation

* remove meta tools implementation details from README ([#40](https://github.com/StackOneHQ/stackone-ai-python/issues/40)) ([10510d4](https://github.com/StackOneHQ/stackone-ai-python/commit/10510d4b93fc4e20aa51a706541a649115900e6d))
* remove obsolete migration section from README ([#56](https://github.com/StackOneHQ/stackone-ai-python/issues/56)) ([bdcf90d](https://github.com/StackOneHQ/stackone-ai-python/commit/bdcf90d07cd29236f1372d53872689ef624f8e03))


### Miscellaneous Chores

* bump minimum Python version to 3.11 ([#81](https://github.com/StackOneHQ/stackone-ai-python/issues/81)) ([527e828](https://github.com/StackOneHQ/stackone-ai-python/commit/527e8284a73f47af741454610f71d462d095f79c))
* remove MkDocs documentation generation feature ([#46](https://github.com/StackOneHQ/stackone-ai-python/issues/46)) ([947863e](https://github.com/StackOneHQ/stackone-ai-python/commit/947863e91160a07fcd60d8ee837fb79a35abf0b0))
* trigger release 2.0.0 ([#82](https://github.com/StackOneHQ/stackone-ai-python/issues/82)) ([daa963b](https://github.com/StackOneHQ/stackone-ai-python/commit/daa963bacda5d79ad1bc7773d6432507c2ebfdb1))


### Code Refactoring

* remove deprecated OAS-based getTools, migrate to fetchTools only ([#42](https://github.com/StackOneHQ/stackone-ai-python/issues/42)) ([d50d5fb](https://github.com/StackOneHQ/stackone-ai-python/commit/d50d5fb20402dd625217b2900287ae7d9e4cb98c))

## [0.3.4](https://github.com/StackOneHQ/stackone-ai-python/compare/stackone-ai-v0.3.3...stackone-ai-v0.3.4) (2025-11-12)


### Features

* Add MCP-backed dynamic tool fetching to Python SDK ([#39](https://github.com/StackOneHQ/stackone-ai-python/issues/39)) ([d72ca80](https://github.com/StackOneHQ/stackone-ai-python/commit/d72ca808233600bd32374c7e2028232eb54167de))
* add provider/action filtering and hybrid BM25 + TF-IDF search ([#37](https://github.com/StackOneHQ/stackone-ai-python/issues/37)) ([a1c688b](https://github.com/StackOneHQ/stackone-ai-python/commit/a1c688b4efaef9257ecec9827baa7ef90529b9f7))

## [0.3.3](https://github.com/StackOneHQ/stackone-ai-python/compare/stackone-ai-v0.3.2...stackone-ai-v0.3.3) (2025-10-17)


### Features

* feedback tool ([#36](https://github.com/StackOneHQ/stackone-ai-python/issues/36)) ([9179918](https://github.com/StackOneHQ/stackone-ai-python/commit/9179918104c0ec4cfe0488713ca325f0e8e7c6f1))
* LangGraph integration helpers and example ([#33](https://github.com/StackOneHQ/stackone-ai-python/issues/33)) ([983e2f7](https://github.com/StackOneHQ/stackone-ai-python/commit/983e2f7e6551e3722f235ea534ae61f24644350e))


### Bug Fixes

* remove async method ([#31](https://github.com/StackOneHQ/stackone-ai-python/issues/31)) ([370699e](https://github.com/StackOneHQ/stackone-ai-python/commit/370699e390e4a46d8b4ae664fed8f5de6395eb9d))


### Documentation

* use uv for installing ([#30](https://github.com/StackOneHQ/stackone-ai-python/issues/30)) ([3c5d8fb](https://github.com/StackOneHQ/stackone-ai-python/commit/3c5d8fb54e61f8f730098e97f8bf2dfc78cf3bec))

## [0.3.2](https://github.com/StackOneHQ/stackone-ai-python/compare/stackone-ai-v0.3.1...stackone-ai-v0.3.2) (2025-08-26)


### Features

* support Python 3.9+ with optional MCP server support ([#28](https://github.com/StackOneHQ/stackone-ai-python/issues/28)) ([1a37776](https://github.com/StackOneHQ/stackone-ai-python/commit/1a377768c15223e25dbaf1e0bcd0c0e8bb0df2e8))

## [0.3.1](https://github.com/StackOneHQ/stackone-ai-python/compare/stackone-ai-v0.3.0...stackone-ai-v0.3.1) (2025-08-19)


### Documentation

* add comprehensive LangChain integration section to README ([#20](https://github.com/StackOneHQ/stackone-ai-python/issues/20)) ([cbf7f68](https://github.com/StackOneHQ/stackone-ai-python/commit/cbf7f68e889839f9a501ad8f2cd47c468ffff47e))
* rename meta tools ([#27](https://github.com/StackOneHQ/stackone-ai-python/issues/27)) ([a9ebc03](https://github.com/StackOneHQ/stackone-ai-python/commit/a9ebc032f784863913b28d4ad3850b80bafee5f4))

## [0.3.0](https://github.com/StackOneHQ/stackone-ai-python/compare/stackone-ai-v0.0.4...stackone-ai-v0.3.0) (2025-08-19)


### Features

* add CLAUDE.md for Claude Code guidance ([#15](https://github.com/StackOneHQ/stackone-ai-python/issues/15)) ([ac9fe98](https://github.com/StackOneHQ/stackone-ai-python/commit/ac9fe9857f44c19394654dfcbe23fecc5cf9fbb0))
* bring Python SDK to feature parity with Node SDK ([#17](https://github.com/StackOneHQ/stackone-ai-python/issues/17)) ([8b6de99](https://github.com/StackOneHQ/stackone-ai-python/commit/8b6de99184227cb7f1580964dc3eae14f8f60fc1))
* remove automatic STACKONE_ACCOUNT_ID environment variable loading ([#23](https://github.com/StackOneHQ/stackone-ai-python/issues/23)) ([aa0aaf6](https://github.com/StackOneHQ/stackone-ai-python/commit/aa0aaf6d6bf528f8e29def9b008db23cf94b97c7))
* simplify meta tool function names to match Node SDK ([#19](https://github.com/StackOneHQ/stackone-ai-python/issues/19)) ([4572609](https://github.com/StackOneHQ/stackone-ai-python/commit/4572609a9b85a88fc3067be12f821ec0bc54e769))


### Miscellaneous Chores

* release 0.3.0 ([#24](https://github.com/StackOneHQ/stackone-ai-python/issues/24)) ([beea911](https://github.com/StackOneHQ/stackone-ai-python/commit/beea91165ed2ba3eb5f5ad6ca8656344561b0b43))

## [0.0.4](https://github.com/StackOneHQ/stackone-ai-python/compare/stackone-ai-v0.0.3...stackone-ai-v0.0.4) (2025-03-05)


### Bug Fixes

* ci ([#14](https://github.com/StackOneHQ/stackone-ai-python/issues/14)) ([08aba6e](https://github.com/StackOneHQ/stackone-ai-python/commit/08aba6e96e55b4bedc7272e3adc91a1745d7859a))
* script dependencies ([#12](https://github.com/StackOneHQ/stackone-ai-python/issues/12)) ([960c5d8](https://github.com/StackOneHQ/stackone-ai-python/commit/960c5d86f33fcda8bae72d58166ee5991e08f4d5))

## [0.0.3](https://github.com/StackOneHQ/stackone-ai-python/compare/stackone-ai-v0.0.2...stackone-ai-v0.0.3) (2025-03-05)


### Features

* create operations and file upload.  ([#4](https://github.com/StackOneHQ/stackone-ai-python/issues/4)) ([c8469e3](https://github.com/StackOneHQ/stackone-ai-python/commit/c8469e3e0f7d7d35aee88edd0585a76411dcfba1))
* docs ([#3](https://github.com/StackOneHQ/stackone-ai-python/issues/3)) ([13575ea](https://github.com/StackOneHQ/stackone-ai-python/commit/13575eacede3c96ee3861611cdac6fca5663d7e9))
* docs site ([7a2984c](https://github.com/StackOneHQ/stackone-ai-python/commit/7a2984c33deb748abe3f282a449075631da80aef))
* langchain tools ([#2](https://github.com/StackOneHQ/stackone-ai-python/issues/2)) ([c2dc5aa](https://github.com/StackOneHQ/stackone-ai-python/commit/c2dc5aadda1104117c60703ccca6ceb63f8fd68d))
* licence ([#7](https://github.com/StackOneHQ/stackone-ai-python/issues/7)) ([feebc08](https://github.com/StackOneHQ/stackone-ai-python/commit/feebc08ee61f9e4569cbc44c4bac4d1060c036ef))
* openai compat tools ([64ac1da](https://github.com/StackOneHQ/stackone-ai-python/commit/64ac1da8f1d4fad090a1822d751e003a2cca2e52))
* release please ([2946dfb](https://github.com/StackOneHQ/stackone-ai-python/commit/2946dfbdaf2d27bdcfa49925c6aeaa59ea1a9a5e))


### Bug Fixes

* all extras ([235b5e3](https://github.com/StackOneHQ/stackone-ai-python/commit/235b5e32da6a0495d9ef082403ab2898c42a1976))
* Andres comments ([#9](https://github.com/StackOneHQ/stackone-ai-python/issues/9)) ([c97f0b7](https://github.com/StackOneHQ/stackone-ai-python/commit/c97f0b75959f556b94049fc2b65e51172339b718))
* ci ([6aae7fa](https://github.com/StackOneHQ/stackone-ai-python/commit/6aae7fafedebf48a86f6940c479463e1daf4bf93))
* clean up docs ([9186ec3](https://github.com/StackOneHQ/stackone-ai-python/commit/9186ec36937dd4d8cae1fe7367a686aeb01a0459))
* docs formatting ([#6](https://github.com/StackOneHQ/stackone-ai-python/issues/6)) ([2ac9a87](https://github.com/StackOneHQ/stackone-ai-python/commit/2ac9a8792bc630e60bed560102ad55e12f4cc7c5))
* type stubs in python packaging ([#11](https://github.com/StackOneHQ/stackone-ai-python/issues/11)) ([97c6ffe](https://github.com/StackOneHQ/stackone-ai-python/commit/97c6ffed7c6aaaef2834a503013805a6d31836d0))
