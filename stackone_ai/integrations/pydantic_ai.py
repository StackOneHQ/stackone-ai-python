"""Pydantic AI integration helpers.

Use ``StackOneToolset`` as a drop-in Pydantic AI toolset, or wrap a single
tool with ``tool_from_stackone``:

    from pydantic_ai import Agent
    from stackone_ai.integrations.pydantic_ai import StackOneToolset

    toolset = StackOneToolset(tools=["workday_list_workers"])
    agent = Agent("openai:gpt-5.4", toolsets=[toolset])

Requires ``pydantic-ai-slim>=1.83`` — install via ``stackone-ai[pydantic-ai]``.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Literal

from stackone_ai.toolset import ExecuteToolsConfig, SearchConfig, StackOneToolSet

try:
    from pydantic_ai.tools import Tool
    from pydantic_ai.toolsets.function import FunctionToolset
except ImportError as _import_error:
    raise ImportError(
        "Install `pydantic-ai-slim` (or `stackone-ai[pydantic-ai]`) to use the Pydantic AI integration."
    ) from _import_error

if TYPE_CHECKING:
    from stackone_ai.models import StackOneTool

__all__ = ("StackOneToolset", "tool_from_stackone")


def _resolve_account_ids(
    account_ids: str | list[str] | None,
    *,
    required: bool = True,
) -> list[str]:
    """Return account IDs from the explicit arg or ``STACKONE_ACCOUNT_ID`` env var.

    Accepts a single string, a list of strings, or ``None``. Raises ``ValueError``
    when nothing is provided and ``required=True``.
    """
    if isinstance(account_ids, str):
        return [account_ids]
    if account_ids is not None:
        return list(account_ids)
    env = os.getenv("STACKONE_ACCOUNT_ID")
    if env:
        return [env]
    if required:
        raise ValueError(
            "StackOne account ID(s) required. "
            "Pass `account_ids='acct-1'` or `account_ids=['acct-1', 'acct-2']`, "
            "or set the `STACKONE_ACCOUNT_ID` environment variable."
        )
    return []


def _tool_from_stackone_tool(stackone_tool: StackOneTool) -> Tool:
    openai_function = stackone_tool.to_openai_function()
    json_schema = openai_function["function"]["parameters"]

    def implementation(**kwargs: Any) -> Any:
        return stackone_tool.execute(kwargs)

    return Tool.from_schema(
        function=implementation,
        name=stackone_tool.name,
        description=stackone_tool.description or "",
        json_schema=json_schema,
    )


def tool_from_stackone(
    tool_name: str,
    *,
    account_ids: str | list[str] | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
) -> Tool:
    """Create a Pydantic AI tool proxy for a single StackOne tool.

    Args:
        tool_name: The StackOne tool to wrap (e.g. ``"workday_list_workers"``).
        account_ids: One or more connected-account IDs. Pass a string for a single
            account, a list for multiple. Falls back to ``STACKONE_ACCOUNT_ID``.
        api_key: StackOne API key. Falls back to ``STACKONE_API_KEY``.
        base_url: Custom StackOne API base URL.

    Returns:
        A Pydantic AI ``Tool`` that proxies calls through to StackOne.
    """
    resolved = _resolve_account_ids(account_ids)
    stackone_toolset = StackOneToolSet(api_key=api_key, base_url=base_url)
    stackone_toolset.set_accounts(resolved)
    tools = stackone_toolset.fetch_tools(actions=[tool_name])
    stackone_tool = tools.get_tool(tool_name)
    if stackone_tool is None:
        raise ValueError(f"Tool {tool_name!r} not found in StackOne")
    return _tool_from_stackone_tool(stackone_tool)


class StackOneToolset(FunctionToolset):
    """A Pydantic AI toolset backed by StackOne tools.

    Args:
        tools: Explicit list of StackOne tool names to load.
        account_ids: One or more connected-account IDs. Pass a string for one
            account, a list for multiple. Falls back to ``STACKONE_ACCOUNT_ID``.
        api_key: StackOne API key. Falls back to ``STACKONE_API_KEY``.
        base_url: Custom StackOne API base URL.
        filter_pattern: Glob pattern (or list of patterns) restricting which
            tools are loaded. Mutually exclusive with ``tools``.
        mode: Set to ``"search_and_execute"`` to expose only two meta-tools
            (``tool_search`` + ``tool_execute``) for on-demand discovery. Mutually
            exclusive with ``tools`` and ``filter_pattern``.
        search_config: Search strategy config. Requires ``mode="search_and_execute"``.
        execute_config: Execution config (e.g. ``{"account_ids": [...]}``).
            Requires ``mode="search_and_execute"``.
        id: Optional toolset identifier forwarded to ``FunctionToolset``.
    """

    def __init__(
        self,
        tools: Sequence[str] | None = None,
        *,
        account_ids: str | list[str] | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        filter_pattern: str | list[str] | None = None,
        mode: Literal["search_and_execute"] | None = None,
        search_config: SearchConfig | None = None,
        execute_config: ExecuteToolsConfig | None = None,
        id: str | None = None,
    ):
        if mode != "search_and_execute" and (search_config is not None or execute_config is not None):
            raise ValueError("'search_config' and 'execute_config' require mode='search_and_execute'")

        if mode == "search_and_execute":
            if tools is not None or filter_pattern is not None:
                raise ValueError("Cannot combine mode='search_and_execute' with 'tools' or 'filter_pattern'")
            has_execute_accounts = execute_config is not None and "account_ids" in execute_config
            if has_execute_accounts and account_ids is not None:
                raise ValueError("Cannot specify both 'account_ids' and 'execute_config[\"account_ids\"]'")
            resolved_search_config: SearchConfig = (
                search_config if search_config is not None else SearchConfig()
            )
            stackone_toolset = StackOneToolSet(
                api_key=api_key,
                base_url=base_url,
                search=resolved_search_config,
                execute=execute_config,
            )
            if not has_execute_accounts:
                stackone_toolset.set_accounts(_resolve_account_ids(account_ids))
            # `_build_tools` is SDK-internal; expose publicly in a later release.
            meta_tools = stackone_toolset._build_tools()
            pydantic_tools = [_tool_from_stackone_tool(t) for t in meta_tools]
            super().__init__(pydantic_tools, id=id)
            return

        if tools is not None and filter_pattern is not None:
            raise ValueError("Cannot specify both 'tools' and 'filter_pattern'")

        stackone_toolset = StackOneToolSet(api_key=api_key, base_url=base_url)
        stackone_toolset.set_accounts(_resolve_account_ids(account_ids))

        if tools is not None:
            actions = list(tools)
        elif isinstance(filter_pattern, str):
            actions = [filter_pattern]
        else:
            actions = filter_pattern

        fetched_tools = stackone_toolset.fetch_tools(actions=actions)
        pydantic_tools = [_tool_from_stackone_tool(tool) for tool in fetched_tools]
        super().__init__(pydantic_tools, id=id)
