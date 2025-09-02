"""LangGraph integration helpers.

These utilities convert StackOne tools into LangGraph prebuilt components.

Usage:
    from stackone_ai import StackOneToolSet
    from stackone_ai.integrations.langgraph import to_tool_node

    toolset = StackOneToolSet()
    tools = toolset.get_tools("hris_*", account_id="...")
    node = to_tool_node(tools)  # langgraph.prebuilt.ToolNode
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from langchain_core.tools import BaseTool

from stackone_ai.models import Tools

if TYPE_CHECKING:  # pragma: no cover - only for typing
    try:
        from langgraph.prebuilt import ToolExecutor, ToolNode
    except Exception:  # pragma: no cover
        ToolExecutor = Any
        ToolNode = Any


def _ensure_langgraph() -> None:
    try:
        from langgraph import prebuilt as _  # noqa: F401
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "LangGraph is not installed. Install with `pip install langgraph` or "
            "`pip install 'stackone-ai[examples]'`"
        ) from e


def _to_langchain_tools(tools: Tools | Sequence[BaseTool]) -> Sequence[BaseTool]:
    if isinstance(tools, Tools):
        return tools.to_langchain()
    return tools


def to_tool_node(tools: Tools | Sequence[BaseTool], **kwargs: Any) -> Any:
    """Create a LangGraph `ToolNode` from StackOne tools or LangChain tools.

    Accepts either a `Tools` collection from this SDK or an existing sequence of
    LangChain `BaseTool` instances and returns a LangGraph `ToolNode` suitable
    for inclusion in a graph.
    """
    _ensure_langgraph()
    from langgraph.prebuilt import ToolNode  # local import with helpful error

    langchain_tools = _to_langchain_tools(tools)
    return ToolNode(langchain_tools, **kwargs)


def to_tool_executor(tools: Tools | Sequence[BaseTool], **kwargs: Any) -> Any:
    """Create a LangGraph `ToolExecutor` from StackOne tools or LangChain tools."""
    _ensure_langgraph()
    from langgraph.prebuilt import ToolExecutor  # local import with helpful error

    langchain_tools = _to_langchain_tools(tools)
    return ToolExecutor(langchain_tools, **kwargs)


def bind_model_with_tools(model: Any, tools: Tools | Sequence[BaseTool]) -> Any:
    """Bind tools to an LLM that supports LangChain's `.bind_tools()` API.

    This is a tiny helper that converts a `Tools` collection to LangChain tools
    and calls `model.bind_tools(...)`.
    """
    langchain_tools = _to_langchain_tools(tools)
    return model.bind_tools(langchain_tools)


def create_react_agent(llm: Any, tools: Tools | Sequence[BaseTool], **kwargs: Any) -> Any:
    """Create a LangGraph ReAct agent using StackOne tools.

    Thin wrapper around `langgraph.prebuilt.create_react_agent` that accepts a
    `Tools` collection from this SDK.
    """
    _ensure_langgraph()
    from langgraph.prebuilt import create_react_agent as _create

    return _create(llm, _to_langchain_tools(tools), **kwargs)
