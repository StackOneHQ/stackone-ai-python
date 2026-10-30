"""Tests for the Pydantic AI integration."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("pydantic_ai", reason="pydantic-ai not installed")

from pydantic_ai import Agent  # noqa: E402
from pydantic_ai.messages import (  # noqa: E402
    ModelMessage,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel  # noqa: E402
from pydantic_ai.tools import Tool  # noqa: E402

from stackone_ai import StackOneToolSet  # noqa: E402
from stackone_ai.tools import StackOneTool, Tools  # noqa: E402
from stackone_ai.types import ExecuteConfig, ToolParameters  # noqa: E402


@pytest.fixture
def sample_tool() -> StackOneTool:
    return StackOneTool(
        description="List employees",
        parameters=ToolParameters(
            type="object",
            properties={"limit": {"type": "integer", "description": "Max results"}},
        ),
        _execute_config=ExecuteConfig(
            headers={},
            name="bamboohr_list_employees",
        ),
        _api_key="test_key",
    )


@pytest.fixture
def second_tool() -> StackOneTool:
    return StackOneTool(
        description="Get an employee",
        parameters=ToolParameters(type="object", properties={}),
        _execute_config=ExecuteConfig(
            headers={},
            name="bamboohr_get_employee",
        ),
        _api_key="test_key",
    )


# --- StackOneTool.to_pydantic_ai_tool() ---


def test_tool_to_pydantic_ai_tool_returns_tool(sample_tool: StackOneTool):
    tool = sample_tool.to_pydantic_ai_tool()
    assert isinstance(tool, Tool)
    assert tool.name == "bamboohr_list_employees"
    assert tool.description == "List employees"


def test_tool_to_pydantic_ai_tool_schema_matches_openai(sample_tool: StackOneTool):
    tool = sample_tool.to_pydantic_ai_tool()
    assert tool.function_schema.json_schema == sample_tool.to_openai_function()["function"]["parameters"]


def test_tool_to_pydantic_ai_tool_executes_through_stackone(
    sample_tool: StackOneTool, monkeypatch: pytest.MonkeyPatch
):
    captured: dict[str, Any] = {}

    def fake_execute(self: StackOneTool, arguments: dict[str, Any]) -> Any:
        captured["args"] = arguments
        return {"ok": True}

    monkeypatch.setattr(StackOneTool, "execute", fake_execute)

    tool = sample_tool.to_pydantic_ai_tool()
    result = tool.function(limit=10)  # type: ignore[operator]

    assert captured["args"] == {"limit": 10}
    assert result == {"ok": True}


# --- Tools.to_pydantic_ai() ---


def test_tools_to_pydantic_ai_converts_all(sample_tool: StackOneTool, second_tool: StackOneTool):
    collection = Tools([sample_tool, second_tool])
    tools = collection.to_pydantic_ai()

    assert len(tools) == 2
    assert [t.name for t in tools] == ["bamboohr_list_employees", "bamboohr_get_employee"]
    assert all(isinstance(t, Tool) for t in tools)


def test_tools_to_pydantic_ai_empty_collection():
    collection = Tools([])
    assert collection.to_pydantic_ai() == []


def test_tools_to_pydantic_ai_with_agent(sample_tool: StackOneTool, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(StackOneTool, "execute", lambda self, arguments: {"employee_ids": ["1", "2"]})
    tools = Tools([sample_tool]).to_pydantic_ai()

    agent = Agent("test", tools=tools)
    result = agent.run_sync("list employees")

    assert "bamboohr_list_employees" in result.output


# --- StackOneToolSet.pydantic_ai() ---


def test_toolset_pydantic_ai_default_uses_fetch_tools(sample_tool: StackOneTool):
    toolset = StackOneToolSet.__new__(StackOneToolSet)
    toolset.fetch_tools = MagicMock(return_value=Tools([sample_tool]))  # type: ignore[method-assign]
    toolset._execute_config = None  # type: ignore[attr-defined]

    tools = toolset.pydantic_ai()

    toolset.fetch_tools.assert_called_once_with(account_ids=None)
    assert len(tools) == 1
    assert tools[0].name == "bamboohr_list_employees"


def test_toolset_pydantic_ai_account_ids_override(sample_tool: StackOneTool):
    toolset = StackOneToolSet.__new__(StackOneToolSet)
    toolset.fetch_tools = MagicMock(return_value=Tools([sample_tool]))  # type: ignore[method-assign]
    toolset._execute_config = {"account_ids": ["from-config"]}  # type: ignore[attr-defined]

    toolset.pydantic_ai(account_ids=["override"])

    toolset.fetch_tools.assert_called_once_with(account_ids=["override"])


def test_execute_config_account_ids_scope_the_catalog_fetch(sample_tool: StackOneTool):
    """Account IDs from the execute config reach the MCP fetch when none are passed.

    Asserted through a real constructor rather than by mocking dispatch: the
    fallback lives in fetch_tools, so a test that bypasses __init__ would only
    be checking which method forwards the argument.
    """
    toolset = StackOneToolSet(api_key="test-key", execute={"account_ids": ["from-config"]})

    seen: list[str | None] = []

    def _capture(endpoint: str, headers: dict[str, str], **_kwargs: object):
        seen.append(headers.get("x-account-id"))
        return []

    with patch("stackone_ai.toolset.fetch_mcp_tools", side_effect=_capture):
        toolset.fetch_tools()

    assert seen == ["from-config"]


# --- An agent run against the mock MCP server ---


def _scripted_model(*steps: Callable[[ToolReturnPart | None], ToolCallPart]) -> FunctionModel:
    """A model that makes one tool call per step, each built from the previous call's return,
    then answers with the last return's content as text."""

    def respond(messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
        returns = [part for message in messages for part in message.parts if isinstance(part, ToolReturnPart)]
        if len(returns) < len(steps):
            return ModelResponse(parts=[steps[len(returns)](returns[-1] if returns else None)])
        return ModelResponse(parts=[TextPart(json.dumps(returns[-1].content))])

    return FunctionModel(respond)


def test_agent_runs_an_individual_tool_over_mcp(mcp_mock_server: str):
    toolset = StackOneToolSet(api_key="test-key", base_url=mcp_mock_server, account_id="test-account")
    model = _scripted_model(lambda _: ToolCallPart("dummy_action", {"foo": "bar"}))

    result = Agent(model, tools=toolset.pydantic_ai()).run_sync("run the dummy action")

    assert json.loads(result.output) == {
        "isError": False,
        "result": {"data": {"action": "dummy_action", "received": {"foo": "bar"}}},
    }


@pytest.mark.parametrize("account_ids", [["acc1"], ["acc1", "acc2"]], ids=["one-account", "two-accounts"])
def test_agent_searches_then_executes_in_search_execute_mode(mcp_mock_server: str, account_ids: list[str]):
    """The same script runs on one account's pair and on the trio over two."""
    toolset = StackOneToolSet(
        api_key="test-key",
        base_url=mcp_mock_server,
        execute={"account_ids": account_ids},
        tool_mode="search_execute",
    )
    tools = toolset.pydantic_ai()

    def execute_the_first_hit(found: ToolReturnPart | None) -> ToolCallPart:
        assert found is not None
        hit = found.content["actions"][0]
        return ToolCallPart(
            "stackone_execute_action",
            {
                "action_id": hit["action_id"],
                "stackone_account_id": hit["available_accounts"][0]["id"],
                **hit["example_request"],
            },
        )

    model = _scripted_model(
        lambda _: ToolCallPart("stackone_search_actions", {"query": "list items"}),
        execute_the_first_hit,
    )

    result = Agent(model, tools=tools).run_sync("list my items")

    assert {tool.name for tool in tools} >= {"stackone_search_actions", "stackone_execute_action"}
    assert json.loads(result.output) == {
        "isError": False,
        "result": {"data": {"nodes": []}, "echoed_query": {"page_size": 25}, "account_id": "acc1"},
    }


# --- Import error path ---


def test_to_pydantic_ai_tool_raises_helpful_import_error_when_package_missing(
    sample_tool: StackOneTool, monkeypatch: pytest.MonkeyPatch
):
    """If `pydantic_ai` is not importable, the helper surfaces a clean install hint."""
    import builtins

    real_import = builtins.__import__

    def fake_import(name: str, *args: Any, **kwargs: Any):
        if name == "pydantic_ai.tools" or name.startswith("pydantic_ai"):
            raise ImportError("No module named 'pydantic_ai'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(ImportError, match="stackone-ai\\[pydantic-ai\\]"):
        sample_tool.to_pydantic_ai_tool()
