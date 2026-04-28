"""Tests for the Pydantic AI integration."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

pytest.importorskip("pydantic_ai", reason="pydantic-ai not installed")

from pydantic_ai import Agent  # noqa: E402
from pydantic_ai.tools import Tool  # noqa: E402

from stackone_ai import StackOneToolSet  # noqa: E402
from stackone_ai.models import ExecuteConfig, StackOneTool, ToolParameters, Tools  # noqa: E402


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
            method="GET",
            url="https://api.example.com/employees",
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
            method="GET",
            url="https://api.example.com/employees/{id}",
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
    toolset._build_tools = MagicMock()  # type: ignore[method-assign]
    toolset._execute_config = None  # type: ignore[attr-defined]

    tools = toolset.pydantic_ai()

    toolset.fetch_tools.assert_called_once_with(account_ids=None)
    toolset._build_tools.assert_not_called()
    assert len(tools) == 1
    assert tools[0].name == "bamboohr_list_employees"


def test_toolset_pydantic_ai_search_and_execute_uses_build_tools(sample_tool: StackOneTool):
    toolset = StackOneToolSet.__new__(StackOneToolSet)
    toolset.fetch_tools = MagicMock()  # type: ignore[method-assign]
    toolset._build_tools = MagicMock(return_value=Tools([sample_tool]))  # type: ignore[method-assign]
    toolset._execute_config = None  # type: ignore[attr-defined]

    tools = toolset.pydantic_ai(mode="search_and_execute")

    toolset._build_tools.assert_called_once_with(account_ids=None)
    toolset.fetch_tools.assert_not_called()
    assert len(tools) == 1


def test_toolset_pydantic_ai_account_ids_override(sample_tool: StackOneTool):
    toolset = StackOneToolSet.__new__(StackOneToolSet)
    toolset.fetch_tools = MagicMock(return_value=Tools([sample_tool]))  # type: ignore[method-assign]
    toolset._build_tools = MagicMock()  # type: ignore[method-assign]
    toolset._execute_config = {"account_ids": ["from-config"]}  # type: ignore[attr-defined]

    toolset.pydantic_ai(account_ids=["override"])

    toolset.fetch_tools.assert_called_once_with(account_ids=["override"])


def test_toolset_pydantic_ai_falls_back_to_execute_config_account_ids(sample_tool: StackOneTool):
    toolset = StackOneToolSet.__new__(StackOneToolSet)
    toolset.fetch_tools = MagicMock(return_value=Tools([sample_tool]))  # type: ignore[method-assign]
    toolset._build_tools = MagicMock()  # type: ignore[method-assign]
    toolset._execute_config = {"account_ids": ["from-config"]}  # type: ignore[attr-defined]

    toolset.pydantic_ai()

    toolset.fetch_tools.assert_called_once_with(account_ids=["from-config"])


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
