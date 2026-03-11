"""Tests for meta tools (tool_search + tool_execute)."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

from stackone_ai.meta_tools import (
    ExecuteMetaTool,
    MetaToolsOptions,
    SearchMetaTool,
    create_meta_tools,
)
from stackone_ai.models import (
    ExecuteConfig,
    StackOneAPIError,
    StackOneTool,
    ToolParameters,
    Tools,
)


def _make_mock_tool(name: str = "test_tool", description: str = "A test tool") -> StackOneTool:
    return StackOneTool(
        description=description,
        parameters=ToolParameters(
            type="object",
            properties={
                "id": {"type": "string", "description": "The ID"},
                "count": {"type": "integer", "description": "A count"},
            },
        ),
        _execute_config=ExecuteConfig(
            name=name,
            method="GET",
            url="http://localhost/test/{id}",
        ),
        _api_key="test-key",
    )


def _make_mock_toolset(tools: list[StackOneTool] | None = None) -> MagicMock:
    toolset = MagicMock()
    toolset.api_key = "test-key"

    mock_tools = Tools(tools or [_make_mock_tool()])
    toolset.search_tools.return_value = mock_tools
    toolset.fetch_tools.return_value = mock_tools
    return toolset


class TestCreateMetaTools:
    def test_returns_tools_collection(self):
        toolset = _make_mock_toolset()
        result = create_meta_tools(toolset)

        assert isinstance(result, Tools)
        assert len(result) == 2

    def test_tool_names(self):
        toolset = _make_mock_toolset()
        result = create_meta_tools(toolset)

        names = [t.name for t in result]
        assert "tool_search" in names
        assert "tool_execute" in names

    def test_search_tool_type(self):
        toolset = _make_mock_toolset()
        result = create_meta_tools(toolset)
        search = result.get_tool("tool_search")
        assert isinstance(search, SearchMetaTool)

    def test_execute_tool_type(self):
        toolset = _make_mock_toolset()
        result = create_meta_tools(toolset)
        execute = result.get_tool("tool_execute")
        assert isinstance(execute, ExecuteMetaTool)

    def test_options_passed_through(self):
        toolset = _make_mock_toolset()
        opts = MetaToolsOptions(account_ids=["acc-1"], connector="bamboohr", top_k=3)
        result = create_meta_tools(toolset, opts)

        search = result.get_tool("tool_search")
        assert search._options.account_ids == ["acc-1"]
        assert search._options.connector == "bamboohr"
        assert search._options.top_k == 3

    def test_private_attrs_excluded_from_serialization(self):
        toolset = _make_mock_toolset()
        result = create_meta_tools(toolset)
        search = result.get_tool("tool_search")

        dumped = search.model_dump()
        assert "_toolset" not in dumped
        assert "_options" not in dumped


class TestToolSearch:
    def test_delegates_to_search_tools(self):
        toolset = _make_mock_toolset()
        meta = create_meta_tools(toolset)
        search = meta.get_tool("tool_search")

        search.execute({"query": "find employees"})

        toolset.search_tools.assert_called_once()
        call_args = toolset.search_tools.call_args
        assert call_args[0][0] == "find employees"

    def test_returns_tool_names_descriptions_and_schemas(self):
        mock_tool = _make_mock_tool(name="bamboohr_list_employees", description="List employees")
        toolset = _make_mock_toolset([mock_tool])
        meta = create_meta_tools(toolset)
        search = meta.get_tool("tool_search")

        result = search.execute({"query": "list employees"})

        assert result["total"] == 1
        tool_info = result["tools"][0]
        assert tool_info["name"] == "bamboohr_list_employees"
        assert tool_info["description"] == "List employees"
        assert "parameters" in tool_info
        assert "id" in tool_info["parameters"]

    def test_passes_connector_from_options(self):
        toolset = _make_mock_toolset()
        opts = MetaToolsOptions(connector="bamboohr")
        meta = create_meta_tools(toolset, opts)
        search = meta.get_tool("tool_search")

        search.execute({"query": "employees"})

        call_kwargs = toolset.search_tools.call_args[1]
        assert call_kwargs["connector"] == "bamboohr"

    def test_passes_account_ids_from_options(self):
        toolset = _make_mock_toolset()
        opts = MetaToolsOptions(account_ids=["acc-1", "acc-2"])
        meta = create_meta_tools(toolset, opts)
        search = meta.get_tool("tool_search")

        search.execute({"query": "employees"})

        call_kwargs = toolset.search_tools.call_args[1]
        assert call_kwargs["account_ids"] == ["acc-1", "acc-2"]

    def test_string_arguments(self):
        toolset = _make_mock_toolset()
        meta = create_meta_tools(toolset)
        search = meta.get_tool("tool_search")

        result = search.execute(json.dumps({"query": "employees"}))

        assert "tools" in result
        toolset.search_tools.assert_called_once()

    def test_validation_error_returns_error_dict(self):
        toolset = _make_mock_toolset()
        meta = create_meta_tools(toolset)
        search = meta.get_tool("tool_search")

        result = search.execute({"query": ""})

        assert "error" in result
        toolset.search_tools.assert_not_called()

    def test_invalid_json_returns_error_dict(self):
        toolset = _make_mock_toolset()
        meta = create_meta_tools(toolset)
        search = meta.get_tool("tool_search")

        result = search.execute("not valid json")

        assert "error" in result

    def test_missing_query_returns_error_dict(self):
        toolset = _make_mock_toolset()
        meta = create_meta_tools(toolset)
        search = meta.get_tool("tool_search")

        result = search.execute({})

        assert "error" in result


class TestToolExecute:
    def test_delegates_to_fetch_and_execute(self):
        toolset = MagicMock()
        toolset.api_key = "test-key"

        # Create a mock tool that returns a known result
        mock_tool = MagicMock()
        mock_tool.name = "test_tool"
        mock_tools = MagicMock()
        mock_tools.get_tool.return_value = mock_tool
        mock_tool.execute.return_value = {"result": "ok"}
        toolset.fetch_tools.return_value = mock_tools

        meta = create_meta_tools(toolset)
        execute = meta.get_tool("tool_execute")

        result = execute.execute({"tool_name": "test_tool", "parameters": {"id": "123"}})

        mock_tool.execute.assert_called_once()
        assert result == {"result": "ok"}

    def test_tool_not_found_returns_error(self):
        toolset = MagicMock()
        toolset.api_key = "test-key"
        mock_tools = MagicMock()
        mock_tools.get_tool.return_value = None
        toolset.fetch_tools.return_value = mock_tools

        meta = create_meta_tools(toolset)
        execute = meta.get_tool("tool_execute")

        result = execute.execute({"tool_name": "nonexistent_tool"})

        assert "error" in result
        assert "not found" in result["error"]

    def test_api_error_returned_as_dict(self):
        toolset = MagicMock()
        toolset.api_key = "test-key"

        mock_tool = MagicMock()
        mock_tool.name = "test_tool"
        mock_tool.execute.side_effect = StackOneAPIError(
            message="Bad Request", status_code=400, response_body="invalid"
        )
        mock_tools = MagicMock()
        mock_tools.get_tool.return_value = mock_tool
        toolset.fetch_tools.return_value = mock_tools

        meta = create_meta_tools(toolset)
        execute = meta.get_tool("tool_execute")

        result = execute.execute({"tool_name": "test_tool", "parameters": {}})

        assert "error" in result
        assert result["status_code"] == 400
        assert result["tool_name"] == "test_tool"

    def test_validation_error_returns_error_dict(self):
        toolset = _make_mock_toolset()
        meta = create_meta_tools(toolset)
        execute = meta.get_tool("tool_execute")

        result = execute.execute({"tool_name": ""})

        assert "error" in result

    def test_invalid_json_returns_error_dict(self):
        toolset = _make_mock_toolset()
        meta = create_meta_tools(toolset)
        execute = meta.get_tool("tool_execute")

        result = execute.execute("not valid json")

        assert "error" in result

    def test_caches_fetched_tools(self):
        toolset = MagicMock()
        toolset.api_key = "test-key"

        mock_tool = MagicMock()
        mock_tool.name = "test_tool"
        mock_tool.execute.return_value = {"ok": True}
        mock_tools = MagicMock()
        mock_tools.get_tool.return_value = mock_tool
        toolset.fetch_tools.return_value = mock_tools

        meta = create_meta_tools(toolset)
        execute = meta.get_tool("tool_execute")

        execute.execute({"tool_name": "test_tool"})
        execute.execute({"tool_name": "test_tool"})

        # fetch_tools should only be called once due to caching
        toolset.fetch_tools.assert_called_once()

    def test_passes_account_ids(self):
        toolset = MagicMock()
        toolset.api_key = "test-key"

        mock_tool = MagicMock()
        mock_tool.name = "test_tool"
        mock_tool.execute.return_value = {"ok": True}
        mock_tools = MagicMock()
        mock_tools.get_tool.return_value = mock_tool
        toolset.fetch_tools.return_value = mock_tools

        opts = MetaToolsOptions(account_ids=["acc-1"])
        meta = create_meta_tools(toolset, opts)
        execute = meta.get_tool("tool_execute")

        execute.execute({"tool_name": "test_tool"})

        toolset.fetch_tools.assert_called_once_with(account_ids=["acc-1"])

    def test_string_arguments(self):
        toolset = MagicMock()
        toolset.api_key = "test-key"

        mock_tool = MagicMock()
        mock_tool.name = "test_tool"
        mock_tool.execute.return_value = {"ok": True}
        mock_tools = MagicMock()
        mock_tools.get_tool.return_value = mock_tool
        toolset.fetch_tools.return_value = mock_tools

        meta = create_meta_tools(toolset)
        execute = meta.get_tool("tool_execute")

        result = execute.execute(json.dumps({"tool_name": "test_tool", "parameters": {}}))

        assert result == {"ok": True}


class TestLangChainConversion:
    def test_meta_tools_convert_to_langchain(self):
        toolset = _make_mock_toolset()
        meta = create_meta_tools(toolset)

        langchain_tools = meta.to_langchain()

        assert len(langchain_tools) == 2
        names = [t.name for t in langchain_tools]
        assert "tool_search" in names
        assert "tool_execute" in names

    def test_execute_tool_parameters_field_is_dict_type(self):
        """The 'parameters' field of tool_execute should map to dict, not str."""
        toolset = _make_mock_toolset()
        meta = create_meta_tools(toolset)
        execute_tool = meta.get_tool("tool_execute")

        langchain_tool = execute_tool.to_langchain()
        annotations = langchain_tool.args_schema.__annotations__

        assert annotations["parameters"] is dict


class TestOpenAIConversion:
    def test_meta_tools_convert_to_openai(self):
        toolset = _make_mock_toolset()
        meta = create_meta_tools(toolset)

        openai_tools = meta.to_openai()

        assert len(openai_tools) == 2
        names = [t["function"]["name"] for t in openai_tools]
        assert "tool_search" in names
        assert "tool_execute" in names

    def test_nullable_fields_not_required(self):
        toolset = _make_mock_toolset()
        meta = create_meta_tools(toolset)

        openai_tools = meta.to_openai()
        search_fn = next(t for t in openai_tools if t["function"]["name"] == "tool_search")
        required = search_fn["function"]["parameters"].get("required", [])

        assert "query" in required
        assert "connector" not in required
        assert "top_k" not in required
