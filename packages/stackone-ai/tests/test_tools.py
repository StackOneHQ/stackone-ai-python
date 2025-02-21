from collections.abc import Sequence
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.tools import BaseTool as LangChainBaseTool
from stackone_ai.models import (
    BaseTool,
    ExecuteConfig,
    ToolDefinition,
    ToolParameters,
    Tools,
)
from stackone_ai.tools import StackOneToolSet


@pytest.fixture
def mock_tool() -> BaseTool:
    """Create a mock tool for testing"""
    return BaseTool(
        description="Test tool",
        parameters=ToolParameters(
            type="object",
            properties={"id": {"type": "string"}},
        ),
        _execute_config=ExecuteConfig(
            headers={},
            method="GET",
            url="https://api.example.com/test/{id}",
            name="test_tool",
        ),
        _api_key="test_key",
    )


@pytest.fixture
def mock_specs() -> dict:
    """Create mock tool specifications"""
    return {
        "hris": {
            "get_employee": ToolDefinition(
                description="Get employee details",
                parameters=ToolParameters(
                    type="object",
                    properties={"id": {"type": "string"}},
                ),
                execute=ExecuteConfig(
                    headers={},
                    method="GET",
                    url="https://api.example.com/employee/{id}",
                    name="get_employee",
                ),
            )
        }
    }


def test_tool_execution(mock_tool):
    """Test tool execution with parameters"""
    with patch("requests.request") as mock_request:
        mock_response = MagicMock()
        mock_response.json.return_value = {"id": "123", "name": "Test User"}
        mock_request.return_value = mock_response

        result = mock_tool.execute({"id": "123"})

        assert result == {"id": "123", "name": "Test User"}
        mock_request.assert_called_once()


def test_tool_execution_with_string_args(mock_tool):
    """Test tool execution with string arguments"""
    with patch("requests.request") as mock_request:
        mock_response = MagicMock()
        mock_response.json.return_value = {"id": "123", "name": "Test User"}
        mock_request.return_value = mock_response

        result = mock_tool.execute('{"id": "123"}')

        assert result == {"id": "123", "name": "Test User"}
        mock_request.assert_called_once()


def test_toolset_initialization(mock_specs):
    """Test StackOneToolSet initialization and tool creation"""
    mock_spec_content = {
        "paths": {
            "/employee/{id}": {
                "get": {
                    "x-speakeasy-name-override": "get_employee",
                    "description": "Get employee details",
                    "parameters": [
                        {
                            "in": "path",
                            "name": "id",
                            "schema": {"type": "string"},
                            "description": "Employee ID",
                        }
                    ],
                }
            }
        }
    }

    # Mock the file operations instead of load_specs
    with (
        patch("stackone_ai.tools.OAS_DIR") as mock_dir,
        patch("json.load") as mock_json,
    ):
        # Setup mocks
        mock_path = MagicMock()
        mock_path.exists.return_value = True
        mock_dir.__truediv__.return_value = mock_path
        mock_json.return_value = mock_spec_content

        # Create and test toolset
        toolset = StackOneToolSet(api_key="test_key")
        tools = toolset.get_tools(vertical="hris", account_id="test_account")

        # Verify results
        assert len(tools) == 1
        tool = tools.get_tool("get_employee")
        assert tool is not None
        assert tool.description == "Get employee details"
        assert tool._api_key == "test_key"
        assert tool._account_id == "test_account"

        # Verify the tool parameters
        assert tool.parameters.properties["id"]["type"] == "string"
        assert tool.parameters.properties["id"]["description"] == "Employee ID"


def test_tool_openai_function_conversion(mock_tool):
    """Test conversion of tool to OpenAI function format"""
    openai_format = mock_tool.to_openai_function()

    assert openai_format["type"] == "function"
    assert openai_format["function"]["name"] == "test_tool"
    assert openai_format["function"]["description"] == "Test tool"
    assert "parameters" in openai_format["function"]
    assert openai_format["function"]["parameters"]["type"] == "object"
    assert "id" in openai_format["function"]["parameters"]["properties"]


def test_unknown_vertical():
    """Test getting tools for unknown vertical"""
    toolset = StackOneToolSet(api_key="test_key")
    tools = toolset.get_tools(vertical="unknown")
    assert len(tools) == 0


def test_tools_container_methods(mock_tool):
    """Test Tools container class methods"""
    tools = [mock_tool]
    tools_container = Tools(tools=tools)

    assert len(tools_container) == 1
    assert tools_container[0] == mock_tool
    assert tools_container.get_tool("test_tool") == mock_tool
    assert tools_container.get_tool("nonexistent") is None

    openai_tools = tools_container.to_openai()
    assert len(openai_tools) == 1
    assert openai_tools[0]["type"] == "function"


def test_to_langchain_conversion(mock_tool):
    """Test conversion of tools to LangChain format"""
    tools = Tools(tools=[mock_tool])
    langchain_tools = tools.to_langchain()

    # Check return type
    assert isinstance(langchain_tools, Sequence)
    assert len(langchain_tools) == 1

    # Check converted tool
    langchain_tool = langchain_tools[0]
    assert isinstance(langchain_tool, LangChainBaseTool)
    assert langchain_tool.name == mock_tool.name
    assert langchain_tool.description == mock_tool.description

    # Check args schema
    assert hasattr(langchain_tool, "args_schema")
    # Just check the field names match
    assert set(langchain_tool.args_schema.__annotations__.keys()) == set(
        mock_tool.parameters.properties.keys()
    )


@pytest.mark.asyncio
async def test_langchain_tool_execution(mock_tool):
    """Test execution of converted LangChain tools"""
    tools = Tools(tools=[mock_tool])
    langchain_tools = tools.to_langchain()
    langchain_tool = langchain_tools[0]

    # Mock the HTTP request
    with patch("requests.request") as mock_request:
        mock_response = MagicMock()
        mock_response.json.return_value = {"id": "test_value", "name": "Test User"}
        mock_request.return_value = mock_response

        # Test sync execution with correct parameter name
        test_args = {"id": "test_value"}
        result = langchain_tool.invoke(test_args)

        # The result should match what our mock returns
        assert result == {"id": "test_value", "name": "Test User"}

        # Test async execution
        async_result = await langchain_tool.ainvoke(test_args)
        assert async_result == {"id": "test_value", "name": "Test User"}

        # Verify the request was made correctly
        mock_request.assert_called_with(
            method="GET",
            url="https://api.example.com/test/test_value",
            headers={
                "Authorization": "Basic dGVzdF9rZXk6",
                "User-Agent": "stackone-python/1.0.0",
            },
        )


def test_to_langchain_empty_tools():
    """Test conversion of empty tools list to LangChain format"""
    tools = Tools(tools=[])
    langchain_tools = tools.to_langchain()

    assert isinstance(langchain_tools, Sequence)
    assert len(langchain_tools) == 0


def test_to_langchain_multiple_tools(mock_tool):
    """Test conversion of multiple tools to LangChain format"""
    # Create a second mock tool with different parameters
    second_tool = mock_tool.__class__(
        description="Second test tool",
        parameters=ToolParameters(type="object", properties={"other_param": "string"}),
        _execute_config=ExecuteConfig(
            headers={}, method="GET", url="https://test.com/api/v2", name="second_test_tool"
        ),
        _api_key="test_key",
    )

    tools = Tools(tools=[mock_tool, second_tool])
    langchain_tools = tools.to_langchain()

    assert len(langchain_tools) == 2
    assert langchain_tools[0].name == mock_tool.name
    assert langchain_tools[1].name == second_tool.name

    # Verify each tool has correct schema
    assert set(langchain_tools[0].args_schema.__annotations__.keys()) == set(
        mock_tool.parameters.properties.keys()
    )
    assert set(langchain_tools[1].args_schema.__annotations__.keys()) == set(
        second_tool.parameters.properties.keys()
    )
