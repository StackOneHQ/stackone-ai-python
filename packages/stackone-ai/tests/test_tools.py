from unittest.mock import MagicMock, patch

import pytest
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
    with patch("stackone_ai.tools.load_specs", return_value=mock_specs):
        toolset = StackOneToolSet(api_key="test_key")
        tools = toolset.get_tools(vertical="hris", account_id="test_account")

        assert len(tools) == 1
        tool = tools.get_tool("get_employee")
        assert tool is not None
        assert tool.description == "Get employee details"
        assert tool._api_key == "test_key"
        assert tool._account_id == "test_account"


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
