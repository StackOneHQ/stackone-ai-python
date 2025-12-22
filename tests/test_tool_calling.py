"""Tests for tool calling functionality"""

import json

import httpx
import pytest
import respx

from stackone_ai import StackOneTool
from stackone_ai.models import ExecuteConfig, ToolParameters


@pytest.fixture
def mock_tool():
    """Create a mock tool for testing"""
    execute_config = ExecuteConfig(
        name="test_tool",
        method="POST",
        url="https://api.example.com/test",
        headers={"Content-Type": "application/json"},
    )

    parameters = ToolParameters(
        type="object",
        properties={
            "name": {"type": "string", "description": "Name parameter"},
            "value": {"type": "number", "description": "Value parameter"},
        },
    )

    tool = StackOneTool(
        description="Test tool",
        parameters=parameters,
        _execute_config=execute_config,
        _api_key="test_api_key",
    )

    return tool


class TestToolCalling:
    """Test tool calling functionality"""

    @respx.mock
    def test_call_with_kwargs(self, mock_tool):
        """Test calling a tool with keyword arguments"""
        # Mock the API response
        route = respx.post("https://api.example.com/test").mock(
            return_value=httpx.Response(200, json={"success": True, "result": "test_result"})
        )

        # Call the tool with kwargs
        result = mock_tool.call(name="test", value=42)

        # Verify the result
        assert result == {"success": True, "result": "test_result"}

        # Verify the request was made correctly
        assert route.called
        assert route.call_count == 1
        request = route.calls[0].request
        assert json.loads(request.content) == {"name": "test", "value": 42}

    @respx.mock
    def test_call_with_dict_arg(self, mock_tool):
        """Test calling a tool with a dictionary argument"""
        # Mock the API response
        route = respx.post("https://api.example.com/test").mock(
            return_value=httpx.Response(200, json={"success": True, "result": "test_result"})
        )

        # Call the tool with a dict
        result = mock_tool.call({"name": "test", "value": 42})

        # Verify the result
        assert result == {"success": True, "result": "test_result"}

        # Verify the request
        assert route.called
        assert route.call_count == 1
        request = route.calls[0].request
        assert json.loads(request.content) == {"name": "test", "value": 42}

    @respx.mock
    def test_call_with_json_string(self, mock_tool):
        """Test calling a tool with a JSON string argument"""
        # Mock the API response
        route = respx.post("https://api.example.com/test").mock(
            return_value=httpx.Response(200, json={"success": True, "result": "test_result"})
        )

        # Call the tool with a JSON string
        result = mock_tool.call('{"name": "test", "value": 42}')

        # Verify the result
        assert result == {"success": True, "result": "test_result"}

        # Verify the request
        assert route.called
        assert route.call_count == 1
        request = route.calls[0].request
        assert json.loads(request.content) == {"name": "test", "value": 42}

    def test_call_with_both_args_and_kwargs_raises_error(self, mock_tool):
        """Test that providing both args and kwargs raises an error"""
        with pytest.raises(ValueError, match="Cannot provide both positional and keyword arguments"):
            mock_tool.call({"name": "test"}, value=42)

    def test_call_with_multiple_args_raises_error(self, mock_tool):
        """Test that providing multiple positional arguments raises an error"""
        with pytest.raises(ValueError, match="Only one positional argument is allowed"):
            mock_tool.call({"name": "test"}, {"value": 42})

    @respx.mock
    def test_call_without_arguments(self, mock_tool):
        """Test calling a tool without any arguments"""
        # Mock the API response
        route = respx.post("https://api.example.com/test").mock(
            return_value=httpx.Response(200, json={"success": True, "result": "no_args"})
        )

        # Call the tool without arguments
        result = mock_tool.call()

        # Verify the result
        assert result == {"success": True, "result": "no_args"}

        # Verify the request body is empty or contains empty JSON
        assert route.called
        assert route.call_count == 1
        request = route.calls[0].request
        # Handle case where body might be None for empty POST
        if request.content:
            assert json.loads(request.content) == {}
        else:
            assert request.content == b""
