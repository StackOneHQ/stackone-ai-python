"""Tests for tool calling functionality"""

import json

import httpx
import pytest
import respx

from stackone_ai import StackOneTool
from stackone_ai.models import ExecuteConfig, ToolParameters
from stackone_ai.toolset import _StackOneRpcTool


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


class TestStackOneRpcTool:
    """Test _StackOneRpcTool functionality"""

    @pytest.fixture
    def rpc_tool(self):
        """Create a mock RPC tool for testing"""
        parameters = ToolParameters(
            type="object",
            properties={
                "employee_id": {"type": "string", "description": "Employee ID"},
            },
        )
        return _StackOneRpcTool(
            name="hibob_get_employee",
            description="Get employee details",
            parameters=parameters,
            api_key="test_api_key",
            base_url="https://api.stackone.com",
            account_id="test_account",
        )

    @respx.mock
    def test_execute_basic(self, rpc_tool):
        """Test basic RPC tool execution"""
        route = respx.post("https://api.stackone.com/actions/rpc").mock(
            return_value=httpx.Response(200, json={"data": {"id": "123", "name": "John"}})
        )

        result = rpc_tool.execute({"employee_id": "123"})

        assert result == {"data": {"id": "123", "name": "John"}}
        assert route.called
        request = route.calls[0].request
        body = json.loads(request.content)
        assert body["action"] == "hibob_get_employee"
        assert body["body"]["employee_id"] == "123"
        assert body["headers"]["x-account-id"] == "test_account"

    @respx.mock
    def test_execute_with_json_string(self, rpc_tool):
        """Test RPC tool execution with JSON string arguments"""
        route = respx.post("https://api.stackone.com/actions/rpc").mock(
            return_value=httpx.Response(200, json={"success": True})
        )

        result = rpc_tool.execute('{"employee_id": "456"}')

        assert result == {"success": True}
        assert route.called
        body = json.loads(route.calls[0].request.content)
        assert body["body"]["employee_id"] == "456"

    @respx.mock
    def test_execute_with_body_payload(self, rpc_tool):
        """Test RPC tool execution with nested body payload"""
        route = respx.post("https://api.stackone.com/actions/rpc").mock(
            return_value=httpx.Response(200, json={"success": True})
        )

        result = rpc_tool.execute({"body": {"name": "Jane", "email": "jane@example.com"}})

        assert result == {"success": True}
        body = json.loads(route.calls[0].request.content)
        assert body["body"]["name"] == "Jane"
        assert body["body"]["email"] == "jane@example.com"

    @respx.mock
    def test_execute_with_path_payload(self, rpc_tool):
        """Test RPC tool execution with path parameters"""
        route = respx.post("https://api.stackone.com/actions/rpc").mock(
            return_value=httpx.Response(200, json={"success": True})
        )

        result = rpc_tool.execute({"path": {"id": "emp123"}})

        assert result == {"success": True}
        body = json.loads(route.calls[0].request.content)
        assert body["path"] == {"id": "emp123"}

    @respx.mock
    def test_execute_with_query_payload(self, rpc_tool):
        """Test RPC tool execution with query parameters"""
        route = respx.post("https://api.stackone.com/actions/rpc").mock(
            return_value=httpx.Response(200, json={"success": True})
        )

        result = rpc_tool.execute({"query": {"limit": "10", "offset": "0"}})

        assert result == {"success": True}
        body = json.loads(route.calls[0].request.content)
        assert body["query"] == {"limit": "10", "offset": "0"}

    @respx.mock
    def test_execute_with_headers_payload(self, rpc_tool):
        """Test RPC tool execution with custom headers"""
        route = respx.post("https://api.stackone.com/actions/rpc").mock(
            return_value=httpx.Response(200, json={"success": True})
        )

        result = rpc_tool.execute({"headers": {"X-Custom-Header": "custom_value"}})

        assert result == {"success": True}
        body = json.loads(route.calls[0].request.content)
        assert body["headers"]["X-Custom-Header"] == "custom_value"
        assert body["headers"]["x-account-id"] == "test_account"

    @respx.mock
    def test_execute_headers_strips_authorization(self, rpc_tool):
        """Test that Authorization header is stripped from action headers"""
        route = respx.post("https://api.stackone.com/actions/rpc").mock(
            return_value=httpx.Response(200, json={"success": True})
        )

        result = rpc_tool.execute({"headers": {"Authorization": "Bearer token", "X-Other": "value"}})

        assert result == {"success": True}
        body = json.loads(route.calls[0].request.content)
        assert "Authorization" not in body["headers"]
        assert body["headers"]["X-Other"] == "value"

    @respx.mock
    def test_execute_headers_skips_none_values(self, rpc_tool):
        """Test that None header values are skipped"""
        route = respx.post("https://api.stackone.com/actions/rpc").mock(
            return_value=httpx.Response(200, json={"success": True})
        )

        result = rpc_tool.execute({"headers": {"X-Present": "value", "X-Absent": None}})

        assert result == {"success": True}
        body = json.loads(route.calls[0].request.content)
        assert body["headers"]["X-Present"] == "value"
        assert "X-Absent" not in body["headers"]

    @respx.mock
    def test_execute_without_account_id(self):
        """Test RPC tool execution without account ID"""
        parameters = ToolParameters(
            type="object",
            properties={},
        )
        tool = _StackOneRpcTool(
            name="test_tool",
            description="Test",
            parameters=parameters,
            api_key="test_key",
            base_url="https://api.stackone.com",
            account_id=None,
        )

        route = respx.post("https://api.stackone.com/actions/rpc").mock(
            return_value=httpx.Response(200, json={"success": True})
        )

        result = tool.execute({})

        assert result == {"success": True}
        body = json.loads(route.calls[0].request.content)
        assert "x-account-id" not in body["headers"]

    @respx.mock
    def test_execute_with_none_arguments(self, rpc_tool):
        """Test RPC tool execution with None arguments"""
        route = respx.post("https://api.stackone.com/actions/rpc").mock(
            return_value=httpx.Response(200, json={"success": True})
        )

        result = rpc_tool.execute(None)

        assert result == {"success": True}
        body = json.loads(route.calls[0].request.content)
        assert body["action"] == "hibob_get_employee"
        assert body["body"] == {}

    def test_parse_arguments_invalid_json(self, rpc_tool):
        """Test that invalid JSON raises ValueError"""
        with pytest.raises(ValueError):
            rpc_tool._parse_arguments("not valid json")

    def test_parse_arguments_non_dict(self, rpc_tool):
        """Test that non-dict JSON raises ValueError"""
        with pytest.raises(ValueError, match="Tool arguments must be a JSON object"):
            rpc_tool._parse_arguments("[1, 2, 3]")

    def test_extract_record_with_dict(self, rpc_tool):
        """Test _extract_record with dict input"""
        result = rpc_tool._extract_record({"key": "value"})
        assert result == {"key": "value"}

    def test_extract_record_with_non_dict(self, rpc_tool):
        """Test _extract_record with non-dict input"""
        assert rpc_tool._extract_record("string") is None
        assert rpc_tool._extract_record(123) is None
        assert rpc_tool._extract_record(None) is None
