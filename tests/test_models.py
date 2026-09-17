from collections.abc import Sequence
from unittest.mock import MagicMock, patch

import httpx
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from langchain_core.tools import BaseTool as LangChainBaseTool
from pydantic import ValidationError

from stackone_ai.tools import StackOneTool, Tools
from stackone_ai.types import (
    ExecuteConfig,
    ParameterLocation,
    StackOneAPIError,
    StackOneError,
    ToolParameters,
    validate_method,
)

# Hypothesis strategies for PBT
VALID_HTTP_METHODS = ["GET", "POST", "PUT", "DELETE", "PATCH"]

# Strategy for case variations of valid HTTP methods
valid_method_case_variants = st.sampled_from(VALID_HTTP_METHODS).flatmap(
    lambda method: st.sampled_from(
        [
            method.lower(),
            method.upper(),
            method.capitalize(),
            method.lower().capitalize(),
        ]
    )
)

# Strategy for invalid HTTP methods
invalid_method_strategy = st.one_of(
    st.sampled_from(["OPTIONS", "HEAD", "TRACE", "CONNECT", "COPY", "MOVE", "INVALID", "FOO"]),
    st.text(min_size=1, max_size=10, alphabet="ABCDEFGHIJKLMNOPQRSTUVWXYZ").filter(
        lambda m: m.upper() not in VALID_HTTP_METHODS
    ),
)

# Strategy for invalid JSON strings (must not be parseable as valid JSON at all)
# Note: Python's json module accepts NaN/Infinity by default, so avoid those
invalid_json_strategy = st.one_of(
    st.just("{incomplete"),
    st.just('{"missing": }'),
    st.just('{"key": value}'),
    st.just("[1, 2, 3"),
    st.just("not json at all"),
    st.just("{trailing}garbage"),
    st.just("{missing closing brace"),
    st.just("undefined"),
    st.just("abc123"),
    st.just("foo bar baz"),
)

# Strategy for valid JSON that is not a dict (arrays, primitives)
# These are all valid JSON but not objects/dicts
non_dict_json_strategy = st.one_of(
    st.just("[]"),
    st.just("[1, 2, 3]"),
    st.just("[1]"),
    st.just("null"),
    st.just("true"),
    st.just("false"),
    st.just("123"),
    st.just("45.67"),
    st.just('"a string"'),
    st.just('["array", "of", "strings"]'),
)

# Strategy for account IDs
account_id_strategy = st.one_of(
    st.none(),
    st.text(alphabet="abcdefghijklmnopqrstuvwxyz0123456789_-", min_size=1, max_size=50),
)


@pytest.fixture
def mock_tool() -> StackOneTool:
    """Create a mock tool for testing"""
    return StackOneTool(
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


def test_tool_execution(mock_tool):
    """Test tool execution with parameters"""
    with patch("httpx.request") as mock_request:
        mock_response = MagicMock()
        mock_response.json.return_value = {"id": "123", "name": "Test User"}
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_request.return_value = mock_response

        result = mock_tool.execute({"id": "123"})

        assert result == {"id": "123", "name": "Test User"}
        mock_request.assert_called_once()


def test_tool_execution_with_string_args(mock_tool):
    """Test tool execution with string arguments"""
    with patch("httpx.request") as mock_request:
        mock_response = MagicMock()
        mock_response.json.return_value = {"id": "123", "name": "Test User"}
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_request.return_value = mock_response

        result = mock_tool.execute('{"id": "123"}')

        assert result == {"id": "123", "name": "Test User"}
        mock_request.assert_called_once()


def test_tool_openai_function_conversion(mock_tool):
    """Test conversion of tool to OpenAI function format"""
    openai_format = mock_tool.to_openai_function()

    assert openai_format["type"] == "function"
    assert openai_format["function"]["name"] == "test_tool"
    assert openai_format["function"]["description"] == "Test tool"
    assert "parameters" in openai_format["function"]
    assert openai_format["function"]["parameters"]["type"] == "object"
    assert "id" in openai_format["function"]["parameters"]["properties"]


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
    assert set(langchain_tool.args_schema["properties"]) == set(mock_tool.parameters.properties)


@pytest.mark.asyncio
async def test_langchain_tool_execution(mock_tool):
    """Test execution of converted LangChain tools"""
    tools = Tools(tools=[mock_tool])
    langchain_tools = tools.to_langchain()
    langchain_tool = langchain_tools[0]

    # Mock the HTTP request
    with patch("httpx.request") as mock_request:
        mock_response = MagicMock()
        mock_response.json.return_value = {"id": "test_value", "name": "Test User"}
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_request.return_value = mock_response

        # Test sync execution with correct parameter name
        test_args = {"id": "test_value"}
        result = langchain_tool._run(**test_args)

        assert result == {"id": "test_value", "name": "Test User"}
        mock_request.assert_called_once()


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
    assert set(langchain_tools[0].args_schema["properties"]) == set(mock_tool.parameters.properties.keys())
    assert set(langchain_tools[1].args_schema["properties"]) == set(second_tool.parameters.properties.keys())


class TestValidateMethod:
    """Test validate_method function"""

    def test_valid_methods(self):
        """Test valid HTTP methods"""
        assert validate_method("get") == "GET"
        assert validate_method("POST") == "POST"
        assert validate_method("put") == "PUT"
        assert validate_method("DELETE") == "DELETE"
        assert validate_method("patch") == "PATCH"

    def test_unsupported_method(self):
        """Test unsupported HTTP method raises ValueError"""
        with pytest.raises(ValueError, match="Unsupported HTTP method"):
            validate_method("OPTIONS")

    @given(method=valid_method_case_variants)
    @settings(max_examples=50)
    def test_valid_methods_case_variations_pbt(self, method: str):
        """PBT: Test valid HTTP methods with various case combinations."""
        result = validate_method(method)
        assert result in VALID_HTTP_METHODS
        assert result == method.upper()

    @given(method=invalid_method_strategy)
    @settings(max_examples=50)
    def test_invalid_methods_pbt(self, method: str):
        """PBT: Test that invalid HTTP methods raise ValueError."""
        with pytest.raises(ValueError, match="Unsupported HTTP method"):
            validate_method(method)


class TestExecuteConfig:
    """Test ExecuteConfig validation"""

    def test_invalid_method_in_config(self):
        """Test that invalid method in ExecuteConfig raises ValidationError"""
        with pytest.raises(ValidationError):
            ExecuteConfig(
                method="INVALID",
                url="https://api.example.com",
                name="test",
            )


class TestStackOneToolExecution:
    """Test StackOneTool execution edge cases"""

    @pytest.fixture
    def tool_with_locations(self) -> StackOneTool:
        """Create a tool with explicit parameter locations"""
        return StackOneTool(
            description="Test tool with param locations",
            parameters=ToolParameters(
                type="object",
                properties={
                    "path_param": {"type": "string"},
                    "query_param": {"type": "string"},
                    "body_param": {"type": "string"},
                },
            ),
            _execute_config=ExecuteConfig(
                headers={},
                method="POST",
                url="https://api.example.com/resource/{path_param}",
                name="test_tool",
                parameter_locations={
                    "path_param": ParameterLocation.PATH,
                    "query_param": ParameterLocation.QUERY,
                    "body_param": ParameterLocation.BODY,
                },
            ),
            _api_key="test_key",
        )

    def test_parameter_location_path(self, tool_with_locations):
        """Test PATH parameter location handling"""
        with patch("httpx.request") as mock_request:
            mock_response = MagicMock()
            mock_response.json.return_value = {"success": True}
            mock_response.status_code = 200
            mock_response.raise_for_status = MagicMock()
            mock_request.return_value = mock_response

            tool_with_locations.execute(
                {
                    "path_param": "test_id",
                    "query_param": "filter",
                    "body_param": "data",
                }
            )

            call_kwargs = mock_request.call_args[1]
            assert "resource/test_id" in call_kwargs["url"]
            assert call_kwargs["params"] == {"query_param": "filter"}
            assert call_kwargs["json"] == {"body_param": "data"}

    def test_account_id_in_headers(self):
        """Test account ID is added to headers"""
        tool = StackOneTool(
            description="Test",
            parameters=ToolParameters(type="object", properties={}),
            _execute_config=ExecuteConfig(
                headers={},
                method="GET",
                url="https://api.example.com/test",
                name="test",
            ),
            _api_key="test_key",
            _account_id="acc123",
        )

        with patch("httpx.request") as mock_request:
            mock_response = MagicMock()
            mock_response.json.return_value = {}
            mock_response.status_code = 200
            mock_response.raise_for_status = MagicMock()
            mock_request.return_value = mock_response

            tool.execute({})

            call_kwargs = mock_request.call_args[1]
            assert call_kwargs["headers"]["x-account-id"] == "acc123"

    def test_invalid_json_arguments(self, mock_tool):
        """Test invalid JSON string raises ValueError"""
        with pytest.raises(ValueError, match="Invalid JSON"):
            mock_tool.execute("not valid json")

    def test_non_dict_arguments(self, mock_tool):
        """Test non-dict JSON raises ValueError"""
        with pytest.raises(ValueError, match="Tool arguments must be a JSON object"):
            mock_tool.execute("[1, 2, 3]")

    @given(invalid_json=invalid_json_strategy)
    @settings(max_examples=50)
    def test_invalid_json_arguments_pbt(self, invalid_json: str):
        """PBT: Test various invalid JSON strings raise ValueError."""
        # Create tool inside the test to avoid fixture issues with Hypothesis
        tool = StackOneTool(
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
        with pytest.raises(ValueError, match="Invalid JSON"):
            tool.execute(invalid_json)

    @given(non_dict_json=non_dict_json_strategy)
    @settings(max_examples=50)
    def test_non_dict_arguments_pbt(self, non_dict_json: str):
        """PBT: Test non-dict JSON values raise ValueError."""
        # Create tool inside the test to avoid fixture issues with Hypothesis
        tool = StackOneTool(
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
        with pytest.raises(ValueError, match="Tool arguments must be a JSON object"):
            tool.execute(non_dict_json)

    def test_form_body_type(self):
        """Test form body type handling"""
        tool = StackOneTool(
            description="Test",
            parameters=ToolParameters(
                type="object",
                properties={"field": {"type": "string"}},
            ),
            _execute_config=ExecuteConfig(
                headers={},
                method="POST",
                url="https://api.example.com/test",
                name="test",
                body_type="form",
            ),
            _api_key="test_key",
        )

        with patch("httpx.request") as mock_request:
            mock_response = MagicMock()
            mock_response.json.return_value = {}
            mock_response.status_code = 200
            mock_response.raise_for_status = MagicMock()
            mock_request.return_value = mock_response

            tool.execute({"field": "value"})

            call_kwargs = mock_request.call_args[1]
            assert call_kwargs["data"] == {"field": "value"}
            assert "json" not in call_kwargs

    def test_http_status_error_with_json_body(self, mock_tool):
        """Test HTTP error with JSON response body"""
        with patch("httpx.request") as mock_request:
            mock_response = MagicMock()
            mock_response.status_code = 400
            mock_response.text = '{"error": "Bad request"}'
            mock_response.json.return_value = {"error": "Bad request"}
            mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
                "Bad Request",
                request=MagicMock(),
                response=mock_response,
            )
            mock_request.return_value = mock_response

            with pytest.raises(StackOneAPIError) as exc_info:
                mock_tool.execute({"id": "123"})

            assert exc_info.value.status_code == 400
            assert exc_info.value.response_body == {"error": "Bad request"}

    def test_http_status_error_with_text_body(self, mock_tool):
        """Test HTTP error with plain text response body"""
        import json as json_module

        with patch("httpx.request") as mock_request:
            mock_response = MagicMock()
            mock_response.status_code = 500
            mock_response.text = "Internal Server Error"
            mock_response.json.side_effect = json_module.JSONDecodeError("No JSON", "", 0)
            mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
                "Server Error",
                request=MagicMock(),
                response=mock_response,
            )
            mock_request.return_value = mock_response

            with pytest.raises(StackOneAPIError) as exc_info:
                mock_tool.execute({"id": "123"})

            assert exc_info.value.status_code == 500
            assert exc_info.value.response_body == "Internal Server Error"

    def test_request_error(self, mock_tool):
        """Test network/request error handling"""
        with patch("httpx.request") as mock_request:
            mock_request.side_effect = httpx.RequestError("Connection failed")

            with pytest.raises(StackOneError, match="Request failed"):
                mock_tool.execute({"id": "123"})

    def test_non_dict_response(self, mock_tool):
        """Test non-dict JSON response is wrapped"""
        with patch("httpx.request") as mock_request:
            mock_response = MagicMock()
            mock_response.json.return_value = ["item1", "item2"]
            mock_response.status_code = 200
            mock_response.raise_for_status = MagicMock()
            mock_request.return_value = mock_response

            result = mock_tool.execute({"id": "123"})
            assert result == {"result": ["item1", "item2"]}


class TestStackOneToolOpenAIConversion:
    """Test OpenAI function conversion edge cases"""

    def test_enum_property(self):
        """Test enum property is included in OpenAI format"""
        tool = StackOneTool(
            description="Test",
            parameters=ToolParameters(
                type="object",
                properties={
                    "status": {
                        "type": "string",
                        "enum": ["active", "inactive"],
                        "description": "Status",
                    }
                },
            ),
            _execute_config=ExecuteConfig(
                headers={},
                method="GET",
                url="https://api.example.com",
                name="test",
            ),
            _api_key="test_key",
        )

        openai_format = tool.to_openai_function()
        props = openai_format["function"]["parameters"]["properties"]
        assert props["status"]["enum"] == ["active", "inactive"]

    def test_array_type_property(self):
        """Test array type with items is converted"""
        tool = StackOneTool(
            description="Test",
            parameters=ToolParameters(
                type="object",
                properties={
                    "tags": {
                        "type": "array",
                        "items": {"type": "string", "description": "Tag"},
                    }
                },
            ),
            _execute_config=ExecuteConfig(
                headers={},
                method="GET",
                url="https://api.example.com",
                name="test",
            ),
            _api_key="test_key",
        )

        openai_format = tool.to_openai_function()
        props = openai_format["function"]["parameters"]["properties"]
        assert props["tags"]["type"] == "array"
        assert props["tags"]["items"]["type"] == "string"

    def test_object_type_property(self):
        """Test object type with nested properties is converted"""
        tool = StackOneTool(
            description="Test",
            parameters=ToolParameters(
                type="object",
                properties={
                    "address": {
                        "type": "object",
                        "properties": {
                            "street": {"type": "string"},
                            "city": {"type": "string"},
                        },
                    }
                },
            ),
            _execute_config=ExecuteConfig(
                headers={},
                method="GET",
                url="https://api.example.com",
                name="test",
            ),
            _api_key="test_key",
        )

        openai_format = tool.to_openai_function()
        props = openai_format["function"]["parameters"]["properties"]
        assert props["address"]["type"] == "object"
        assert "street" in props["address"]["properties"]

    def test_non_dict_property(self):
        """Test non-dict property is converted to string type"""
        tool = StackOneTool(
            description="Test",
            parameters=ToolParameters(
                type="object",
                properties={
                    "simple": "string",  # non-dict property
                },
            ),
            _execute_config=ExecuteConfig(
                headers={},
                method="GET",
                url="https://api.example.com",
                name="test",
            ),
            _api_key="test_key",
        )

        openai_format = tool.to_openai_function()
        props = openai_format["function"]["parameters"]["properties"]
        assert props["simple"]["type"] == "string"


class TestStackOneToolLangChainConversion:
    """The LangChain args schema must be the served schema, not a lossy rebuild.

    It used to be rebuilt from each property's top-level `type`, which discarded every
    nested object's fields, every enum, bound, item type and union — so a model was
    told "pass an object" with no field names. These pin that it is passed through.
    """

    @staticmethod
    def _tool(properties: dict) -> StackOneTool:
        return StackOneTool(
            description="Test",
            parameters=ToolParameters(type="object", properties=properties),
            _execute_config=ExecuteConfig(
                headers={}, method="GET", url="https://api.example.com", name="test"
            ),
            _api_key="test_key",
        )

    def test_nested_object_fields_survive(self):
        served = {
            "body_variables": {
                "type": "object",
                "description": "Variables",
                "properties": {"teamId": {"type": "string"}, "title": {"type": "string"}},
                "required": ["teamId", "title"],
                "nullable": False,
            }
        }
        schema = self._tool(served).to_langchain().args_schema
        nested = schema["properties"]["body_variables"]
        assert set(nested["properties"]) == {"teamId", "title"}
        assert nested["required"] == ["teamId", "title"]

    def test_constraints_survive(self):
        served = {
            "status": {"type": "string", "enum": ["open", "closed"], "nullable": False},
            "count": {"type": "integer", "minimum": 1, "maximum": 10, "nullable": True},
            "tags": {"type": "array", "items": {"type": "string"}, "nullable": True},
        }
        schema = self._tool(served).to_langchain().args_schema
        assert schema["properties"]["status"]["enum"] == ["open", "closed"]
        assert schema["properties"]["count"]["minimum"] == 1
        assert schema["properties"]["tags"]["items"] == {"type": "string"}

    def test_requiredness_matches_to_openai_function(self):
        tool = self._tool(
            {"a": {"type": "string", "nullable": False}, "b": {"type": "string", "nullable": True}}
        )
        assert tool.to_langchain().args_schema == tool.to_openai_function()["function"]["parameters"]

    def test_internal_nullable_marker_is_not_exposed(self):
        tool = self._tool({"a": {"type": "string", "nullable": False}})
        assert "nullable" not in tool.to_langchain().args_schema["properties"]["a"]

    @pytest.mark.asyncio
    async def test_arun_method(self):
        """Test async _arun method"""
        tool = StackOneTool(
            description="Test",
            parameters=ToolParameters(
                type="object",
                properties={"id": {"type": "string"}},
            ),
            _execute_config=ExecuteConfig(
                headers={},
                method="GET",
                url="https://api.example.com",
                name="test",
            ),
            _api_key="test_key",
        )

        lc_tool = tool.to_langchain()

        with patch("httpx.request") as mock_request:
            mock_response = MagicMock()
            mock_response.json.return_value = {"result": "async_test"}
            mock_response.status_code = 200
            mock_response.raise_for_status = MagicMock()
            mock_request.return_value = mock_response

            result = await lc_tool._arun(id="123")
            assert result == {"result": "async_test"}


class TestStackOneToolAccountId:
    """Test account ID methods"""

    def test_set_and_get_account_id(self):
        """Test setting and getting account ID"""
        tool = StackOneTool(
            description="Test",
            parameters=ToolParameters(type="object", properties={}),
            _execute_config=ExecuteConfig(
                headers={},
                method="GET",
                url="https://api.example.com",
                name="test",
            ),
            _api_key="test_key",
        )

        assert tool.get_account_id() is None

        tool.set_account_id("new_account")
        assert tool.get_account_id() == "new_account"

        tool.set_account_id(None)
        assert tool.get_account_id() is None

    @given(account_id=account_id_strategy)
    @settings(max_examples=50)
    def test_account_id_round_trip_pbt(self, account_id: str | None):
        """PBT: Test setting and getting various account ID values."""
        tool = StackOneTool(
            description="Test",
            parameters=ToolParameters(type="object", properties={}),
            _execute_config=ExecuteConfig(
                headers={},
                method="GET",
                url="https://api.example.com",
                name="test",
            ),
            _api_key="test_key",
        )

        tool.set_account_id(account_id)
        assert tool.get_account_id() == account_id


class TestToolsContainer:
    """Test Tools container class"""

    @pytest.fixture
    def sample_tools(self) -> list[StackOneTool]:
        """Create sample tools for testing"""
        tool1 = StackOneTool(
            description="Tool 1",
            parameters=ToolParameters(type="object", properties={}),
            _execute_config=ExecuteConfig(
                headers={},
                method="GET",
                url="https://api.example.com/1",
                name="tool_1",
            ),
            _api_key="key",
            _account_id="acc1",
        )
        tool2 = StackOneTool(
            description="Tool 2",
            parameters=ToolParameters(type="object", properties={}),
            _execute_config=ExecuteConfig(
                headers={},
                method="GET",
                url="https://api.example.com/2",
                name="tool_2",
            ),
            _api_key="key",
        )
        return [tool1, tool2]

    def test_iteration(self, sample_tools):
        """Test Tools is iterable"""
        tools = Tools(sample_tools)
        collected = list(tools)
        assert len(collected) == 2
        assert collected[0].name == "tool_1"
        assert collected[1].name == "tool_2"

    def test_set_account_id_all_tools(self, sample_tools):
        """Test set_account_id sets for all tools"""
        tools = Tools(sample_tools)
        tools.set_account_id("new_account")

        for tool in tools:
            assert tool.get_account_id() == "new_account"

    def test_get_account_id_returns_first_non_none(self, sample_tools):
        """Test get_account_id returns first non-None account ID"""
        tools = Tools(sample_tools)
        assert tools.get_account_id() == "acc1"

    def test_get_account_id_returns_none_when_all_none(self):
        """Test get_account_id returns None when all tools have None"""
        tool = StackOneTool(
            description="Test",
            parameters=ToolParameters(type="object", properties={}),
            _execute_config=ExecuteConfig(
                headers={},
                method="GET",
                url="https://api.example.com",
                name="test",
            ),
            _api_key="key",
        )
        tools = Tools([tool])
        assert tools.get_account_id() is None


class TestOpenAISchemaPassThrough:
    """The served schema must reach the model intact.

    The conformance suite's --strict-schema gate requires that the schema listed
    to a model is the schema the server served. An earlier allowlist copied only
    type/description/enum, silently dropping every constraint, so a model could
    not generate valid arguments for a constrained field.
    """

    @staticmethod
    def _tool(properties: dict) -> StackOneTool:
        return StackOneTool(
            description="Test tool",
            parameters=ToolParameters(type="object", properties=properties),
            _execute_config=ExecuteConfig(
                headers={}, method="POST", url="https://api.example.com/x", name="schema_tool"
            ),
            _api_key="key",
        )

    def test_preserves_constraint_keywords(self):
        tool = self._tool(
            {
                "email": {
                    "type": "string",
                    "format": "email",
                    "pattern": r"^\S+@\S+$",
                    "nullable": False,
                },
                "count": {"type": "integer", "minimum": 1, "maximum": 100, "default": 10},
            }
        )

        props = tool.to_openai_function()["function"]["parameters"]["properties"]

        assert props["email"]["format"] == "email"
        assert props["email"]["pattern"] == r"^\S+@\S+$"
        assert props["count"]["minimum"] == 1
        assert props["count"]["maximum"] == 100
        assert props["count"]["default"] == 10

    def test_preserves_composition_and_nested_required(self):
        tool = self._tool(
            {
                "payload": {
                    "type": "object",
                    "properties": {"a": {"type": "string"}, "b": {"type": "integer"}},
                    "required": ["a"],
                    "nullable": False,
                },
                "either": {"oneOf": [{"type": "string"}, {"type": "integer"}], "nullable": True},
            }
        )

        props = tool.to_openai_function()["function"]["parameters"]["properties"]

        assert props["payload"]["required"] == ["a"]
        assert props["payload"]["properties"]["b"]["type"] == "integer"
        assert props["either"]["oneOf"] == [{"type": "string"}, {"type": "integer"}]

    def test_preserves_unknown_keywords(self):
        """A keyword this SDK has never heard of must still reach the model."""
        tool = self._tool({"x": {"type": "string", "x-vendor-hint": "something", "nullable": True}})

        props = tool.to_openai_function()["function"]["parameters"]["properties"]

        assert props["x"]["x-vendor-hint"] == "something"

    def test_strips_internal_nullable_marker_and_derives_required(self):
        """`nullable` is an SDK-internal marker; it becomes JSON Schema `required`."""
        tool = self._tool(
            {
                "needed": {"type": "string", "nullable": False},
                "optional": {"type": "string", "nullable": True},
            }
        )

        params = tool.to_openai_function()["function"]["parameters"]

        assert "nullable" not in params["properties"]["needed"]
        assert "nullable" not in params["properties"]["optional"]
        assert params["required"] == ["needed"]

    def test_strips_nested_internal_marker(self):
        tool = self._tool(
            {
                "obj": {
                    "type": "object",
                    "properties": {"inner": {"type": "string", "nullable": True}},
                    "nullable": False,
                }
            }
        )

        props = tool.to_openai_function()["function"]["parameters"]["properties"]

        assert "nullable" not in props["obj"]["properties"]["inner"]
