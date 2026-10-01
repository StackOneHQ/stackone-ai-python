from collections.abc import Sequence
from typing import Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from langchain_core.tools import BaseTool as LangChainBaseTool
from pydantic import ValidationError

from stackone_ai.tools import StackOneMcpTool, StackOneTool, Tools
from stackone_ai.types import (
    ExecuteConfig,
    StackOneAPIError,
    StackOneError,
    ToolParameters,
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


def _mcp_tool(name: str = "test_tool", account_id: str | None = None) -> StackOneMcpTool:
    return StackOneMcpTool(
        name=name,
        description="Test tool",
        parameters=ToolParameters(type="object", properties={"id": {"type": "string"}}),
        api_key="test_key",
        endpoint="https://api.example.com/mcp",
        account_id=account_id,
    )


@pytest.fixture
def mcp_calls(monkeypatch) -> list[dict[str, Any]]:
    """Replace the MCP transport; each call is recorded and answered with a fixed payload."""
    calls: list[dict[str, Any]] = []

    def fake_call(endpoint, headers, name, arguments, **kwargs):
        calls.append(
            {"endpoint": endpoint, "headers": headers, "name": name, "arguments": arguments, **kwargs}
        )
        return {"id": arguments.get("id"), "name": "Test User"}

    monkeypatch.setattr("stackone_ai.tools.call_mcp_tool", fake_call)
    return calls


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
            name="test_tool",
        ),
        _api_key="test_key",
    )


def test_base_tool_has_no_executor(mock_tool):
    """A hand-built StackOneTool has nothing to call; it must say so, not send anything."""
    with pytest.raises(StackOneError, match=r'Tool "test_tool" has no executor\. Override execute\(\)'):
        mock_tool.execute({"id": "123"})


def test_tool_execution(mcp_calls):
    """Test tool execution with parameters"""
    result = _mcp_tool().execute({"id": "123"})

    assert result == {"id": "123", "name": "Test User"}
    assert len(mcp_calls) == 1
    assert mcp_calls[0]["name"] == "test_tool"
    assert mcp_calls[0]["arguments"] == {"id": "123"}


def test_tool_execution_with_string_args(mcp_calls):
    """Test tool execution with string arguments"""
    result = _mcp_tool().execute('{"id": "123"}')

    assert result == {"id": "123", "name": "Test User"}
    assert mcp_calls[0]["arguments"] == {"id": "123"}


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
async def test_langchain_tool_execution(mcp_calls):
    """Test execution of converted LangChain tools"""
    langchain_tool = Tools(tools=[_mcp_tool()]).to_langchain()[0]

    result = langchain_tool._run(id="test_value")

    assert result == {"id": "test_value", "name": "Test User"}
    assert len(mcp_calls) == 1


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
        _execute_config=ExecuteConfig(headers={}, name="second_test_tool"),
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


class TestToolParametersRequired:
    """`required` is a declared field, so a hand-built tool's `required=[...]` type-checks."""

    def test_it_defaults_to_none_and_is_left_out_of_a_dump(self):
        parameters = ToolParameters(type="object", properties={})
        assert parameters.required is None
        assert "required" not in parameters.model_dump()

    @pytest.mark.parametrize("served", [["b", "a"], None, "id", [1]])
    def test_a_served_value_is_kept_verbatim_even_when_malformed(self, served):
        parameters = ToolParameters(type="object", properties={}, required=served)
        assert parameters.required == served
        assert parameters.model_dump()["required"] == served

    def test_the_migration_recipe_runs(self):
        class GetEmployee(StackOneTool):
            def __init__(self) -> None:
                super().__init__(
                    description="Get an employee",
                    parameters=ToolParameters(
                        type="object", properties={"id": {"type": "string"}}, required=["id"]
                    ),
                    _execute_config=ExecuteConfig(name="get_employee"),
                )

            def execute(self, arguments=None):
                return {"id": dict(arguments or {})["id"]}

        tool = GetEmployee()
        assert tool.name == "get_employee"
        assert tool.execute({"id": "e1"}) == {"id": "e1"}
        assert tool.to_openai_function()["function"]["parameters"]["required"] == ["id"]


class TestExecuteConfig:
    """ExecuteConfig carries only what an MCP call uses."""

    def test_defaults(self):
        config = ExecuteConfig(name="test")
        assert config.headers == {}
        assert config.timeout == 60.0

    @pytest.mark.parametrize("field", ["method", "url", "body_type", "parameter_locations"])
    def test_http_fields_are_refused_not_ignored(self, field):
        """Silently ignoring them would let a caller believe a URL or method took effect."""
        with pytest.raises(ValidationError):
            ExecuteConfig(name="test", **{field: "x"})


class TestStackOneToolExecution:
    """Test StackOneTool execution edge cases"""

    def test_account_id_in_headers(self, mcp_calls):
        """Test account ID is added to headers"""
        _mcp_tool(account_id="acc123").execute({})
        assert mcp_calls[0]["headers"]["x-account-id"] == "acc123"

    def test_no_account_id_sends_no_account_header(self, mcp_calls):
        _mcp_tool().execute({})
        assert "x-account-id" not in mcp_calls[0]["headers"]

    def test_set_account_id_retargets_the_call(self, mcp_calls):
        tool = _mcp_tool(account_id="acc1")
        tool.set_account_id("acc2")
        tool.execute({})
        assert mcp_calls[0]["headers"]["x-account-id"] == "acc2"

    def test_sdk_headers_are_set_after_caller_headers(self, mcp_calls):
        """Configured headers cannot replace the credential or retarget the account."""
        tool = StackOneMcpTool(
            name="t",
            description="",
            parameters=ToolParameters(type="object", properties={}),
            api_key="test_key",
            endpoint="https://api.example.com/mcp",
            account_id="acc1",
            headers={
                "authorization": "Bearer stolen",
                " X-Account-Id ": "victim",
                "USER-AGENT": "spoof",
                "X-Trace": "abc",
            },
        )
        tool.execute({})
        headers = mcp_calls[0]["headers"]
        assert headers["X-Trace"] == "abc"
        assert headers["Authorization"].startswith("Basic ")
        assert headers["x-account-id"] == "acc1"
        assert headers["User-Agent"].startswith("stackone-ai-python/")
        assert {name.strip().lower() for name in headers} == {
            "x-trace",
            "authorization",
            "x-account-id",
            "user-agent",
        }
        assert list(headers)[-3:] == ["User-Agent", "Authorization", "x-account-id"]

    def test_the_timeout_reaches_the_call(self, mcp_calls):
        tool = StackOneMcpTool(
            name="t",
            description="",
            parameters=ToolParameters(type="object", properties={}),
            api_key="k",
            endpoint="https://api.example.com/mcp",
            account_id=None,
            timeout=7.5,
        )
        tool.execute({})
        assert mcp_calls[0]["timeout"] == 7.5

    def test_invalid_json_arguments(self):
        """Test invalid JSON string raises ValueError"""
        with pytest.raises(ValueError, match="Invalid JSON"):
            _mcp_tool().execute("not valid json")

    def test_non_dict_arguments(self):
        """Test non-dict JSON raises ValueError"""
        with pytest.raises(ValueError, match="Tool arguments must be a JSON object"):
            _mcp_tool().execute("[1, 2, 3]")

    @given(invalid_json=invalid_json_strategy)
    @settings(max_examples=50)
    def test_invalid_json_arguments_pbt(self, invalid_json: str):
        """PBT: Test various invalid JSON strings raise ValueError."""
        with pytest.raises(ValueError, match="Invalid JSON"):
            _mcp_tool().execute(invalid_json)

    @given(non_dict_json=non_dict_json_strategy)
    @settings(max_examples=50)
    def test_non_dict_arguments_pbt(self, non_dict_json: str):
        """PBT: Test non-dict JSON values raise ValueError."""
        with pytest.raises(ValueError, match="Tool arguments must be a JSON object"):
            _mcp_tool().execute(non_dict_json)

    def test_api_error_propagates(self, monkeypatch):
        def reject(*_args, **_kwargs):
            raise StackOneAPIError("Tool failed", 400, {"error": "Bad request"})

        monkeypatch.setattr("stackone_ai.tools.call_mcp_tool", reject)
        with pytest.raises(StackOneAPIError) as exc_info:
            _mcp_tool().execute({"id": "123"})
        assert exc_info.value.status_code == 400
        assert exc_info.value.response_body == {"error": "Bad request"}


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
            _execute_config=ExecuteConfig(headers={}, name="test"),
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
    async def test_arun_method(self, mcp_calls):
        """Test async _arun method"""
        lc_tool = _mcp_tool().to_langchain()

        result = await lc_tool._arun(id="123")
        assert result == {"id": "123", "name": "Test User"}


class TestStackOneToolAccountId:
    """Test account ID methods"""

    def test_set_and_get_account_id(self):
        """Test setting and getting account ID"""
        tool = StackOneTool(
            description="Test",
            parameters=ToolParameters(type="object", properties={}),
            _execute_config=ExecuteConfig(
                headers={},
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
            _execute_config=ExecuteConfig(headers={}, name="schema_tool"),
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

    def test_strips_internal_nullable_marker_without_deriving_required(self):
        """`nullable` is an SDK-internal marker; `required` comes only from the root."""
        tool = self._tool(
            {
                "needed": {"type": "string", "nullable": False},
                "optional": {"type": "string", "nullable": True},
            }
        )

        params = tool.to_openai_function()["function"]["parameters"]

        assert "nullable" not in params["properties"]["needed"]
        assert "nullable" not in params["properties"]["optional"]
        assert "required" not in params

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

    def test_preserves_field_named_nullable(self):
        """A field legitimately named 'nullable' must not be stripped."""
        tool = self._tool(
            {
                "nullable": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}},
                    "nullable": False,
                }
            }
        )

        props = tool.to_openai_function()["function"]["parameters"]["properties"]
        assert "nullable" in props
        assert props["nullable"]["type"] == "object"


class TestExecuteOpenAIToolCalls:
    """Tools.execute_openai_tool_calls: model tool calls in, `tool` messages out."""

    @staticmethod
    def _tools(monkeypatch, behaviour):
        tool = StackOneMcpTool(
            name="linear_list_issues",
            description="",
            parameters=ToolParameters(type="object", properties={"body_variables": {"type": "object"}}),
            api_key="k",
            endpoint="https://api.example.com/mcp",
            account_id="acc1",
        )
        monkeypatch.setattr(StackOneMcpTool, "execute", lambda _self, arguments=None: behaviour(arguments))
        return Tools([tool])

    def test_runs_each_call_and_pairs_the_result_with_its_id(self, monkeypatch):
        seen: list[object] = []
        tools = self._tools(monkeypatch, lambda args: seen.append(args) or {"data": {"n": 1}})

        messages = tools.execute_openai_tool_calls(
            [
                {
                    "id": "call_1",
                    "function": {"name": "linear_list_issues", "arguments": '{"body_variables": {}}'},
                }
            ]
        )

        assert messages == [{"role": "tool", "tool_call_id": "call_1", "content": '{"data": {"n": 1}}'}]
        assert seen == ['{"body_variables": {}}']

    def test_accepts_openai_sdk_objects(self, monkeypatch):
        from types import SimpleNamespace

        tools = self._tools(monkeypatch, lambda _args: {"ok": True})
        call = SimpleNamespace(
            id="call_2", function=SimpleNamespace(name="linear_list_issues", arguments="{}")
        )
        assert tools.execute_openai_tool_calls([call])[0]["tool_call_id"] == "call_2"

    def test_a_failed_call_is_reported_to_the_model_not_raised(self, monkeypatch):
        def reject(_args):
            raise StackOneAPIError("400 Bad Request", 400, {"message": "path.id is missing"})

        tools = self._tools(monkeypatch, reject)
        [message] = tools.execute_openai_tool_calls(
            [{"id": "c", "function": {"name": "linear_list_issues", "arguments": "{}"}}]
        )
        assert "path.id is missing" in message["content"]

    def test_an_unknown_tool_is_reported_not_raised(self, monkeypatch):
        tools = self._tools(monkeypatch, lambda _args: {})
        [message] = tools.execute_openai_tool_calls(
            [{"id": "c", "function": {"name": "invented_tool", "arguments": "{}"}}]
        )
        assert "Unknown tool" in message["content"]

    def test_non_text_content_parts_serialise_instead_of_crashing(self, monkeypatch):
        """Non-text MCP content parts are objects, which json.dumps cannot encode."""
        import json

        from mcp.types import ImageContent

        image = ImageContent(type="image", data="AAAA", mimeType="image/png")
        tools = self._tools(monkeypatch, lambda _args: {"a": 1, "content_parts": [image]})
        [message] = tools.execute_openai_tool_calls(
            [{"id": "c", "function": {"name": "linear_list_issues", "arguments": "{}"}}]
        )
        content = json.loads(message["content"])
        assert content["a"] == 1
        assert content["content_parts"] == [str(image)]

    def test_a_download_link_is_passed_through(self, monkeypatch):
        import json

        link = {
            "download_url": "https://downloads.example.com/f/1",
            "expires_at": "2026-01-01T00:00:00.000Z",
            "file": {"name": "a.pdf", "content_type": "application/pdf", "content_length": 3},
        }
        tools = self._tools(monkeypatch, lambda _args: link)
        [message] = tools.execute_openai_tool_calls(
            [{"id": "c", "function": {"name": "linear_list_issues", "arguments": "{}"}}]
        )
        assert json.loads(message["content"]) == link

    def test_no_tool_calls_is_no_messages(self, monkeypatch):
        tools = self._tools(monkeypatch, lambda _args: {})
        assert tools.execute_openai_tool_calls(None) == []
