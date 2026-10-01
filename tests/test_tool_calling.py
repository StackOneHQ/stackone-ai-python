"""Tests for tool calling functionality"""

import logging
from typing import Any

import pytest

from stackone_ai import StackOneError, StackOneTool, ToolArgumentsError
from stackone_ai.tools import StackOneMcpTool, _header_text
from stackone_ai.types import ExecuteConfig, ToolParameters


def _mcp_tool(properties: dict[str, Any] | None = None, account_id: str | None = "real-account"):
    return StackOneMcpTool(
        name="linear_acct_execute_action",
        description="Execute",
        parameters=ToolParameters(type="object", properties=properties or {}),
        api_key="test_key",
        endpoint="https://api.example.com/mcp",
        account_id=account_id,
    )


@pytest.fixture
def seen(monkeypatch) -> dict[str, Any]:
    """Replace the MCP transport and record what each call would have sent."""
    captured: dict[str, Any] = {}

    def fake_call(endpoint, headers, name, arguments, **_kwargs):
        captured.update(endpoint=endpoint, headers=headers, name=name, arguments=arguments)
        return {"success": True, "result": "test_result"}

    monkeypatch.setattr("stackone_ai.tools.call_mcp_tool", fake_call)
    return captured


@pytest.fixture
def mock_tool():
    """An MCP tool with two declared parameters"""
    return StackOneMcpTool(
        name="test_tool",
        description="Test tool",
        parameters=ToolParameters(
            type="object",
            properties={
                "name": {"type": "string", "description": "Name parameter"},
                "value": {"type": "number", "description": "Value parameter"},
            },
        ),
        api_key="test_api_key",
        endpoint="https://api.example.com/mcp",
        account_id="acc1",
    )


class TestToolCalling:
    """Test tool calling functionality"""

    def test_call_with_kwargs(self, mock_tool, seen):
        """Test calling a tool with keyword arguments"""
        result = mock_tool.call(name="test", value=42)

        assert result == {"success": True, "result": "test_result"}
        assert seen["name"] == "test_tool"
        assert seen["endpoint"] == "https://api.example.com/mcp"
        assert seen["arguments"] == {"name": "test", "value": 42}

    def test_call_with_dict_arg(self, mock_tool, seen):
        """Test calling a tool with a dictionary argument"""
        result = mock_tool.call({"name": "test", "value": 42})

        assert result == {"success": True, "result": "test_result"}
        assert seen["arguments"] == {"name": "test", "value": 42}

    def test_call_with_json_string(self, mock_tool, seen):
        """Test calling a tool with a JSON string argument"""
        result = mock_tool.call('{"name": "test", "value": 42}')

        assert result == {"success": True, "result": "test_result"}
        assert seen["arguments"] == {"name": "test", "value": 42}

    def test_call_with_both_args_and_kwargs_raises_error(self, mock_tool):
        """Test that providing both args and kwargs raises an error"""
        with pytest.raises(ValueError, match="Cannot provide both positional and keyword arguments"):
            mock_tool.call({"name": "test"}, value=42)

    def test_call_with_multiple_args_raises_error(self, mock_tool):
        """Test that providing multiple positional arguments raises an error"""
        with pytest.raises(ValueError, match="Only one positional argument is allowed"):
            mock_tool.call({"name": "test"}, {"value": 42})

    def test_call_without_arguments(self, mock_tool, seen):
        """Test calling a tool without any arguments sends an empty object"""
        mock_tool.call()
        assert seen["arguments"] == {}

    def test_execute_with_none_arguments(self, mock_tool, seen):
        mock_tool.execute(None)
        assert seen["arguments"] == {}

    def test_parse_arguments_invalid_json(self, mock_tool):
        """Test that invalid JSON raises ValueError"""
        with pytest.raises(ValueError, match="Invalid JSON"):
            mock_tool._parse_arguments("not valid json")

    def test_parse_arguments_non_dict(self, mock_tool):
        """Test that non-dict JSON raises ValueError"""
        with pytest.raises(ValueError, match="Tool arguments must be a JSON object"):
            mock_tool._parse_arguments("[1, 2, 3]")

    def test_arguments_are_not_split_or_renamed(self, seen):
        """No envelope splitting, no prefix routing: the server maps arguments itself."""
        tool = _mcp_tool({"path_id": {"type": "string"}, "path_to_file": {"type": "string"}})
        arguments = {
            "path_id": "1",
            "path_to_file": "/tmp/x",
            "query": {"limit": 5},
            "body_foo": 1,
            "foo": 2,
            "body": {"foo": 9},
        }
        tool.execute(arguments)
        assert seen["arguments"] == arguments

    def test_the_callers_arguments_are_not_mutated(self, seen):
        tool = _mcp_tool({"headers": {"type": "object", "properties": {}}})
        arguments = {"headers": {"Authorization": "Bearer stolen"}}
        tool.execute(arguments)
        assert arguments == {"headers": {"Authorization": "Bearer stolen"}}
        assert seen["arguments"] == {"headers": {}}


class TestBaseToolHasNoExecutor:
    """The base class describes a tool; only a subclass can run one (mirrors Node's BaseTool)."""

    @pytest.fixture
    def base_tool(self):
        """No API key: the base class cannot execute, so it has nothing to authenticate."""
        return StackOneTool(
            description="Hand-built",
            parameters=ToolParameters(type="object", properties={}),
            _execute_config=ExecuteConfig(name="hand_built"),
        )

    def test_execute_raises_a_stackone_error_naming_the_fix(self, base_tool):
        with pytest.raises(StackOneError, match=r'Tool "hand_built" has no executor\. Override execute\(\)'):
            base_tool.execute({})

    def test_call_raises_too(self, base_tool):
        with pytest.raises(StackOneError, match="has no executor"):
            base_tool.call()

    def test_an_override_runs(self):
        class EchoTool(StackOneTool):
            def execute(self, arguments=None):
                return {"echo": self._parse_arguments(arguments)}

        tool = EchoTool(
            description="",
            parameters=ToolParameters(type="object", properties={}),
            _execute_config=ExecuteConfig(name="echo"),
            _api_key="k",
        )
        assert tool.call(a=1) == {"echo": {"a": 1}}
        # Still accepted and kept, for overrides that authenticate with it.
        assert tool._api_key == "k"


class TestMcpToolHeaderGuard:
    """The guard on a nested ``headers`` argument, which the model controls."""

    def test_undeclared_headers_are_all_dropped(self, seen):
        """An allowlist, not a denylist: a two-name denylist let Proxy-Authorization,
        x-stackone-account-id, Cookie and X-Api-Key through."""
        _mcp_tool().execute(
            {
                "action_id": "linear_list_issues",
                "headers": {
                    "Authorization": "Bearer stolen",
                    "Proxy-Authorization": "Basic stolen",
                    "x-account-id": "victim-account",
                    "x-stackone-account-id": "victim-account",
                    "Cookie": "session=x",
                    "X-Api-Key": "stolen",
                    "X-Anything": "nope",
                },
            }
        )
        assert seen["arguments"]["headers"] == {}

    @pytest.mark.parametrize(
        "name",
        [" authorization", "AUTHORIZATION\t", "X-Account-Id", " x-account-id "],
    )
    def test_whitespace_and_case_variants_do_not_slip_past(self, seen, name):
        _mcp_tool().execute({"action_id": "a", "headers": {name: "stolen"}})
        assert seen["arguments"]["headers"] == {}

    @pytest.mark.parametrize("name", ["x-custom\x1f", "\x1cx-custom", "x-custom\x85"])
    def test_names_are_trimmed_as_javascript_trims_them(self, seen, name, caplog):
        """str.strip() also strips \\x1c-\\x1f and \\x85, which Node keeps and refuses as malformed."""
        with caplog.at_level(logging.WARNING, logger="stackone.tools"):
            _mcp_tool({"headers": {"type": "object"}}).execute({"headers": {name: "v"}})
        assert seen["arguments"]["headers"] == {}
        assert "Dropping malformed header" in caplog.text

    @pytest.mark.parametrize("name", ["\ufeffx-custom", "x-custom\u3000", "\u00a0x-custom\u2028"])
    def test_javascript_whitespace_is_trimmed(self, seen, name):
        _mcp_tool({"headers": {"type": "object"}}).execute({"headers": {name: "v"}})
        assert seen["arguments"]["headers"] == {"x-custom": "v"}

    def test_a_byte_order_mark_does_not_hide_an_sdk_header(self, seen):
        _mcp_tool({"headers": {"type": "object"}}).execute({"headers": {"\ufeffAuthorization": "stolen"}})
        assert seen["arguments"]["headers"] == {}

    @pytest.mark.parametrize("value", ["a\r\nEvil: 1", "trailing\n", "bad\rvalue"])
    def test_crlf_injection_is_rejected(self, seen, value):
        """`$` also matches before a trailing newline, so this needs fullmatch."""
        _mcp_tool().execute({"action_id": "a", "headers": {"X-Probe": value}})
        assert seen["arguments"]["headers"] == {}

    def test_a_nested_declared_header_survives(self, seen):
        """The allowlist is the schema itself, so a future action needing a header
        works with no SDK release."""
        tool = _mcp_tool({"headers": {"type": "object", "properties": {"x-trace": {"type": "string"}}}})
        tool.execute({"headers": {"X-Trace": "abc", "X-Other": "no"}})
        assert seen["arguments"]["headers"] == {"X-Trace": "abc"}

    def test_a_flat_declaration_does_not_declare_a_nested_header(self, seen):
        """Each form is declared on its own: `headers_x-trace` allows only the flat argument."""
        tool = _mcp_tool({"headers_x-trace": {"type": "string"}})
        tool.execute({"action_id": "a", "headers": {"X-Trace": "abc"}})
        assert seen["arguments"] == {"action_id": "a", "headers": {}}

    def test_an_open_headers_object_forwards_any_header_but_the_sdks(self, seen):
        """`*_execute_action` serves `headers` with no `properties`, which declares every name."""
        tool = _mcp_tool({"headers": {"type": "object"}})
        tool.execute(
            {
                "headers": {
                    "x-custom": "kept",
                    "Authorization": "Bearer stolen",
                    " X-Account-Id ": "victim-account",
                    "user-agent": "spoofed",
                }
            }
        )
        assert seen["arguments"]["headers"] == {"x-custom": "kept"}

    def test_an_object_with_additional_properties_allowed_is_open(self, seen):
        tool = _mcp_tool({"headers": {"type": "object", "additionalProperties": {"type": "string"}}})
        tool.execute({"headers": {"x-custom": "kept"}})
        assert seen["arguments"]["headers"] == {"x-custom": "kept"}

    @pytest.mark.parametrize(
        "schema",
        [
            pytest.param({"type": "object", "additionalProperties": False}, id="closed-object"),
            pytest.param({}, id="no-type"),
            pytest.param({"additionalProperties": True}, id="no-type-additional-allowed"),
            pytest.param({"type": "string"}, id="not-an-object"),
        ],
    )
    def test_any_other_schema_without_properties_declares_no_header(self, seen, schema):
        """Only `type: "object"` not closed by `additionalProperties: false` is open, as in Node."""
        _mcp_tool({"headers": schema}).execute({"headers": {"x-custom": "dropped"}})
        assert seen["arguments"]["headers"] == {}

    @pytest.mark.parametrize("name", ["Authorization", "x-account-id", "User-Agent"])
    def test_sdk_owned_headers_are_refused_even_when_declared(self, seen, name):
        tool = _mcp_tool({"headers": {"type": "object", "properties": {name.lower(): {"type": "string"}}}})
        tool.execute({"headers": {name: "stolen"}})
        assert seen["arguments"]["headers"] == {}

    def test_none_values_are_skipped(self, seen):
        tool = _mcp_tool(
            {
                "headers": {
                    "type": "object",
                    "properties": {"x-present": {"type": "string"}, "x-absent": {"type": "string"}},
                }
            }
        )
        tool.execute({"headers": {"X-Present": "value", "X-Absent": None}})
        assert seen["arguments"]["headers"] == {"X-Present": "value"}

    def test_values_are_written_as_node_writes_them(self, seen):
        """str() sent Python's spelling: "True", "['a', 'b']", "{'k': 1}", "1.0"."""
        _mcp_tool({"headers": {"type": "object"}}).execute(
            {
                "headers": {
                    "x-bool": True,
                    "x-list": ["a", "b"],
                    "x-obj": {"k": 1},
                    "x-float": 1.0,
                    "x-num": 1.5,
                    "x-null": None,
                }
            }
        )
        assert seen["arguments"]["headers"] == {
            "x-bool": "true",
            "x-list": '["a","b"]',
            "x-obj": '{"k":1}',
            "x-float": "1",
            "x-num": "1.5",
        }

    def test_a_value_json_cannot_hold_is_an_argument_error(self, seen):
        with pytest.raises(ValueError, match="could not be encoded"):
            _mcp_tool({"headers": {"type": "object"}}).execute({"headers": {"x-set": [{"a"}]}})
        assert seen == {}

    def test_a_self_referential_header_value_is_an_argument_error(self, seen):
        """A cycle would otherwise recurse forever in _header_text, raising RecursionError."""
        cyclic: dict[str, Any] = {}
        cyclic["self"] = cyclic
        with pytest.raises(ToolArgumentsError, match="could not be encoded"):
            _mcp_tool({"headers": {"type": "object"}}).execute({"headers": {"x-cyclic": cyclic}})
        assert seen == {}

    def test_a_non_object_headers_argument_is_dropped(self, seen, caplog):
        """A string headers argument isn't a header container and isn't an ordinary field either."""
        with caplog.at_level(logging.WARNING, logger="stackone.tools"):
            _mcp_tool().execute({"headers": "not-an-object", "q": 1})
        assert seen["arguments"] == {"q": 1}
        assert "Dropping header argument 'headers' from a tool call: not an object" in caplog.text

    def test_a_list_headers_argument_is_dropped(self, seen, caplog):
        with caplog.at_level(logging.WARNING, logger="stackone.tools"):
            _mcp_tool().execute({"headers": ["a"], "q": 1})
        assert seen["arguments"] == {"q": 1}
        assert "Dropping header argument 'headers' from a tool call: not an object" in caplog.text

    def test_a_non_object_headers_argument_is_sent_as_given_when_declared_as_a_string(self, seen):
        """A schema that declares `headers` itself as a string makes it an ordinary argument."""
        _mcp_tool({"headers": {"type": "string"}}).execute({"headers": "not-an-object"})
        assert seen["arguments"] == {"headers": "not-an-object"}

    def test_a_non_object_headers_argument_is_sent_as_given_when_the_declared_type_list_excludes_object(
        self, seen
    ):
        """`["string", "null"]` never includes "object", so `headers` is an ordinary field."""
        _mcp_tool({"headers": {"type": ["string", "null"]}}).execute({"headers": "not-an-object"})
        assert seen["arguments"] == {"headers": "not-an-object"}

    def test_a_headers_object_is_still_a_container_when_the_declared_type_list_includes_object(self, seen):
        """`["object", "null"]` includes "object", so `headers` is still sanitised as a container."""
        tool = _mcp_tool({"headers": {"type": ["object", "null"]}})
        tool.execute({"headers": {"x-trace": "value"}})
        assert seen["arguments"] == {"headers": {"x-trace": "value"}}

    def test_the_request_is_scoped_to_the_tools_account(self, seen):
        tool = _mcp_tool({"headers": {"type": "object"}, "headers_x-account-id": {"type": "string"}})
        tool.execute({"headers": {"x-account-id": "victim"}, "headers_x-account-id": "victim"})
        assert seen["headers"]["x-account-id"] == "real-account"
        assert seen["arguments"] == {"headers": {}}

    def test_no_account_sends_no_account_header(self, seen):
        """A tool built with no account scopes nothing; the server refuses such a call."""
        _mcp_tool(account_id=None).execute({})
        assert "x-account-id" not in seen["headers"]


@pytest.mark.parametrize(
    ("value", "text"),
    [
        ("as is", "as is"),
        (False, "false"),
        (7, "7"),
        (-0.0, "0"),
        (0.1, "0.1"),
        (1e-7, "1e-7"),
        (1.5e-7, "1.5e-7"),
        (0.000001, "0.000001"),
        (1e20, "100000000000000000000"),
        (1e21, "1e+21"),
        (1.5e300, "1.5e+300"),
        (float("nan"), "NaN"),
        (float("-inf"), "-Infinity"),
        ({"k": [1.0, 2.5e-7, None, "é\n"]}, '{"k":[1,2.5e-7,null,"é\\n"]}'),
        ({"k": float("nan")}, '{"k":null}'),
    ],
)
def test_header_text_matches_node(value, text):
    """Expected values are what Node's headerText returns for the same input."""
    assert _header_text(value) == text


class TestFlatHeaderArguments:
    """A top-level `headers_<name>` is a header argument too, and gets the same guard."""

    @pytest.mark.parametrize("name", ["headers_x-account-id", "headers_Authorization", "headers_ user-agent"])
    def test_sdk_owned_headers_are_refused_even_when_declared(self, seen, name, caplog):
        tool = _mcp_tool({name: {"type": "string"}})
        with caplog.at_level(logging.WARNING, logger="stackone.tools"):
            tool.execute({name: "stolen", "q": 1})
        assert seen["arguments"] == {"q": 1}
        assert "it is set by the SDK" in caplog.text

    def test_an_undeclared_one_is_dropped(self, seen, caplog):
        with caplog.at_level(logging.WARNING, logger="stackone.tools"):
            _mcp_tool({"headers": {"type": "object"}}).execute({"headers_foo": "bar", "q": 1})
        assert seen["arguments"] == {"q": 1}
        assert "'headers_foo' from a tool call: it is not declared by the schema" in caplog.text

    def test_a_declared_one_is_forwarded_as_given(self, seen):
        """A declared flat header is an ordinary top-level argument: its value is not stringified."""
        _mcp_tool({"headers_foo": {"type": "string"}, "headers_n": {"type": "integer"}}).execute(
            {"headers_foo": "bar", "headers_n": 7, "q": 1}
        )
        assert seen["arguments"] == {"headers_foo": "bar", "headers_n": 7, "q": 1}

    def test_it_is_declared_only_under_its_exact_key(self, seen):
        """Matched as the schema property it is, as in Node: `headers_Foo` does not declare `headers_foo`."""
        _mcp_tool({"headers_Foo": {"type": "string"}}).execute({"headers_foo": "bar", "q": 1})
        assert seen["arguments"] == {"q": 1}

    def test_a_declared_one_with_a_malformed_value_is_dropped(self, seen):
        _mcp_tool({"headers_foo": {"type": "string"}}).execute({"headers_foo": "a\r\nInjected: 1"})
        assert seen["arguments"] == {}

    def test_a_declared_one_with_a_number_value_is_sent_as_given(self, seen):
        _mcp_tool({"headers_n": {"type": "integer"}}).execute({"headers_n": 7})
        assert seen["arguments"] == {"headers_n": 7}

    def test_a_declared_one_with_a_list_value_is_dropped(self, seen, caplog):
        with caplog.at_level(logging.WARNING, logger="stackone.tools"):
            _mcp_tool({"headers_foo": {"type": "array"}}).execute(
                {"headers_foo": ["a\r\nx-account-id: B"], "q": 1}
            )
        assert seen["arguments"] == {"q": 1}
        assert "'headers_foo' from a tool call: not a string, number or boolean" in caplog.text

    def test_a_declared_one_with_a_dict_value_is_dropped(self, seen, caplog):
        with caplog.at_level(logging.WARNING, logger="stackone.tools"):
            _mcp_tool({"headers_foo": {"type": "object"}}).execute({"headers_foo": {"a": "b"}, "q": 1})
        assert seen["arguments"] == {"q": 1}
        assert "'headers_foo' from a tool call: not a string, number or boolean" in caplog.text

    def test_non_header_arguments_are_untouched(self, seen):
        """Only `headers` and `headers_<name>` are header arguments; bare lookalikes are not."""
        arguments = {
            "x-account-id": "not-a-header",
            "authorization": "not-a-header",
            "header_foo": 1,
            "Headers_foo": 2,
            "path": {"headers_x": "y"},
            "query": {"limit": 5},
        }
        _mcp_tool().execute(arguments)
        assert seen["arguments"] == arguments


class TestDeclaredHeaderValuesAreStillValidated:
    """The allowlist runs first, so the value grammar is only reached for a DECLARED
    header — which is exactly where a model-supplied value needs checking."""

    @pytest.fixture
    def tool(self):
        return _mcp_tool(
            {"headers": {"type": "object", "properties": {"x-trace": {"type": "string"}}}}, account_id="acct"
        )

    @pytest.mark.parametrize("value", ["trailing\n", "a\r\nInjected: 1", "bad\rvalue"])
    def test_crlf_in_a_declared_header_is_dropped(self, tool, value):
        """`$` matches before a trailing newline, so this needs fullmatch, not match."""
        assert tool._sanitise_headers({"X-Trace": value}) == {}

    def test_a_clean_declared_header_survives(self, tool):
        assert tool._sanitise_headers({"X-Trace": "abc-123"}) == {"X-Trace": "abc-123"}

    def test_the_warning_says_why_a_header_was_dropped(self, tool, caplog):
        with caplog.at_level(logging.WARNING, logger="stackone.tools"):
            tool._sanitise_headers({"X-Other": "a", "Authorization": "b"})
        assert "'X-Other' from a tool call: it is not declared by the schema" in caplog.text
        assert "'Authorization' from a tool call: it is set by the SDK" in caplog.text


@pytest.mark.parametrize(
    "value", ["half an emoji \ud83d", {"a", "set"}, b"bytes", float("nan"), float("inf"), float("-inf")]
)
def test_unencodable_arguments_raise_value_error(value, seen):
    """These would otherwise fail inside the MCP client and surface as a transport error.

    NaN and Infinity are not JSON: the MCP client would send them as null.
    """
    with pytest.raises(ValueError, match="could not be encoded"):
        _mcp_tool().execute({"q": value})
    assert seen == {}


@pytest.mark.parametrize(
    "call",
    [
        pytest.param(lambda tool: tool.execute({"q": float("nan")}), id="unencodable"),
        pytest.param(lambda tool: tool.execute({"headers": {"x": {"a", "set"}}}), id="unencodable-header"),
        pytest.param(lambda tool: tool.execute("not json"), id="invalid-json"),
        pytest.param(lambda tool: tool.execute("[1, 2]"), id="not-an-object"),
        pytest.param(lambda tool: tool.call({"q": 1}, q=2), id="args-and-kwargs"),
        pytest.param(lambda tool: tool.call({}, {}), id="two-positional"),
    ],
)
def test_an_argument_error_is_a_stackone_error_and_a_value_error(call, seen):
    """`except StackOneError` catches everything the SDK raises; `except ValueError` still works."""
    with pytest.raises(ToolArgumentsError) as excinfo:
        call(_mcp_tool({"headers": {"type": "object"}}))
    assert isinstance(excinfo.value, StackOneError)
    assert isinstance(excinfo.value, ValueError)
    assert seen == {}
