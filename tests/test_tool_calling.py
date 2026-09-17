"""Tests for tool calling functionality"""

import json

import httpx
import pytest
import respx

from stackone_ai import StackOneTool
from stackone_ai.tools import StackOneRpcTool
from stackone_ai.types import (
    ExecuteConfig,
    ToolParameters,
    filename_from_content_disposition,
    is_json_content_type,
)
from tests.conftest import TEST_BASE_URL


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
    """Test StackOneRpcTool functionality"""

    @pytest.fixture
    def rpc_tool(self):
        """Create a mock RPC tool for testing"""
        parameters = ToolParameters(
            type="object",
            properties={
                "employee_id": {"type": "string", "description": "Employee ID"},
            },
        )
        return StackOneRpcTool(
            name="hibob_get_employee",
            description="Get employee details",
            parameters=parameters,
            api_key="test_api_key",
            base_url=TEST_BASE_URL,
            account_id="test_account",
        )

    @respx.mock
    def test_execute_basic(self, rpc_tool):
        """Test basic RPC tool execution"""
        route = respx.post(f"{TEST_BASE_URL}/actions/rpc").mock(
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
        route = respx.post(f"{TEST_BASE_URL}/actions/rpc").mock(
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
        route = respx.post(f"{TEST_BASE_URL}/actions/rpc").mock(
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
        route = respx.post(f"{TEST_BASE_URL}/actions/rpc").mock(
            return_value=httpx.Response(200, json={"success": True})
        )

        result = rpc_tool.execute({"path": {"id": "emp123"}})

        assert result == {"success": True}
        body = json.loads(route.calls[0].request.content)
        assert body["path"] == {"id": "emp123"}

    @respx.mock
    def test_execute_with_query_payload(self, rpc_tool):
        """Test RPC tool execution with query parameters"""
        route = respx.post(f"{TEST_BASE_URL}/actions/rpc").mock(
            return_value=httpx.Response(200, json={"success": True})
        )

        result = rpc_tool.execute({"query": {"limit": "10", "offset": "0"}})

        assert result == {"success": True}
        body = json.loads(route.calls[0].request.content)
        assert body["query"] == {"limit": "10", "offset": "0"}

    @respx.mock
    def test_execute_with_headers_payload(self, rpc_tool):
        """Test RPC tool execution with custom headers"""
        route = respx.post(f"{TEST_BASE_URL}/actions/rpc").mock(
            return_value=httpx.Response(200, json={"success": True})
        )

        result = rpc_tool.execute({"headers": {"X-Custom-Header": "custom_value"}})

        assert result == {"success": True}
        body = json.loads(route.calls[0].request.content)
        assert "X-Custom-Header" not in body["headers"]  # undeclared by the served schema
        assert body["headers"]["x-account-id"] == "test_account"

    @respx.mock
    def test_execute_headers_strips_authorization(self, rpc_tool):
        """Test that Authorization header is stripped from action headers"""
        route = respx.post(f"{TEST_BASE_URL}/actions/rpc").mock(
            return_value=httpx.Response(200, json={"success": True})
        )

        result = rpc_tool.execute({"headers": {"Authorization": "Bearer token", "X-Other": "value"}})

        assert result == {"success": True}
        body = json.loads(route.calls[0].request.content)
        assert "Authorization" not in body["headers"]
        assert "X-Other" not in body["headers"]  # undeclared by the served schema

    @respx.mock
    @pytest.mark.parametrize(
        "header_name",
        ["authorization", "AUTHORIZATION", "AuThOrIzAtIon"],
    )
    def test_execute_headers_strips_authorization_any_case(self, rpc_tool, header_name):
        """Reserved headers are stripped case-insensitively.

        HTTP header names are case-insensitive, and tool arguments are model-controlled,
        so a case variant must not smuggle a credential into the RPC envelope.
        """
        route = respx.post(f"{TEST_BASE_URL}/actions/rpc").mock(
            return_value=httpx.Response(200, json={"success": True})
        )

        rpc_tool.execute({"headers": {header_name: "Bearer attacker-token"}})

        body = json.loads(route.calls[0].request.content)
        assert all(key.lower() != "authorization" for key in body["headers"])

    @respx.mock
    def test_execute_headers_cannot_override_account_id(self, rpc_tool):
        """A tool call must not be able to retarget another account.

        x-account-id scopes the request to a tenant; letting model-supplied headers
        override it would allow lateral movement across every account the key reaches.
        """
        route = respx.post(f"{TEST_BASE_URL}/actions/rpc").mock(
            return_value=httpx.Response(200, json={"success": True})
        )

        rpc_tool.execute({"headers": {"x-account-id": "victim_account"}})

        body = json.loads(route.calls[0].request.content)
        assert body["headers"]["x-account-id"] == "test_account"

    @respx.mock
    def test_execute_headers_skips_none_values(self, rpc_tool):
        """Test that None header values are skipped"""
        route = respx.post(f"{TEST_BASE_URL}/actions/rpc").mock(
            return_value=httpx.Response(200, json={"success": True})
        )

        result = rpc_tool.execute({"headers": {"X-Present": "value", "X-Absent": None}})

        assert result == {"success": True}
        body = json.loads(route.calls[0].request.content)
        assert "X-Present" not in body["headers"]  # undeclared by the served schema
        assert "X-Absent" not in body["headers"]

    @respx.mock
    def test_execute_without_account_id_sends_no_account_anywhere(self):
        """A tool built with no account scopes nothing — and the server refuses it.

        This used to assert only that the envelope omitted x-account-id, which is the
        same shape as the assertion that pinned the bug that shipped: a green test
        recording what the client happened to send. The point worth pinning is that
        an unscoped request is not a usable request, so the HTTP header is checked
        too — that is the one the API actually reads.
        """
        parameters = ToolParameters(
            type="object",
            properties={},
        )
        tool = StackOneRpcTool(
            name="test_tool",
            description="Test",
            parameters=parameters,
            api_key="test_key",
            base_url=TEST_BASE_URL,
            account_id=None,
        )

        route = respx.post(f"{TEST_BASE_URL}/actions/rpc").mock(
            return_value=httpx.Response(200, json={"success": True})
        )

        result = tool.execute({})

        assert result == {"success": True}
        request = route.calls[0].request
        body = json.loads(request.content)
        assert "x-account-id" not in body["headers"]
        # The header the API reads. The mock server 400s when it is absent, which is
        # what makes tests/test_fetch_tools.py::TestRpcToolExecution meaningful.
        assert "x-account-id" not in request.headers

    @respx.mock
    def test_execute_with_none_arguments(self, rpc_tool):
        """Test RPC tool execution with None arguments"""
        route = respx.post(f"{TEST_BASE_URL}/actions/rpc").mock(
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

    def test_split_envelope_params_routes_flat_prefixed_keys(self, rpc_tool):
        """flat_prefixed keys are bucketed into the RPC envelope by their location prefix"""
        actual = rpc_tool._split_envelope_params(
            {
                "path_id": "123",
                "query_limit": 10,
                "headers_x-custom": "value",
                "body_name": "test",
            }
        )
        assert actual["path"] == {"id": "123"}
        assert actual["query"] == {"limit": 10}
        assert actual["headers"] == {"x-custom": "value"}
        assert actual["body"] == {"name": "test"}

    def test_split_envelope_params_buckets_nested_and_unprefixed_keys(self, rpc_tool):
        """Bare nested envelopes are accepted and unprefixed keys fall through to the body"""
        actual = rpc_tool._split_envelope_params(
            {
                "body": {"nested": "value"},
                "path": {"id": "1"},
                "extra": "x",
            }
        )
        assert actual["path"] == {"id": "1"}
        assert actual["query"] == {}
        assert actual["body"] == {"nested": "value", "extra": "x"}


class TestBinaryDownloadResponse:
    """File-download actions return raw bytes + metadata instead of failing on JSON parsing.

    The StackOne API serves file downloads as raw binary with the file's own MIME type
    (e.g. application/pdf) and a Content-Disposition header - never JSON. The returned
    shape mirrors the StackOne generated SDKs' download response (content + content_type +
    status_code + headers), with content as raw bytes (the Python analog of the Java
    client's byte[] body / the TypeScript client's response stream).
    """

    @respx.mock
    def test_binary_response_returns_content_dict(self, mock_tool):
        """A non-JSON (binary) body is returned as bytes + metadata, not JSON-parsed."""
        # Leading bytes of a real PDF; the 0xc4 byte is invalid UTF-8 and is exactly
        # what makes the unconditional response.json() raise UnicodeDecodeError.
        pdf_bytes = b"%PDF-1.4\n%\xc4\xe5\xf2\xe5\xeb\xa7\xf3\xa0\xd0\xc4\xc6\n1 0 obj\n"
        respx.post("https://api.example.com/test").mock(
            return_value=httpx.Response(
                200,
                headers={
                    "content-type": "application/pdf",
                    "content-disposition": 'attachment; filename="download.pdf"',
                },
                content=pdf_bytes,
            )
        )

        result = mock_tool.execute({"name": "report", "value": 1})

        assert result["content"] == pdf_bytes
        assert result["content_type"] == "application/pdf"
        assert result["status_code"] == 200
        assert result["file_name"] == "download.pdf"
        assert result["headers"]["content-type"] == "application/pdf"

    @respx.mock
    def test_rpc_download_action_returns_content_dict(self):
        """The RPC download path (e.g. googledrive_unified_download_file) returns bytes.

        Reproduces the reported failure: a download action invoked through /actions/rpc
        previously raised UnicodeDecodeError because the binary body was JSON-parsed.
        """
        parameters = ToolParameters(
            type="object",
            properties={"id": {"type": "string", "description": "File ID"}},
        )
        tool = StackOneRpcTool(
            name="googledrive_unified_download_file",
            description="Download a file",
            parameters=parameters,
            api_key="test_api_key",
            base_url=TEST_BASE_URL,
            account_id="test_account",
        )

        rtf_bytes = b"{\\rtf1\\ansi\\ansicpg1252\\\xc4\xe5 hello}"
        respx.post(f"{TEST_BASE_URL}/actions/rpc").mock(
            return_value=httpx.Response(
                200,
                headers={
                    "content-type": "application/rtf",
                    "content-disposition": 'attachment; filename="download.rtf"',
                },
                content=rtf_bytes,
            )
        )

        result = tool.execute({"path": {"id": "file-123"}})

        assert result["content"] == rtf_bytes
        assert result["content_type"] == "application/rtf"
        assert result["file_name"] == "download.rtf"

    @respx.mock
    def test_octet_stream_without_filename(self, mock_tool):
        """A binary body with no Content-Disposition still returns content with file_name=None."""
        blob = b"\x00\x01\x02\xc4\xff\xfe"
        respx.post("https://api.example.com/test").mock(
            return_value=httpx.Response(
                200,
                headers={"content-type": "application/octet-stream"},
                content=blob,
            )
        )

        result = mock_tool.execute({})

        assert result["content"] == blob
        assert result["content_type"] == "application/octet-stream"
        assert result["file_name"] is None

    @respx.mock
    def test_json_response_still_parsed(self, mock_tool):
        """Regression guard: JSON responses are unchanged - parsed to a dict, not wrapped."""
        respx.post("https://api.example.com/test").mock(
            return_value=httpx.Response(200, json={"id": "123", "ok": True})
        )

        result = mock_tool.execute({"name": "x", "value": 1})

        assert result == {"id": "123", "ok": True}
        assert "content" not in result

    @respx.mock
    def test_json_with_charset_param_still_parsed(self, mock_tool):
        """A JSON Content-Type with parameters (charset) is still parsed as JSON."""
        respx.post("https://api.example.com/test").mock(
            return_value=httpx.Response(
                200,
                headers={"content-type": "application/json; charset=utf-8"},
                content=b'{"ok": true}',
            )
        )

        result = mock_tool.execute({})

        assert result == {"ok": True}

    @respx.mock
    def test_missing_content_type_returns_bytes(self, mock_tool):
        """A body with no Content-Type is treated as opaque content (bytes), not JSON.

        Pins the deliberate contract: the SDK trusts Content-Type to decide JSON vs
        file, so an absent Content-Type is returned as raw bytes rather than risking
        a UTF-8/JSON decode of binary. (StackOne always labels JSON as application/json.)
        """
        blob = b"\xff\xd8\xff\xe0\x00\x10JFIF"  # JPEG magic bytes, no content-type
        respx.post("https://api.example.com/test").mock(return_value=httpx.Response(200, content=blob))

        result = mock_tool.execute({})

        assert result["content"] == blob
        assert result["content_type"] == "application/octet-stream"
        assert result["file_name"] is None


class TestResponseHelpers:
    """Unit tests for the Content-Type and Content-Disposition helpers."""

    @pytest.mark.parametrize(
        ("content_type", "expected"),
        [
            ("application/json", True),
            ("application/json; charset=utf-8", True),
            ("APPLICATION/JSON", True),
            ("application/problem+json", True),
            ("application/vnd.api+json", True),
            ("", False),
            ("application/pdf", False),
            ("application/octet-stream", False),
            ("text/plain", False),
            ("text/json-but-not-really", False),
        ],
    )
    def test_is_json_content_type(self, content_type, expected):
        assert is_json_content_type(content_type) is expected

    @pytest.mark.parametrize(
        ("header", "expected"),
        [
            ('attachment; filename="download.pdf"', "download.pdf"),
            ("attachment; filename=download.pdf", "download.pdf"),
            ('inline; filename="my report.docx"', "my report.docx"),
            # RFC 5987 extended form is percent-decoded and takes precedence.
            ("attachment; filename=\"fallback.txt\"; filename*=UTF-8''na%C3%AFve.txt", "naïve.txt"),
            # Non-UTF-8 charset is honoured: 0xA3 is "£" in ISO-8859-1, not UTF-8.
            ("attachment; filename*=ISO-8859-1'en'%A3%20rates.txt", "£ rates.txt"),
            # Unknown charset label falls back to UTF-8 instead of raising.
            ("attachment; filename*=bogus-charset''%C2%A3.txt", "£.txt"),
            # Non-conformant quoted extended value: surrounding quotes are stripped.
            ("attachment; filename*=\"UTF-8''na%C3%AFve.txt\"", "naïve.txt"),
            ("attachment", None),
            (None, None),
            ("", None),
        ],
    )
    def test_filename_from_content_disposition(self, header, expected):
        assert filename_from_content_disposition(header) == expected


class TestEnvelopeSplitIsSchemaAware:
    """The prefix pattern alone cannot tell a path param from a body field that
    happens to start with "path_". The served schema settles it."""

    @pytest.fixture
    def rpc_tool(self):
        return StackOneRpcTool(
            name="test_action",
            description="Test",
            parameters=ToolParameters(type="object", properties={}),
            api_key="test_api_key",
            base_url=TEST_BASE_URL,
            account_id="test-account",
        )

    def test_a_bare_schema_keeps_prefix_lookalikes_in_the_body(self, rpc_tool):
        """No declared key is location-prefixed, so `path_to_file` is a real field name.

        Splitting it would send the server a path component it has no use for and drop
        the argument the model supplied — silently.
        """
        actual = rpc_tool._split_envelope_params({"path_to_file": "/tmp/x"}, {"path_to_file", "name"})
        assert actual["body"] == {"path_to_file": "/tmp/x"}
        assert actual["path"] == {}

    def test_a_prefixed_schema_splits_every_match(self, rpc_tool):
        """Under flat_prefixed every parameter is prefixed, so a match really is located."""
        actual = rpc_tool._split_envelope_params(
            {"path_id": "1", "query_offset": 10}, {"path_id", "query_limit"}
        )
        assert actual["path"] == {"id": "1"}
        # Undeclared but prefixed: the model may be working from a newer schema than the
        # cached listing. Routing it to the body would silently drop the argument.
        assert actual["query"] == {"offset": 10}

    def test_a_declared_reserved_word_is_a_field_not_a_container(self, rpc_tool):
        """A served property literally named `query` must not be rejected as malformed."""
        actual = rpc_tool._split_envelope_params({"query": "sales"}, {"query", "id"})
        assert actual["body"] == {"query": "sales"}

    def test_no_schema_trusts_every_match(self, rpc_tool):
        """Direct callers without a schema keep the old, purely pattern-based behaviour."""
        actual = rpc_tool._split_envelope_params({"path_to_file": "/tmp/x"})
        assert actual["path"] == {"to_file": "/tmp/x"}

    def test_scalar_under_a_reserved_key_is_rejected_not_dropped(self, rpc_tool):
        with pytest.raises(ValueError, match="envelope container"):
            rpc_tool._split_envelope_params({"query": "not-an-object"})

    def test_precedence_does_not_depend_on_caller_key_order(self, rpc_tool):
        """flat_prefixed beats nested beats bare, whatever order the dict is built in."""
        forwards = rpc_tool._split_envelope_params({"body_foo": 1, "foo": 2})
        backwards = rpc_tool._split_envelope_params({"foo": 2, "body_foo": 1})
        assert forwards["body"] == backwards["body"] == {"foo": 1}

        nested_first = rpc_tool._split_envelope_params({"body": {"foo": 9}, "foo": 2})
        bare_first = rpc_tool._split_envelope_params({"foo": 2, "body": {"foo": 9}})
        assert nested_first["body"] == bare_first["body"] == {"foo": 9}

    def test_empty_schema_falls_back_to_trusting_prefixes(self, rpc_tool):
        """An empty declared set means "no schema", not "nothing is declared".

        Treating it as an allowlist would route every path_* key into the body and
        silently drop every path parameter.
        """
        assert rpc_tool._split_envelope_params({"path_id": "1"}, set())["path"] == {"id": "1"}


class TestMcpToolHeaderGuard:
    """The header guard on the MCP path — the one search()/execute() actually use.

    This had no coverage at all: the whole `_sanitise_headers` call could be deleted
    from StackOneMcpTool.execute and every test still passed. Every existing header
    test drives the RPC tool only.
    """

    @pytest.fixture
    def mcp_tool(self):
        from stackone_ai.tools import StackOneMcpTool

        return StackOneMcpTool(
            name="linear_acct_execute_action",
            description="Execute",
            parameters=ToolParameters(type="object", properties={}),
            api_key="test_key",
            endpoint="https://api.example.com/mcp",
            headers={"Authorization": "Basic real", "x-account-id": "real-account"},
            account_id="real-account",
        )

    @staticmethod
    def _capture(monkeypatch):
        seen: dict[str, object] = {}

        def fake_call(endpoint, headers, name, arguments):
            seen["arguments"] = arguments
            return {"ok": True}

        monkeypatch.setattr("stackone_ai.tools.call_mcp_tool", fake_call)
        return seen

    def test_undeclared_headers_are_all_dropped(self, mcp_tool, monkeypatch):
        """An allowlist, not a denylist: a two-name denylist let Proxy-Authorization,
        x-stackone-account-id, Cookie and X-Api-Key through. No served action declares
        a headers_* property, so nothing model-supplied belongs here."""
        seen = self._capture(monkeypatch)
        mcp_tool.execute(
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
    def test_whitespace_and_case_variants_do_not_slip_past(self, mcp_tool, monkeypatch, name):
        seen = self._capture(monkeypatch)
        mcp_tool.execute({"action_id": "a", "headers": {name: "stolen"}})
        assert seen["arguments"]["headers"] == {}

    @pytest.mark.parametrize("value", ["a\r\nEvil: 1", "trailing\n", "bad\rvalue"])
    def test_crlf_injection_is_rejected(self, mcp_tool, monkeypatch, value):
        """`$` also matches before a trailing newline, so this needs fullmatch."""
        seen = self._capture(monkeypatch)
        mcp_tool.execute({"action_id": "a", "headers": {"X-Probe": value}})
        assert seen["arguments"]["headers"] == {}

    def test_a_header_the_served_schema_declares_survives(self, monkeypatch):
        """The allowlist is the schema itself, so a future action needing a header
        works with no SDK release."""
        from stackone_ai.tools import StackOneMcpTool

        tool = StackOneMcpTool(
            name="linear_acct_execute_action",
            description="Execute",
            parameters=ToolParameters(type="object", properties={"headers_x-trace": {"type": "string"}}),
            api_key="test_key",
            endpoint="https://api.example.com/mcp",
            headers={},
            account_id="real-account",
        )
        seen = self._capture(monkeypatch)
        tool.execute({"action_id": "a", "headers": {"X-Trace": "abc", "X-Other": "no"}})
        assert seen["arguments"]["headers"] == {"X-Trace": "abc"}
