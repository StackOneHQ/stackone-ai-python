"""Tests for StackOneToolSet."""

import asyncio
import base64
import fnmatch
import os
import string
from unittest.mock import patch

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from stackone_ai.toolset import (
    StackOneToolSet,
    ToolsetConfigError,
    ToolsetError,
    ToolsetLoadError,
    _build_auth_header,
    _run_async,
)

# Hypothesis strategies for PBT
# API key strategy with printable ASCII characters
api_key_strategy = st.text(
    alphabet="".join(chr(i) for i in range(32, 127)),
    min_size=1,
    max_size=200,
)

# Tool name strategy (lowercase letters, digits, underscores)
tool_name_strategy = st.text(
    alphabet=string.ascii_lowercase + string.digits + "_",
    min_size=1,
    max_size=50,
)

# Glob pattern strategy
glob_pattern_strategy = st.text(
    alphabet=string.ascii_lowercase + string.digits + "_*?",
    min_size=1,
    max_size=50,
)

# Provider name strategy
provider_name_strategy = st.text(
    alphabet=string.ascii_lowercase,
    min_size=2,
    max_size=20,
)


class TestToolsetErrors:
    """Test toolset error classes."""

    def test_toolset_error_inheritance(self):
        """Test ToolsetError is base exception."""
        error = ToolsetError("test error")
        assert isinstance(error, Exception)
        assert str(error) == "test error"

    def test_toolset_config_error_inheritance(self):
        """Test ToolsetConfigError inherits from ToolsetError."""
        error = ToolsetConfigError("config error")
        assert isinstance(error, ToolsetError)
        assert isinstance(error, Exception)

    def test_toolset_load_error_inheritance(self):
        """Test ToolsetLoadError inherits from ToolsetError."""
        error = ToolsetLoadError("load error")
        assert isinstance(error, ToolsetError)
        assert isinstance(error, Exception)


class TestBuildAuthHeader:
    """Test _build_auth_header function."""

    def test_builds_basic_auth_header(self):
        """Test building Basic auth header from API key."""
        result = _build_auth_header("test_api_key")
        # Base64 of "test_api_key:"
        assert result.startswith("Basic ")
        assert result == "Basic dGVzdF9hcGlfa2V5Og=="

    def test_builds_auth_header_with_special_chars(self):
        """Test auth header with special characters in key."""
        result = _build_auth_header("key:with:colons")
        assert result.startswith("Basic ")

    @given(api_key=api_key_strategy)
    @settings(max_examples=100)
    def test_auth_header_format_pbt(self, api_key: str):
        """PBT: Test auth header format for various API keys."""
        result = _build_auth_header(api_key)

        # Should start with "Basic "
        assert result.startswith("Basic ")

        # Should be valid base64
        encoded_part = result.replace("Basic ", "")
        decoded = base64.b64decode(encoded_part).decode("utf-8")

        # Decoded should be "api_key:"
        assert decoded == f"{api_key}:"

    @given(api_key=api_key_strategy)
    @settings(max_examples=100)
    def test_auth_header_round_trip_pbt(self, api_key: str):
        """PBT: Test that auth header can be decoded back to original key."""
        result = _build_auth_header(api_key)
        encoded_part = result.replace("Basic ", "")
        decoded = base64.b64decode(encoded_part).decode("utf-8")

        # Should be able to extract original key (remove trailing colon)
        # The format is "api_key:" so we remove the last character
        extracted_key = decoded[:-1] if decoded.endswith(":") else decoded
        assert extracted_key == api_key


class TestRunAsync:
    """Test _run_async function."""

    def test_run_async_outside_event_loop(self):
        """Test running async function when no event loop exists."""

        async def simple_coroutine():
            return "result"

        result = _run_async(simple_coroutine())
        assert result == "result"

    def test_run_async_inside_event_loop(self):
        """Test running async function when already inside an event loop."""

        async def inner_coroutine():
            return "inner_result"

        async def outer_coroutine():
            # This simulates calling _run_async from within an event loop
            return _run_async(inner_coroutine())

        # Run the outer coroutine which calls _run_async internally
        result = asyncio.run(outer_coroutine())
        assert result == "inner_result"

    def test_run_async_propagates_exceptions(self):
        """Test that exceptions from coroutines are propagated."""

        async def failing_coroutine():
            raise ValueError("test error")

        with pytest.raises(ValueError, match="test error"):
            _run_async(failing_coroutine())

    def test_run_async_propagates_exceptions_from_thread(self):
        """Test that exceptions are propagated when running in a thread."""

        async def failing_coroutine():
            raise RuntimeError("thread error")

        async def wrapper():
            return _run_async(failing_coroutine())

        with pytest.raises(RuntimeError, match="thread error"):
            asyncio.run(wrapper())


class TestStackOneToolSetInit:
    """Test StackOneToolSet initialization."""

    def test_init_with_api_key(self):
        """Test initialization with explicit API key."""
        toolset = StackOneToolSet(api_key="test_key")
        assert toolset.api_key == "test_key"
        assert toolset.account_id is None
        assert toolset.base_url == "https://api.stackone.com"

    def test_init_with_env_api_key(self):
        """Test initialization with API key from environment."""
        with patch.dict(os.environ, {"STACKONE_API_KEY": "env_key"}):
            toolset = StackOneToolSet()
            assert toolset.api_key == "env_key"

    def test_init_without_api_key_raises(self):
        """Test that missing API key raises ToolsetConfigError."""
        with patch.dict(os.environ, {}, clear=True):
            # Ensure STACKONE_API_KEY is not set
            os.environ.pop("STACKONE_API_KEY", None)
            with pytest.raises(ToolsetConfigError, match="API key must be provided"):
                StackOneToolSet()

    def test_init_with_account_id(self):
        """Test initialization with account ID."""
        toolset = StackOneToolSet(api_key="test_key", account_id="acc123")
        assert toolset.account_id == "acc123"

    def test_init_with_custom_base_url(self):
        """Test initialization with custom base URL."""
        toolset = StackOneToolSet(api_key="test_key", base_url="https://custom.api.com")
        assert toolset.base_url == "https://custom.api.com"


class TestStackOneToolSetNormalizeSchemaProperties:
    """Test _normalize_schema_properties method."""

    def test_normalizes_properties_with_required(self):
        """Test normalizing schema with required fields."""
        toolset = StackOneToolSet(api_key="test_key")
        schema = {
            "type": "object",
            "properties": {
                "required_field": {"type": "string", "description": "Required"},
                "optional_field": {"type": "string", "description": "Optional"},
            },
            "required": ["required_field"],
        }

        result = toolset._normalize_schema_properties(schema)

        assert result["required_field"]["nullable"] is False
        assert result["optional_field"]["nullable"] is True

    def test_handles_non_dict_properties(self):
        """Test handling non-dict property values."""
        toolset = StackOneToolSet(api_key="test_key")
        schema = {
            "type": "object",
            "properties": {
                "simple_field": "string value",
            },
        }

        result = toolset._normalize_schema_properties(schema)

        assert result["simple_field"]["description"] == "string value"

    def test_handles_missing_properties(self):
        """Test handling schema without properties."""
        toolset = StackOneToolSet(api_key="test_key")
        schema = {"type": "object"}

        result = toolset._normalize_schema_properties(schema)

        assert result == {}

    def test_handles_non_dict_properties_value(self):
        """Test handling when properties is not a dict."""
        toolset = StackOneToolSet(api_key="test_key")
        schema = {
            "type": "object",
            "properties": "not a dict",
        }

        result = toolset._normalize_schema_properties(schema)

        assert result == {}


class TestStackOneToolSetBuildMcpHeaders:
    """Test _build_mcp_headers method."""

    def test_builds_headers_without_account(self):
        """Test building MCP headers without account ID."""
        toolset = StackOneToolSet(api_key="test_key")
        headers = toolset._build_mcp_headers(None)

        assert "Authorization" in headers
        assert "User-Agent" in headers
        assert "x-account-id" not in headers

    def test_builds_headers_with_account(self):
        """Test building MCP headers with account ID."""
        toolset = StackOneToolSet(api_key="test_key")
        headers = toolset._build_mcp_headers("acc123")

        assert "Authorization" in headers
        assert "User-Agent" in headers
        assert headers["x-account-id"] == "acc123"


def test_set_accounts():
    """Test setting account IDs for filtering"""
    toolset = StackOneToolSet(api_key="test_key")
    result = toolset.set_accounts(["acc1", "acc2"])

    # Should return self for chaining
    assert result is toolset
    assert toolset._account_ids == ["acc1", "acc2"]


def test_filter_by_provider():
    """Test provider filtering"""
    toolset = StackOneToolSet(api_key="test_key")

    # Test matching providers
    assert toolset._filter_by_provider("hibob_list_employees", ["hibob", "bamboohr"])
    assert toolset._filter_by_provider("bamboohr_create_job", ["hibob", "bamboohr"])

    # Test non-matching providers
    assert not toolset._filter_by_provider("workday_list_contacts", ["hibob", "bamboohr"])

    # Test case-insensitive matching
    assert toolset._filter_by_provider("HIBOB_list_employees", ["hibob"])
    assert toolset._filter_by_provider("hibob_list_employees", ["HIBOB"])


def test_filter_by_action():
    """Test action filtering with glob patterns"""
    toolset = StackOneToolSet(api_key="test_key")

    # Test exact match
    assert toolset._filter_by_action("hibob_list_employees", ["hibob_list_employees"])

    # Test glob pattern
    assert toolset._filter_by_action("hibob_list_employees", ["*_list_employees"])
    assert toolset._filter_by_action("bamboohr_list_employees", ["*_list_employees"])
    assert toolset._filter_by_action("hibob_list_employees", ["hibob_*"])
    assert toolset._filter_by_action("hibob_create_employee", ["hibob_*"])

    # Test non-matching patterns
    assert not toolset._filter_by_action("workday_list_contacts", ["*_list_employees"])
    assert not toolset._filter_by_action("bamboohr_create_job", ["hibob_*"])


@given(
    tool_name=tool_name_strategy,
    pattern=glob_pattern_strategy,
)
@settings(max_examples=100)
def test_filter_by_action_matches_fnmatch_pbt(tool_name: str, pattern: str):
    """PBT: Test that action filtering matches Python fnmatch behavior."""
    toolset = StackOneToolSet(api_key="test_key")

    result = toolset._filter_by_action(tool_name, [pattern])
    expected = fnmatch.fnmatch(tool_name, pattern)

    assert result == expected, f"Mismatch for tool='{tool_name}', pattern='{pattern}'"


@given(
    provider=provider_name_strategy,
    action=st.text(alphabet=string.ascii_lowercase + "_", min_size=1, max_size=20),
    entity=st.text(alphabet=string.ascii_lowercase, min_size=1, max_size=20),
)
@settings(max_examples=100)
def test_filter_by_provider_case_insensitive_pbt(provider: str, action: str, entity: str):
    """PBT: Test that provider filtering is case-insensitive."""
    toolset = StackOneToolSet(api_key="test_key")
    tool_name = f"{provider}_{action}_{entity}"

    # Should match regardless of case
    assert toolset._filter_by_provider(tool_name, [provider.lower()])
    assert toolset._filter_by_provider(tool_name, [provider.upper()])
    assert toolset._filter_by_provider(tool_name.upper(), [provider.lower()])
    assert toolset._filter_by_provider(tool_name.lower(), [provider.upper()])
