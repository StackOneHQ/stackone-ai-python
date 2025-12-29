"""Tests for feedback tool."""

from __future__ import annotations

import json
import os
import string

import httpx
import pytest
import respx
from hypothesis import given, settings
from hypothesis import strategies as st

from stackone_ai.feedback import create_feedback_tool
from stackone_ai.models import StackOneError

# Hypothesis strategies for PBT
# Various whitespace characters including Unicode
WHITESPACE_CHARS = " \t\n\r\u00a0\u2003\u2009"
whitespace_strategy = st.text(alphabet=WHITESPACE_CHARS, min_size=1, max_size=20)

# Valid non-empty strings (stripped)
valid_string_strategy = st.text(
    alphabet=string.ascii_letters + string.digits + "_-",
    min_size=1,
    max_size=50,
).filter(lambda s: s.strip())

# Invalid JSON strings (strings that cannot be parsed as valid JSON at all)
# Note: Python's json module accepts NaN/Infinity by default, so avoid those
invalid_json_strategy = st.one_of(
    st.just("{incomplete"),
    st.just('{"missing": }'),
    st.just('{"key": value}'),
    st.just("[1, 2, 3"),
    st.just("{trailing}garbage"),
    st.just("{missing closing brace"),
    st.just("undefined"),
    st.just("not valid json"),
    st.just("abc123"),
    st.just("foo bar baz"),
)


class TestFeedbackToolValidation:
    """Test suite for feedback tool input validation."""

    def test_missing_required_fields(self) -> None:
        """Test validation errors for missing required fields."""
        tool = create_feedback_tool(api_key="test_key")

        with pytest.raises(StackOneError, match="account_id"):
            tool.execute({"feedback": "Great tools!", "tool_names": ["test_tool"]})

        with pytest.raises(StackOneError, match="tool_names"):
            tool.execute({"feedback": "Great tools!", "account_id": "acc_123456"})

        with pytest.raises(StackOneError, match="feedback"):
            tool.execute({"account_id": "acc_123456", "tool_names": ["test_tool"]})

    def test_empty_and_whitespace_validation(self) -> None:
        """Test validation for empty and whitespace-only strings."""
        tool = create_feedback_tool(api_key="test_key")

        with pytest.raises(StackOneError, match="non-empty"):
            tool.execute({"feedback": "   ", "account_id": "acc_123456", "tool_names": ["test_tool"]})

        with pytest.raises(StackOneError, match="non-empty"):
            tool.execute({"feedback": "Great!", "account_id": "   ", "tool_names": ["test_tool"]})

        with pytest.raises(StackOneError, match="tool_names"):
            tool.execute({"feedback": "Great!", "account_id": "acc_123456", "tool_names": []})

        with pytest.raises(StackOneError, match="At least one tool name"):
            tool.execute({"feedback": "Great!", "account_id": "acc_123456", "tool_names": ["   ", "  "]})

    def test_multiple_account_ids_validation(self) -> None:
        """Test validation with multiple account IDs."""
        tool = create_feedback_tool(api_key="test_key")

        with pytest.raises(StackOneError, match="At least one account ID is required"):
            tool.execute({"feedback": "Great tools!", "account_id": [], "tool_names": ["test_tool"]})

        with pytest.raises(StackOneError, match="At least one valid account ID is required"):
            tool.execute({"feedback": "Great tools!", "account_id": ["", "   "], "tool_names": ["test_tool"]})

    def test_invalid_account_id_type(self) -> None:
        """Test validation with invalid account ID type (not string or list)."""
        tool = create_feedback_tool(api_key="test_key")

        # Pydantic validates input types before our custom validator runs
        with pytest.raises(StackOneError, match="(account_id|Input should be a valid)"):
            tool.execute({"feedback": "Great tools!", "account_id": 12345, "tool_names": ["test_tool"]})

        with pytest.raises(StackOneError, match="(account_id|Input should be a valid)"):
            tool.execute(
                {"feedback": "Great tools!", "account_id": {"nested": "dict"}, "tool_names": ["test_tool"]}
            )

    def test_invalid_json_input(self) -> None:
        """Test that invalid JSON input raises appropriate error."""
        tool = create_feedback_tool(api_key="test_key")

        with pytest.raises(StackOneError, match="Invalid JSON"):
            tool.execute("not valid json {}")

        with pytest.raises(StackOneError, match="Invalid JSON"):
            tool.execute("{missing closing brace")

    @given(whitespace=whitespace_strategy)
    @settings(max_examples=50)
    def test_whitespace_feedback_validation_pbt(self, whitespace: str) -> None:
        """PBT: Test validation for various whitespace patterns in feedback."""
        tool = create_feedback_tool(api_key="test_key")

        with pytest.raises(StackOneError, match="non-empty"):
            tool.execute({"feedback": whitespace, "account_id": "acc_123456", "tool_names": ["test_tool"]})

    @given(whitespace=whitespace_strategy)
    @settings(max_examples=50)
    def test_whitespace_account_id_validation_pbt(self, whitespace: str) -> None:
        """PBT: Test validation for various whitespace patterns in account_id."""
        tool = create_feedback_tool(api_key="test_key")

        with pytest.raises(StackOneError, match="non-empty"):
            tool.execute({"feedback": "Great!", "account_id": whitespace, "tool_names": ["test_tool"]})

    @given(whitespace_list=st.lists(whitespace_strategy, min_size=1, max_size=5))
    @settings(max_examples=50)
    def test_whitespace_tool_names_validation_pbt(self, whitespace_list: list[str]) -> None:
        """PBT: Test validation for lists containing only whitespace tool names."""
        tool = create_feedback_tool(api_key="test_key")

        with pytest.raises(StackOneError, match="At least one tool name"):
            tool.execute({"feedback": "Great!", "account_id": "acc_123456", "tool_names": whitespace_list})

    @given(
        whitespace_list=st.lists(whitespace_strategy, min_size=1, max_size=5),
    )
    @settings(max_examples=50)
    def test_whitespace_account_ids_list_validation_pbt(self, whitespace_list: list[str]) -> None:
        """PBT: Test validation for lists containing only whitespace account IDs."""
        tool = create_feedback_tool(api_key="test_key")

        with pytest.raises(StackOneError, match="At least one valid account ID is required"):
            tool.execute(
                {
                    "feedback": "Great tools!",
                    "account_id": whitespace_list,
                    "tool_names": ["test_tool"],
                }
            )

    @given(invalid_json=invalid_json_strategy)
    @settings(max_examples=50)
    def test_invalid_json_input_pbt(self, invalid_json: str) -> None:
        """PBT: Test that various invalid JSON inputs raise appropriate error."""
        tool = create_feedback_tool(api_key="test_key")

        with pytest.raises(StackOneError, match="Invalid JSON"):
            tool.execute(invalid_json)

    @respx.mock
    def test_json_string_input(self) -> None:
        """Test that JSON string input is properly parsed."""
        tool = create_feedback_tool(api_key="test_key")

        route = respx.post("https://api.stackone.com/ai/tool-feedback").mock(
            return_value=httpx.Response(200, json={"message": "Success"})
        )

        json_string = json.dumps(
            {"feedback": "Great tools!", "account_id": "acc_123456", "tool_names": ["test_tool"]}
        )
        result = tool.execute(json_string)
        assert result == {"message": "Success"}
        assert route.called
        assert route.calls[0].response.status_code == 200


class TestFeedbackToolExecution:
    """Test suite for feedback tool execution."""

    @respx.mock
    def test_single_account_execution(self) -> None:
        """Test execution with single account ID."""
        tool = create_feedback_tool(api_key="test_key")
        api_response = {"message": "Feedback successfully stored", "trace_id": "test-trace-id"}

        route = respx.post("https://api.stackone.com/ai/tool-feedback").mock(
            return_value=httpx.Response(200, json=api_response)
        )

        result = tool.execute(
            {
                "feedback": "Great tools!",
                "account_id": "acc_123456",
                "tool_names": ["data_export", "analytics"],
            }
        )

        assert result == api_response
        assert route.called
        assert route.call_count == 1
        assert route.calls[0].response.status_code == 200
        request = route.calls[0].request
        body = json.loads(request.content)
        assert body["feedback"] == "Great tools!"
        assert body["account_id"] == "acc_123456"
        assert body["tool_names"] == ["data_export", "analytics"]

    @respx.mock
    def test_call_method_interface(self) -> None:
        """Test that the .call() method works correctly."""
        tool = create_feedback_tool(api_key="test_key")
        api_response = {"message": "Success", "trace_id": "test-trace-id"}

        route = respx.post("https://api.stackone.com/ai/tool-feedback").mock(
            return_value=httpx.Response(200, json=api_response)
        )

        result = tool.call(
            feedback="Testing the .call() method interface.",
            account_id="acc_test004",
            tool_names=["meta_collect_tool_feedback"],
        )

        assert result == api_response
        assert route.called
        assert route.call_count == 1
        assert route.calls[0].response.status_code == 200

    @respx.mock
    def test_api_error_handling(self) -> None:
        """Test that API errors are handled properly."""
        tool = create_feedback_tool(api_key="test_key")

        route = respx.post("https://api.stackone.com/ai/tool-feedback").mock(
            return_value=httpx.Response(401, json={"error": "Unauthorized"})
        )

        with pytest.raises(StackOneError):
            tool.execute(
                {
                    "feedback": "Great tools!",
                    "account_id": "acc_123456",
                    "tool_names": ["test_tool"],
                }
            )

        assert route.called
        assert route.calls[0].response.status_code == 401

    @respx.mock
    def test_multiple_account_ids_execution(self) -> None:
        """Test execution with multiple account IDs - both success and mixed scenarios."""
        tool = create_feedback_tool(api_key="test_key")
        api_response = {"message": "Feedback successfully stored", "trace_id": "test-trace-id"}

        # Test all successful case
        route = respx.post("https://api.stackone.com/ai/tool-feedback").mock(
            return_value=httpx.Response(200, json=api_response)
        )

        result = tool.execute(
            {
                "feedback": "Great tools!",
                "account_id": ["acc_123456", "acc_789012", "acc_345678"],
                "tool_names": ["test_tool"],
            }
        )

        assert result == {
            "message": "Feedback sent to 3 account(s)",
            "total_accounts": 3,
            "successful": 3,
            "failed": 0,
            "results": [
                {
                    "account_id": "acc_123456",
                    "status": "success",
                    "result": {"message": "Feedback successfully stored", "trace_id": "test-trace-id"},
                },
                {
                    "account_id": "acc_789012",
                    "status": "success",
                    "result": {"message": "Feedback successfully stored", "trace_id": "test-trace-id"},
                },
                {
                    "account_id": "acc_345678",
                    "status": "success",
                    "result": {"message": "Feedback successfully stored", "trace_id": "test-trace-id"},
                },
            ],
        }
        assert route.call_count == 3
        assert route.calls[0].response.status_code == 200
        assert route.calls[1].response.status_code == 200
        assert route.calls[2].response.status_code == 200

    @respx.mock
    def test_multiple_account_ids_mixed_success(self) -> None:
        """Test execution with multiple account IDs - mixed success and error."""
        tool = create_feedback_tool(api_key="test_key")

        def custom_side_effect(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            account_id = body.get("account_id")
            if account_id == "acc_123456":
                return httpx.Response(200, json={"message": "Success"})
            else:
                return httpx.Response(401, json={"error": "Unauthorized"})

        route = respx.post("https://api.stackone.com/ai/tool-feedback").mock(side_effect=custom_side_effect)

        result = tool.execute(
            {
                "feedback": "Great tools!",
                "account_id": ["acc_123456", "acc_unauthorized"],
                "tool_names": ["test_tool"],
            }
        )

        assert result == {
            "message": "Feedback sent to 2 account(s)",
            "total_accounts": 2,
            "successful": 1,
            "failed": 1,
            "results": [
                {
                    "account_id": "acc_123456",
                    "status": "success",
                    "result": {"message": "Success"},
                },
                {
                    "account_id": "acc_unauthorized",
                    "status": "error",
                    "error": (
                        "Client error '401 Unauthorized' for url "
                        "'https://api.stackone.com/ai/tool-feedback'\n"
                        "For more information check: "
                        "https://developer.mozilla.org/en-US/docs/Web/HTTP/Status/401"
                    ),
                },
            ],
        }
        assert route.call_count == 2
        assert route.calls[0].response.status_code == 200
        assert route.calls[1].response.status_code == 401

    def test_tool_integration(self) -> None:
        """Test that feedback tool integrates properly with toolset."""
        feedback_tool = create_feedback_tool(api_key="test_key")

        assert feedback_tool is not None
        assert feedback_tool.name == "meta_collect_tool_feedback"
        assert "feedback" in feedback_tool.description.lower()

        # Test OpenAI format
        openai_format = feedback_tool.to_openai_function()
        assert openai_format["type"] == "function"
        assert openai_format["function"]["name"] == "meta_collect_tool_feedback"
        assert "feedback" in openai_format["function"]["parameters"]["properties"]
        assert "account_id" in openai_format["function"]["parameters"]["properties"]
        assert "tool_names" in openai_format["function"]["parameters"]["properties"]


@pytest.mark.integration
@pytest.mark.skip(reason="Live integration test - requires valid API key with feedback permissions")
def test_live_feedback_submission() -> None:
    """Submit feedback to the live API and assert a successful response."""
    import uuid

    api_key = os.getenv("STACKONE_API_KEY")
    if not api_key:
        pytest.skip("STACKONE_API_KEY env var required for live feedback test")

    base_url = os.getenv("STACKONE_BASE_URL", "https://api.stackone.com")

    feedback_tool = create_feedback_tool(api_key=api_key, base_url=base_url)
    assert feedback_tool is not None, "Feedback tool must be available"

    feedback_token = uuid.uuid4().hex[:8]
    result = feedback_tool.execute(
        {
            "feedback": f"CI live test feedback {feedback_token}",
            "account_id": f"acc-ci-{feedback_token}",
            "tool_names": ["hibob_list_employees"],
        }
    )

    assert isinstance(result, dict)
    assert result.get("message", "").lower().startswith("feedback")
    assert "trace_id" in result and result["trace_id"]
