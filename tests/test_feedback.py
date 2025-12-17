"""Tests for feedback tool."""

# TODO: Remove when Python 3.9 support is dropped
from __future__ import annotations

import json
import os
from unittest.mock import Mock, patch

import pytest

from stackone_ai.feedback import create_feedback_tool
from stackone_ai.models import StackOneError


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

    def test_json_string_input(self) -> None:
        """Test that JSON string input is properly parsed."""
        tool = create_feedback_tool(api_key="test_key")

        with patch("requests.request") as mock_request:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"message": "Success"}
            mock_response.raise_for_status = Mock()
            mock_request.return_value = mock_response

            json_string = json.dumps(
                {"feedback": "Great tools!", "account_id": "acc_123456", "tool_names": ["test_tool"]}
            )
            result = tool.execute(json_string)
            assert result["message"] == "Success"


class TestFeedbackToolExecution:
    """Test suite for feedback tool execution."""

    def test_single_account_execution(self) -> None:
        """Test execution with single account ID."""
        tool = create_feedback_tool(api_key="test_key")
        api_response = {"message": "Feedback successfully stored", "trace_id": "test-trace-id"}

        with patch("requests.request") as mock_request:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = api_response
            mock_response.raise_for_status = Mock()
            mock_request.return_value = mock_response

            result = tool.execute(
                {
                    "feedback": "Great tools!",
                    "account_id": "acc_123456",
                    "tool_names": ["data_export", "analytics"],
                }
            )

            assert result == api_response
            mock_request.assert_called_once()
            call_kwargs = mock_request.call_args[1]
            assert call_kwargs["method"] == "POST"
            assert call_kwargs["url"] == "https://api.stackone.com/ai/tool-feedback"
            assert call_kwargs["json"]["feedback"] == "Great tools!"
            assert call_kwargs["json"]["account_id"] == "acc_123456"
            assert call_kwargs["json"]["tool_names"] == ["data_export", "analytics"]

    def test_call_method_interface(self) -> None:
        """Test that the .call() method works correctly."""
        tool = create_feedback_tool(api_key="test_key")
        api_response = {"message": "Success", "trace_id": "test-trace-id"}

        with patch("requests.request") as mock_request:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = api_response
            mock_response.raise_for_status = Mock()
            mock_request.return_value = mock_response

            result = tool.call(
                feedback="Testing the .call() method interface.",
                account_id="acc_test004",
                tool_names=["meta_collect_tool_feedback"],
            )

            assert result == api_response
            mock_request.assert_called_once()

    def test_api_error_handling(self) -> None:
        """Test that API errors are handled properly."""
        tool = create_feedback_tool(api_key="test_key")

        with patch("requests.request") as mock_request:
            mock_response = Mock()
            mock_response.status_code = 401
            mock_response.text = '{"error": "Unauthorized"}'
            mock_response.json.return_value = {"error": "Unauthorized"}
            mock_response.raise_for_status.side_effect = Exception("401 Client Error: Unauthorized")
            mock_request.return_value = mock_response

            with pytest.raises(StackOneError):
                tool.execute(
                    {
                        "feedback": "Great tools!",
                        "account_id": "acc_123456",
                        "tool_names": ["test_tool"],
                    }
                )

    def test_multiple_account_ids_execution(self) -> None:
        """Test execution with multiple account IDs - both success and mixed scenarios."""
        tool = create_feedback_tool(api_key="test_key")
        api_response = {"message": "Feedback successfully stored", "trace_id": "test-trace-id"}

        # Test all successful case
        with patch("requests.request") as mock_request:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = api_response
            mock_response.raise_for_status = Mock()
            mock_request.return_value = mock_response

            result = tool.execute(
                {
                    "feedback": "Great tools!",
                    "account_id": ["acc_123456", "acc_789012", "acc_345678"],
                    "tool_names": ["test_tool"],
                }
            )

            assert result["message"] == "Feedback sent to 3 account(s)"
            assert result["total_accounts"] == 3
            assert result["successful"] == 3
            assert result["failed"] == 0
            assert len(result["results"]) == 3
            assert mock_request.call_count == 3

        # Test mixed success/error case
        def mock_request_side_effect(*args, **kwargs):
            account_id = kwargs.get("json", {}).get("account_id")
            if account_id == "acc_123456":
                mock_response = Mock()
                mock_response.status_code = 200
                mock_response.json.return_value = {"message": "Success"}
                mock_response.raise_for_status = Mock()
                return mock_response
            else:
                mock_response = Mock()
                mock_response.status_code = 401
                mock_response.text = '{"error": "Unauthorized"}'
                mock_response.json.return_value = {"error": "Unauthorized"}
                mock_response.raise_for_status.side_effect = Exception("401 Client Error: Unauthorized")
                return mock_response

        with patch("requests.request") as mock_request:
            mock_request.side_effect = mock_request_side_effect

            result = tool.execute(
                {
                    "feedback": "Great tools!",
                    "account_id": ["acc_123456", "acc_unauthorized"],
                    "tool_names": ["test_tool"],
                }
            )

            assert result["total_accounts"] == 2
            assert result["successful"] == 1
            assert result["failed"] == 1
            assert len(result["results"]) == 2

            success_result = next(r for r in result["results"] if r["account_id"] == "acc_123456")
            assert success_result["status"] == "success"

            error_result = next(r for r in result["results"] if r["account_id"] == "acc_unauthorized")
            assert error_result["status"] == "error"
            assert "401 Client Error: Unauthorized" in error_result["error"]

    def test_tool_integration(self) -> None:
        """Test that feedback tool integrates properly with toolset."""
        from stackone_ai.feedback import create_feedback_tool

        with patch.dict("os.environ", {"STACKONE_API_KEY": "test_key"}):
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
def test_live_feedback_submission() -> None:
    """Submit feedback to the live API and assert a successful response."""
    import uuid

    from stackone_ai.feedback import create_feedback_tool

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
            "tool_names": ["hris_list_employees"],
        }
    )

    assert isinstance(result, dict)
    assert result.get("message", "").lower().startswith("feedback")
    assert "trace_id" in result and result["trace_id"]
