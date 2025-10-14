"""Tests for feedback tool."""

# TODO: Remove when Python 3.9 support is dropped
from __future__ import annotations

import json
import os
from typing import Any
from unittest.mock import Mock, patch

import pytest

# Disable implicit feedback for tests BEFORE importing anything else
os.environ["STACKONE_IMPLICIT_FEEDBACK_ENABLED"] = "false"
os.environ.pop("LANGSMITH_API_KEY", None)  # Make sure no Langsmith key is set

from stackone_ai.feedback import create_feedback_tool
from stackone_ai.models import StackOneError


# Mock the implicit feedback manager globally for tests
@pytest.fixture(autouse=True)
def mock_implicit_feedback() -> Any:
    """Mock implicit feedback manager to avoid Langsmith initialization."""
    with patch("stackone_ai.implicit_feedback.get_implicit_feedback_manager") as mock_manager:
        mock_instance = Mock()
        mock_instance.record_tool_call = Mock()
        mock_manager.return_value = mock_instance
        yield mock_manager


class TestFeedbackToolValidation:
    """Test suite for feedback tool input validation."""

    def test_throws_when_account_id_missing(self) -> None:
        """Test that tool throws when account_id is missing."""
        tool = create_feedback_tool(api_key="test_key")

        with pytest.raises(StackOneError, match="account_id"):
            tool.execute({"feedback": "Great tools!", "tool_names": ["test_tool"]})

    def test_throws_when_tool_names_missing(self) -> None:
        """Test that tool throws when tool_names is missing."""
        tool = create_feedback_tool(api_key="test_key")

        with pytest.raises(StackOneError, match="tool_names"):
            tool.execute({"feedback": "Great tools!", "account_id": "acc_123456"})

    def test_throws_when_feedback_missing(self) -> None:
        """Test that tool throws when feedback is missing."""
        tool = create_feedback_tool(api_key="test_key")

        with pytest.raises(StackOneError, match="feedback"):
            tool.execute({"account_id": "acc_123456", "tool_names": ["test_tool"]})

    def test_throws_when_feedback_empty_string(self) -> None:
        """Test that tool throws when feedback is empty after trimming."""
        tool = create_feedback_tool(api_key="test_key")

        with pytest.raises(StackOneError, match="non-empty"):
            tool.execute({"feedback": "   ", "account_id": "acc_123456", "tool_names": ["test_tool"]})

    def test_throws_when_account_id_empty_string(self) -> None:
        """Test that tool throws when account_id is empty after trimming."""
        tool = create_feedback_tool(api_key="test_key")

        with pytest.raises(StackOneError, match="non-empty"):
            tool.execute({"feedback": "Great!", "account_id": "   ", "tool_names": ["test_tool"]})

    def test_throws_when_tool_names_empty_array(self) -> None:
        """Test that tool throws when tool_names is empty array."""
        tool = create_feedback_tool(api_key="test_key")

        with pytest.raises(StackOneError):
            tool.execute({"feedback": "Great!", "account_id": "acc_123456", "tool_names": []})

    def test_throws_when_tool_names_only_whitespace(self) -> None:
        """Test that tool throws when tool_names contains only whitespace strings."""
        tool = create_feedback_tool(api_key="test_key")

        with pytest.raises(StackOneError, match="At least one tool name"):
            tool.execute({"feedback": "Great!", "account_id": "acc_123456", "tool_names": ["   ", "  "]})

    def test_accepts_json_string_arguments(self) -> None:
        """Test that tool accepts JSON string as arguments."""
        tool = create_feedback_tool(api_key="test_key")

        with patch("requests.request") as mock_request:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "message": "Feedback successfully stored",
                "key": "test-key.json",
                "submitted_at": "2025-10-08T11:44:16.123Z",
                "trace_id": "test-trace-id",
            }
            mock_response.raise_for_status = Mock()
            mock_request.return_value = mock_response

            json_string = json.dumps(
                {"feedback": "Great tools!", "account_id": "acc_123456", "tool_names": ["test_tool"]}
            )

            result = tool.execute(json_string)
            assert result["message"] == "Feedback successfully stored"


class TestFeedbackToolExecution:
    """Test suite for feedback tool execution."""

    def test_submits_feedback_to_api(self) -> None:
        """Test that feedback is submitted to the API."""
        tool = create_feedback_tool(api_key="test_key")

        api_response = {
            "message": "Feedback successfully stored",
            "key": "2025-10-08T11-44-16.123Z-a3f7b2c1d4e5f6a7b8c9d0e1f2a3b4c5.json",
            "submitted_at": "2025-10-08T11:44:16.123Z",
            "trace_id": "30d37876-cb1a-4138-9225-197355e0b6c9",
        }

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

            assert result["message"] == "Feedback successfully stored"
            assert result["trace_id"] == "30d37876-cb1a-4138-9225-197355e0b6c9"

            # Verify the API was called correctly
            mock_request.assert_called_once()
            call_kwargs = mock_request.call_args[1]
            assert call_kwargs["method"] == "POST"
            assert call_kwargs["url"] == "https://api.stackone.com/ai/tool-feedback"
            assert call_kwargs["json"]["feedback"] == "Great tools!"
            assert call_kwargs["json"]["account_id"] == "acc_123456"
            assert call_kwargs["json"]["tool_names"] == ["data_export", "analytics"]

    def test_trims_whitespace_from_strings(self) -> None:
        """Test that whitespace is trimmed from string inputs."""
        tool = create_feedback_tool(api_key="test_key")

        api_response = {
            "message": "Feedback successfully stored",
            "key": "test-key.json",
            "submitted_at": "2025-10-08T11:44:16.123Z",
            "trace_id": "test-trace-id",
        }

        with patch("requests.request") as mock_request:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = api_response
            mock_response.raise_for_status = Mock()
            mock_request.return_value = mock_response

            tool.execute(
                {
                    "feedback": "  Great tools!  ",
                    "account_id": "  acc_123456  ",
                    "tool_names": ["  hris_get_employee  ", " crm_update_employee "],
                }
            )

            # Verify trimmed values were sent to API
            call_kwargs = mock_request.call_args[1]
            assert call_kwargs["json"]["feedback"] == "Great tools!"
            assert call_kwargs["json"]["account_id"] == "acc_123456"
            assert call_kwargs["json"]["tool_names"] == ["hris_get_employee", "crm_update_employee"]

    def test_filters_empty_tool_names(self) -> None:
        """Test that empty tool names are filtered out."""
        tool = create_feedback_tool(api_key="test_key")

        api_response = {
            "message": "Feedback successfully stored",
            "key": "test-key.json",
            "submitted_at": "2025-10-08T11:44:16.123Z",
            "trace_id": "test-trace-id",
        }

        with patch("requests.request") as mock_request:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = api_response
            mock_response.raise_for_status = Mock()
            mock_request.return_value = mock_response

            tool.execute(
                {
                    "feedback": "Great tools!",
                    "account_id": "acc_123456",
                    "tool_names": ["hris_get_employee", "", "  ", "crm_update_employee"],
                }
            )

            # Verify empty tool names were filtered
            call_kwargs = mock_request.call_args[1]
            assert call_kwargs["json"]["tool_names"] == ["hris_get_employee", "crm_update_employee"]

    def test_uses_custom_base_url(self) -> None:
        """Test that custom base URL is used when provided."""
        tool = create_feedback_tool(api_key="test_key", base_url="https://custom.api.com")

        api_response = {
            "message": "Feedback successfully stored",
            "key": "test-key.json",
            "submitted_at": "2025-10-08T11:44:16.123Z",
            "trace_id": "test-trace-id",
        }

        with patch("requests.request") as mock_request:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = api_response
            mock_response.raise_for_status = Mock()
            mock_request.return_value = mock_response

            tool.execute(
                {
                    "feedback": "Great tools!",
                    "account_id": "acc_123456",
                    "tool_names": ["test_tool"],
                }
            )

            call_kwargs = mock_request.call_args[1]
            assert call_kwargs["url"] == "https://custom.api.com/ai/tool-feedback"

    def test_includes_account_id_in_header(self) -> None:
        """Test that account_id is included in request header when provided."""
        tool = create_feedback_tool(api_key="test_key", account_id="acc_header_123")

        api_response = {
            "message": "Feedback successfully stored",
            "key": "test-key.json",
            "submitted_at": "2025-10-08T11:44:16.123Z",
            "trace_id": "test-trace-id",
        }

        with patch("requests.request") as mock_request:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = api_response
            mock_response.raise_for_status = Mock()
            mock_request.return_value = mock_response

            tool.execute(
                {
                    "feedback": "Great tools!",
                    "account_id": "acc_body_123",
                    "tool_names": ["test_tool"],
                }
            )

            call_kwargs = mock_request.call_args[1]
            assert call_kwargs["headers"]["x-account-id"] == "acc_header_123"


class TestFeedbackToolErrors:
    """Test suite for feedback tool error handling."""

    def test_handles_api_errors(self) -> None:
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

    def test_handles_network_errors(self) -> None:
        """Test that network errors are handled properly."""
        tool = create_feedback_tool(api_key="test_key")

        with patch("requests.request") as mock_request:
            mock_request.side_effect = Exception("Network error")

            with pytest.raises(StackOneError, match="Network error"):
                tool.execute(
                    {
                        "feedback": "Great tools!",
                        "account_id": "acc_123456",
                        "tool_names": ["test_tool"],
                    }
                )

    def test_handles_invalid_json_string(self) -> None:
        """Test that invalid JSON string is handled properly."""
        tool = create_feedback_tool(api_key="test_key")

        with pytest.raises(StackOneError, match="Invalid JSON"):
            tool.execute("not valid json")


class TestFeedbackToolIntegration:
    """Test suite for feedback tool integration with toolset."""

    def test_feedback_tool_is_included_in_toolset(self) -> None:
        """Test that feedback tool is included when loading tools."""
        from stackone_ai import StackOneToolSet

        with patch.dict("os.environ", {"STACKONE_API_KEY": "test_key"}):
            toolset = StackOneToolSet()
            tools = toolset.get_tools("meta_*")

            feedback_tool = tools.get_tool("meta_collect_tool_feedback")
            assert feedback_tool is not None
            assert feedback_tool.name == "meta_collect_tool_feedback"
            assert "feedback" in feedback_tool.description.lower()

    def test_feedback_tool_excluded_by_negative_filter(self) -> None:
        """Test that feedback tool can be excluded with negative filter."""
        from stackone_ai import StackOneToolSet

        with patch.dict("os.environ", {"STACKONE_API_KEY": "test_key"}):
            toolset = StackOneToolSet()
            tools = toolset.get_tools(["meta_*", "!meta_collect_tool_feedback"])

            feedback_tool = tools.get_tool("meta_collect_tool_feedback")
            assert feedback_tool is None

    def test_feedback_tool_has_correct_structure(self) -> None:
        """Test that feedback tool has the correct structure for AI consumption."""
        from stackone_ai import StackOneToolSet

        with patch.dict("os.environ", {"STACKONE_API_KEY": "test_key"}):
            toolset = StackOneToolSet()
            tools = toolset.get_tools("meta_collect_tool_feedback")

            feedback_tool = tools.get_tool("meta_collect_tool_feedback")
            assert feedback_tool is not None

            # Test OpenAI format
            openai_format = feedback_tool.to_openai_function()
            assert openai_format["type"] == "function"
            assert openai_format["function"]["name"] == "meta_collect_tool_feedback"
            assert "feedback" in openai_format["function"]["parameters"]["properties"]
            assert "account_id" in openai_format["function"]["parameters"]["properties"]
            assert "tool_names" in openai_format["function"]["parameters"]["properties"]

            # Test LangChain format
            langchain_tool = feedback_tool.to_langchain()
            assert langchain_tool.name == "meta_collect_tool_feedback"
            assert "feedback" in langchain_tool.description.lower()
