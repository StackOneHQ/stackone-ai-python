"""Comprehensive tests for feedback tool."""

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

    def test_validation_errors(self) -> None:
        """Test all validation error cases in one comprehensive test."""
        tool = create_feedback_tool(api_key="test_key")

        # Test missing required fields
        with pytest.raises(StackOneError, match="account_id"):
            tool.execute({"feedback": "Great tools!", "tool_names": ["test_tool"]})

        with pytest.raises(StackOneError, match="tool_names"):
            tool.execute({"feedback": "Great tools!", "account_id": "acc_123456"})

        with pytest.raises(StackOneError, match="feedback"):
            tool.execute({"account_id": "acc_123456", "tool_names": ["test_tool"]})

        # Test empty/whitespace strings
        with pytest.raises(StackOneError, match="non-empty"):
            tool.execute({"feedback": "   ", "account_id": "acc_123456", "tool_names": ["test_tool"]})

        with pytest.raises(StackOneError, match="non-empty"):
            tool.execute({"feedback": "Great!", "account_id": "   ", "tool_names": ["test_tool"]})

        with pytest.raises(StackOneError, match="tool_names"):
            tool.execute({"feedback": "Great!", "account_id": "acc_123456", "tool_names": []})

        with pytest.raises(StackOneError, match="At least one tool name"):
            tool.execute({"feedback": "Great!", "account_id": "acc_123456", "tool_names": ["   ", "  "]})

        # Test JSON string input
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

    def test_submits_feedback_to_api(self) -> None:
        """Test that feedback is submitted to the API with proper structure."""
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

    def test_call_method_works(self) -> None:
        """Test that the .call() method works correctly."""
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

            result = tool.call(
                feedback="Testing the .call() method interface.",
                account_id="acc_test004",
                tool_names=["meta_collect_tool_feedback"],
            )

            assert result == api_response
            mock_request.assert_called_once()
            call_kwargs = mock_request.call_args[1]
            assert call_kwargs["json"]["feedback"] == "Testing the .call() method interface."
            assert call_kwargs["json"]["account_id"] == "acc_test004"
            assert call_kwargs["json"]["tool_names"] == ["meta_collect_tool_feedback"]

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


class TestFeedbackToolIntegration:
    """Test suite for feedback tool integration."""

    def test_feedback_tool_integration(self) -> None:
        """Test that feedback tool integrates properly with toolset and has correct structure."""
        from stackone_ai import StackOneToolSet

        with patch.dict("os.environ", {"STACKONE_API_KEY": "test_key"}):
            toolset = StackOneToolSet()
            tools = toolset.get_tools("meta_collect_tool_feedback")

            feedback_tool = tools.get_tool("meta_collect_tool_feedback")
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

            # Test LangChain format
            langchain_tool = feedback_tool.to_langchain()
            assert langchain_tool.name == "meta_collect_tool_feedback"
            assert "feedback" in langchain_tool.description.lower()

    def test_feedback_tool_smoke(self) -> None:
        """Lightweight smoke test for basic functionality."""
        tool = create_feedback_tool(api_key="test_key")

        api_response = {
            "message": "Feedback successfully stored",
            "trace_id": "trace-123",
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
                    "tool_names": ["test_tool"],
                }
            )

            assert result == api_response
            mock_request.assert_called_once()


@pytest.mark.integration
def test_live_feedback_submission() -> None:
    """Submit feedback to the live API and assert a successful response."""
    import uuid

    api_key = os.getenv("STACKONE_API_KEY")
    if not api_key:
        pytest.skip("STACKONE_API_KEY env var required for live feedback test")

    base_url = os.getenv("STACKONE_BASE_URL", "https://api.stackone.com")
    from stackone_ai import StackOneToolSet

    toolset = StackOneToolSet(api_key=api_key, base_url=base_url)

    tools = toolset.get_tools("meta_collect_tool_feedback")
    feedback_tool = tools.get_tool("meta_collect_tool_feedback")
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


def test_implicit_feedback_integration() -> None:
    """Test implicit feedback system integration."""
    from stackone_ai.implicit_feedback import (
        BehaviorAnalyzer,
        BehaviorAnalyzerConfig,
        ImplicitFeedbackManager,
        SessionTracker,
    )
    from stackone_ai.implicit_feedback.data import ToolCallRecord
    from datetime import datetime, timedelta, timezone

    class StubLangsmithClient:
        def __init__(self) -> None:
            self.is_ready = True
            self.runs: list[dict[str, object]] = []
            self.feedback: list[dict[str, object]] = []

        def create_run(self, **kwargs: object) -> dict[str, object]:
            self.runs.append(kwargs)
            return {"id": f"run-{len(self.runs)}"}

        def create_feedback(
            self,
            *,
            run_id: str,
            key: str,
            score: float | None = None,
            comment: str | None = None,
            metadata: dict[str, object] | None = None,
        ) -> None:
            self.feedback.append(
                {
                    "run_id": run_id,
                    "key": key,
                    "score": score,
                    "comment": comment,
                    "metadata": metadata,
                }
            )

    analyzer = BehaviorAnalyzer()
    tracker = SessionTracker(analyzer)
    client = StubLangsmithClient()

    manager = ImplicitFeedbackManager(
        enabled=True,
        session_tracker=tracker,
        langsmith_client=client,  # type: ignore[arg-type]
    )

    start = datetime.now(timezone.utc)
    first_end = start + timedelta(seconds=2)
    manager.record_tool_call(
        tool_name="crm.search",
        start_time=start,
        end_time=first_end,
        status="success",
        params={"query": "alpha"},
        result={"count": 1},
        error=None,
        session_id="session-1",
        user_id="user-1",
        metadata={"source": "test"},
        fire_and_forget=False,
    )

    second_start = first_end + timedelta(seconds=3)
    manager.record_tool_call(
        tool_name="crm.search",
        start_time=second_start,
        end_time=second_start + timedelta(seconds=1),
        status="success",
        params={"query": "alpha"},
        result={"count": 0},
        error=None,
        session_id="session-1",
        user_id="user-1",
        metadata={"source": "test"},
        fire_and_forget=False,
    )

    assert len(client.runs) == 2
    assert client.feedback, "Expected implicit feedback events"
    feedback_entry = client.feedback[0]
    assert feedback_entry["key"] == "refinement_needed"
    assert feedback_entry["run_id"] == "run-2"
    assert isinstance(feedback_entry["metadata"], dict)
    assert feedback_entry["metadata"].get("tool_name") == "crm.search"
