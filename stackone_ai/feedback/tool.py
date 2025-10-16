"""Feedback collection tool for StackOne."""

# TODO: Remove when Python 3.9 support is dropped
from __future__ import annotations

import json

from pydantic import BaseModel, Field, field_validator

from ..models import (
    ExecuteConfig,
    JsonDict,
    ParameterLocation,
    StackOneError,
    StackOneTool,
    ToolParameters,
)


class FeedbackInput(BaseModel):
    """Input schema for feedback tool."""

    feedback: str = Field(..., min_length=1, description="User feedback text")
    account_id: str = Field(..., min_length=1, description="Account identifier")
    tool_names: list[str] = Field(..., min_length=1, description="List of tool names")

    @field_validator("feedback", "account_id")
    @classmethod
    def validate_non_empty_trimmed(cls, v: str) -> str:
        """Validate that string is non-empty after trimming."""
        trimmed = v.strip()
        if not trimmed:
            raise ValueError("Field must be a non-empty string")
        return trimmed

    @field_validator("tool_names")
    @classmethod
    def validate_tool_names(cls, v: list[str]) -> list[str]:
        """Validate and clean tool names."""
        cleaned = [name.strip() for name in v if name.strip()]
        if not cleaned:
            raise ValueError("At least one tool name is required")
        return cleaned


class FeedbackTool(StackOneTool):
    """Extended tool for collecting feedback with enhanced validation."""

    def execute(self, arguments: str | JsonDict | None = None) -> JsonDict:
        """
        Execute the feedback tool with enhanced validation.

        Args:
            arguments: Tool arguments as string or dict

        Returns:
            Response from the API

        Raises:
            StackOneError: If validation or API call fails
        """
        try:
            # Parse input
            if isinstance(arguments, str):
                raw_params = json.loads(arguments)
            else:
                raw_params = arguments or {}

            # Validate with Pydantic
            parsed_params = FeedbackInput(**raw_params)

            # Build validated request body
            validated_arguments = {
                "feedback": parsed_params.feedback,
                "account_id": parsed_params.account_id,
                "tool_names": parsed_params.tool_names,
            }

            # Use the parent execute method with validated arguments
            return super().execute(validated_arguments)

        except json.JSONDecodeError as exc:
            raise StackOneError(f"Invalid JSON in arguments: {exc}") from exc
        except ValueError as exc:
            raise StackOneError(f"Validation error: {exc}") from exc
        except Exception as error:
            if isinstance(error, StackOneError):
                raise
            raise StackOneError(f"Error executing feedback tool: {error}") from error


def create_feedback_tool(
    api_key: str,
    account_id: str | None = None,
    base_url: str = "https://api.stackone.com",
) -> FeedbackTool:
    """
    Create a feedback collection tool.

    Args:
        api_key: API key for authentication
        account_id: Optional account ID
        base_url: Base URL for the API

    Returns:
        FeedbackTool configured for feedback collection
    """
    name = "meta_collect_tool_feedback"
    description = (
        "Collects user feedback on StackOne tool performance. "
        "First ask the user, \"Are you ok with sending feedback to StackOne?\" "
        "and mention that the LLM will take care of sending it. "
        "Call this tool only when the user explicitly answers yes."
    )

    parameters = ToolParameters(
        type="object",
        properties={
            "account_id": {
                "type": "string",
                "description": 'Account identifier (e.g., "acc_123456")',
            },
            "feedback": {
                "type": "string",
                "description": "Verbatim feedback from the user about their experience with StackOne tools.",
            },
            "tool_names": {
                "type": "array",
                "items": {
                    "type": "string",
                },
                "description": "Array of tool names being reviewed",
            },
        },
    )

    execute_config = ExecuteConfig(
        name=name,
        method="POST",
        url=f"{base_url}/ai/tool-feedback",
        body_type="json",
        parameter_locations={
            "feedback": ParameterLocation.BODY,
            "account_id": ParameterLocation.BODY,
            "tool_names": ParameterLocation.BODY,
        },
    )

    # Create instance by calling parent class __init__ directly since FeedbackTool is a subclass
    tool = FeedbackTool.__new__(FeedbackTool)
    StackOneTool.__init__(
        tool,
        description=description,
        parameters=parameters,
        _execute_config=execute_config,
        _api_key=api_key,
        _account_id=account_id,
    )

    return tool
