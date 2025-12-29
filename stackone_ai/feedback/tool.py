"""Feedback collection tool for StackOne."""

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
    account_id: str | list[str] = Field(..., description="Account identifier(s) - single ID or list of IDs")
    tool_names: list[str] = Field(..., min_length=1, description="List of tool names")

    @field_validator("feedback")
    @classmethod
    def validate_feedback(cls, v: str) -> str:
        """Validate that feedback is non-empty after trimming."""
        trimmed = v.strip()
        if not trimmed:
            raise ValueError("Feedback must be a non-empty string")
        return trimmed

    @field_validator("account_id")
    @classmethod
    def validate_account_id(cls, v: str | list[str]) -> list[str]:
        """Validate and normalize account ID(s) to a list."""
        if isinstance(v, str):
            trimmed = v.strip()
            if not trimmed:
                raise ValueError("Account ID must be a non-empty string")
            return [trimmed]

        if isinstance(v, list):
            if not v:
                raise ValueError("At least one account ID is required")
            cleaned = [str(item).strip() for item in v if str(item).strip()]
            if not cleaned:
                raise ValueError("At least one valid account ID is required")
            return cleaned

        raise ValueError("Account ID must be a string or list of strings")

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

    def execute(
        self, arguments: str | JsonDict | None = None, *, options: JsonDict | None = None
    ) -> JsonDict:
        """
        Execute the feedback tool with enhanced validation.

        If multiple account IDs are provided, sends the same feedback to each account individually.

        Args:
            arguments: Tool arguments as string or dict
            options: Execution options

        Returns:
            Combined response from all API calls

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

            # Get list of account IDs (already normalized by validator)
            account_ids = parsed_params.account_id
            feedback = parsed_params.feedback
            tool_names = parsed_params.tool_names

            # If only one account ID, use the parent execute method
            if len(account_ids) == 1:
                validated_arguments = {
                    "feedback": feedback,
                    "account_id": account_ids[0],
                    "tool_names": tool_names,
                }
                return super().execute(validated_arguments, options=options)

            # Multiple account IDs - send to each individually
            results = []
            errors = []

            for account_id in account_ids:
                try:
                    validated_arguments = {
                        "feedback": feedback,
                        "account_id": account_id,
                        "tool_names": tool_names,
                    }
                    result = super().execute(validated_arguments, options=options)
                    results.append({"account_id": account_id, "status": "success", "result": result})
                except Exception as exc:
                    error_msg = str(exc)
                    errors.append({"account_id": account_id, "status": "error", "error": error_msg})
                    results.append({"account_id": account_id, "status": "error", "error": error_msg})

            # Return combined results
            return {
                "message": f"Feedback sent to {len(account_ids)} account(s)",
                "total_accounts": len(account_ids),
                "successful": len([r for r in results if r["status"] == "success"]),
                "failed": len(errors),
                "results": results,
            }

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
        'First ask the user, "Are you ok with sending feedback to StackOne?" '
        "and mention that the LLM will take care of sending it. "
        "Call this tool only when the user explicitly answers yes."
    )

    parameters = ToolParameters(
        type="object",
        properties={
            "account_id": {
                "oneOf": [
                    {
                        "type": "string",
                        "description": 'Single account identifier (e.g., "acc_123456")',
                    },
                    {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of account identifiers for multiple accounts",
                    },
                ],
                "description": "Account identifier(s) - single ID or list of IDs",
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
