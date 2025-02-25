"""
This example demonstrates error handling when using the StackOne SDK.

Run the following command to see the output:

```bash
uv run examples/error_handling.py
```
"""

import os

from dotenv import load_dotenv

from stackone_ai import StackOneToolSet
from stackone_ai.models import StackOneAPIError
from stackone_ai.toolset import ToolsetConfigError, ToolsetLoadError

load_dotenv()


def error_handling() -> None:
    """
    Example 1: Configuration error - missing API key
    """
    original_api_key = os.environ.pop("STACKONE_API_KEY", None)
    try:
        try:
            StackOneToolSet(api_key=None)
            raise AssertionError("Expected ToolsetConfigError")
        except ToolsetConfigError as e:
            assert (
                str(e)
                == "API key must be provided either through api_key parameter or STACKONE_API_KEY environment variable"
            )
    finally:
        if original_api_key:
            os.environ["STACKONE_API_KEY"] = original_api_key

    """
    Example 2: Invalid vertical error
    """
    toolset = StackOneToolSet()
    try:
        # Use a non-existent vertical to trigger error
        tools = toolset.get_tools("nonexistent_vertical_*")
        # If we get here, no tools were found but no error was raised
        assert len(tools) == 0, "Expected no tools for nonexistent vertical"
    except ToolsetLoadError as e:
        assert "Error loading tools" in str(e)

    """
    Example 3: API error - invalid request
    """
    toolset = StackOneToolSet()
    tools = toolset.get_tools("crm_*")

    # Try to make an API call without required parameters
    list_contacts = tools.get_tool("crm_list_contacts")
    assert list_contacts is not None, "Expected crm_list_contacts tool to exist"

    try:
        # Execute without required parameters should raise error
        list_contacts.execute({})
        raise AssertionError("Expected StackOneAPIError")
    except StackOneAPIError as e:
        assert e.status_code >= 400, "Expected error status code"
        assert e.response_body is not None, "Expected error response body"


if __name__ == "__main__":
    error_handling()
