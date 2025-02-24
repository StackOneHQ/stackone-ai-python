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
    # Example 1: Configuration error - missing API key
    try:
        print("\n1. Testing configuration error (missing API key)...")
        original_api_key = os.environ.pop("STACKONE_API_KEY", None)
        try:
            StackOneToolSet(api_key=None)
            raise AssertionError("Expected ToolsetConfigError")
        finally:
            if original_api_key:
                os.environ["STACKONE_API_KEY"] = original_api_key
    except ToolsetConfigError as e:
        print("✗ Config Error:", e)

    # Example 2: Invalid vertical error
    try:
        print("\n2. Testing invalid vertical...")
        toolset = StackOneToolSet()
        toolset.get_tools(vertical="invalid_vertical")
        raise AssertionError("Expected ToolsetLoadError")
    except ToolsetLoadError as e:
        print("✗ Load Error:", e)

    # Example 3: API error - invalid request
    try:
        print("\n3. Testing API error...")
        toolset = StackOneToolSet()
        tools = toolset.get_tools(vertical="crm")

        # Try to make an API call without required parameters
        list_contacts = tools.get_tool("crm_list_contacts")
        assert list_contacts is not None, "Expected crm_list_contacts tool to exist"

        list_contacts.execute()
        raise AssertionError("Expected StackOneAPIError")
    except StackOneAPIError as e:
        print("✗ API Error:", e)
        print("  Status:", e.status_code)
        print("  Response:", e.response_body)


if __name__ == "__main__":
    error_handling()
