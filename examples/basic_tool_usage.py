"""
This example shows how to use StackOne tools.

## Requirements
- STACKONE_API_KEY
- STACKONE_ACCOUNT_ID
"""

import os

from dotenv import load_dotenv
from stackone_ai import StackOneToolSet


def main():
    # Load environment variables from .env
    load_dotenv()

    api_key = os.getenv("STACKONE_API_KEY")
    account_id = os.getenv("STACKONE_ACCOUNT_ID")

    if not api_key or not account_id:
        raise ValueError("STACKONE_API_KEY or STACKONE_ACCOUNT_ID not found in .env file")

    # Initialize the toolset with your API key
    toolset = StackOneToolSet(api_key=api_key)

    """
    ## Using Tools

    Once initialized, you can get tools for specific verticals:
    """

    # Get HRIS tools
    tools = toolset.get_tools(vertical="hris", account_id=account_id)

    """
    Then use specific tools by name:
    """

    # Example: Get an employee by ID
    try:
        # The tool name comes from the x-speakeasy-name-override in the OpenAPI spec
        employee_tool = tools.get_tool("get_employee")
        if employee_tool:
            employees = employee_tool.execute({"id": "c28xIQaWQ6MzM5MzczMDA2NzMzMzkwNzIwNA"})
            print("Employees retrieved:", employees)
    except Exception as e:
        print(f"Error getting employee: {e}")


if __name__ == "__main__":
    main()
