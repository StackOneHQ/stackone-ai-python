import os

from dotenv import load_dotenv
from stackone_ai import StackOneToolSet


def demonstrate_error_handling():
    # Load environment variables from root .env
    load_dotenv()

    api_key = os.getenv("STACKONE_API_KEY")
    account_id = os.getenv("STACKONE_ACCOUNT_ID")

    if not api_key:
        raise ValueError("STACKONE_API_KEY not found in .env file")

    toolset = StackOneToolSet(api_key=api_key)

    # Example 1: Handle unknown vertical
    try:
        tools = toolset.get_tools(vertical="unknown_vertical")
        print("Tools for unknown vertical:", tools)  # Will print empty dict
    except Exception as e:
        print(f"Error getting tools: {e}")

    # Example 2: Handle API errors with account_id
    tools = toolset.get_tools(vertical="crm", account_id=account_id)

    # Example of handling various API errors
    try:
        # Try with invalid ID
        contacts_tool = tools.get_tool("get_contacts")
        if contacts_tool:
            result = contacts_tool.execute({"id": "invalid_id"})
    except Exception as e:
        print(f"API Error: {e}")

    # Example 3: Handle missing account ID
    tools_no_account = toolset.get_tools(vertical="crm")  # No account_id
    try:
        contacts_tool = tools_no_account.get_tool("get_contacts")
        if contacts_tool:
            result = contacts_tool.execute({"id": "123"})
            print("Result without account ID:", result)
    except Exception as e:
        print(f"Error when account ID is missing: {e}")


if __name__ == "__main__":
    demonstrate_error_handling()
