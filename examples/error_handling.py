from dotenv import load_dotenv
from stackone_ai import StackOneToolSet
from stackone_ai.toolset import ToolsetConfigError, ToolsetLoadError

load_dotenv()


def error_handling() -> None:
    try:
        # Example 1: Handle missing API key
        invalid_toolset = StackOneToolSet(api_key=None)
    except ToolsetConfigError as e:
        print("Config Error:", e)
        # Config Error: API key must be provided either through api_key parameter or STACKONE_API_KEY environment variable

    toolset = StackOneToolSet()

    # Example 2: Handle unknown vertical
    try:
        tools = toolset.get_tools(vertical="unknown_vertical")
    except ToolsetLoadError as e:
        print("Vertical Load Error:", e)
        # Vertical Load Error: No spec file found for vertical: unknown_vertical

    # Example 3: Handle API errors with account_id
    tools = toolset.get_tools(vertical="crm", account_id="test_id")
    try:
        # Try with invalid ID
        contacts_tool = tools.get_tool("get_contact")
        if contacts_tool:
            result = contacts_tool.execute({"id": "invalid_id"})
    except Exception as e:
        print(f"API Error: {e}")
        # API Error: 400 Client Error: Bad Request for url: https://api.stackone.com/unified/crm/contacts/invalid_id

    # Example 4: Handle missing account ID
    tools_no_account = toolset.get_tools(vertical="crm", account_id=None)
    try:
        list_contacts_tool = tools_no_account.get_tool("list_contacts")
        if list_contacts_tool:
            result = list_contacts_tool.execute()
            print("Result without account ID:", result)
    except Exception as e:
        print(f"Error when account ID is missing: {e}")
        # Error when account ID is missing: 501 Server Error: Not Implemented for url: https://api.stackone.com/unified/crm/contacts


if __name__ == "__main__":
    error_handling()
