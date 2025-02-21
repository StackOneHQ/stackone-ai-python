from dotenv import load_dotenv
from stackone_ai import StackOneToolSet

load_dotenv()


def error_handling() -> None:
    toolset = StackOneToolSet()

    # Example 1: Handle unknown vertical
    tools = toolset.get_tools(vertical="unknown_vertical")
    print("Tools for unknown vertical:", tools._tool_map)
    # {}

    # Example 2: Handle API errors with account_id
    tools = toolset.get_tools(vertical="crm", account_id="test_id")
    try:
        # Try with invalid ID
        contacts_tool = tools.get_tool("get_contact")
        if contacts_tool:
            result = contacts_tool.execute({"id": "invalid_id"})
    except Exception as e:
        print(f"API Error: {e}")
        # 400 Client Error: Bad Request for url: https://api.stackone.com/unified/crm/contacts/invalid_id

    # Example 3: Handle missing account ID
    tools_no_account = toolset.get_tools(vertical="crm", account_id=None)
    try:
        list_contacts_tool = tools_no_account.get_tool("list_contacts")
        if list_contacts_tool:
            result = list_contacts_tool.execute()
            print("Result without account ID:", result)
    except Exception as e:
        print(f"Error when account ID is missing: {e}")
        # 501 Server Error: Not Implemented for url: https://api.stackone.com/unified/crm/contacts


if __name__ == "__main__":
    error_handling()
