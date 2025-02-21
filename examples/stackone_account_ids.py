"""
Handling StackOne account IDs with the StackOne Tools.

```bash
uv run examples/stackone_account_ids.py
```
"""

from dotenv import load_dotenv
from stackone_ai import StackOneToolSet

load_dotenv()


def stackone_account_ids():
    toolset = StackOneToolSet()

    # Filter by vertical and set the account ID
    tools = toolset.get_tools(vertical="hris", account_id="test_id")

    # You can over write the account ID here..
    tools.set_account_id("a_different_id")

    employee_tool = tools.get_tool("get_employee")
    if employee_tool:
        # You can even set the account ID on a per-tool basis
        employee_tool.set_account_id("again_another_id")

        print(employee_tool.get_account_id())


if __name__ == "__main__":
    stackone_account_ids()
