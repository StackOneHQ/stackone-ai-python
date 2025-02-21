"""
# Getting Started

StackOne AI provides a unified interface for accessing various SaaS tools through AI-friendly APIs.

## Installation

```bash
pip install stackone-ai
```


## Quick Start

Here's a simple example. All examples are complete and runnable.
"""

from dotenv import load_dotenv
from stackone_ai import StackOneToolSet

"""
## Authenticate with StackOne

```bash
export STACKONE_API_KEY=<your-api-key>
```

or load from a .env file:
"""

load_dotenv()

"""
## Account IDs

StackOne uses account IDs to identify different integrations.
See the example [stackone_account_ids.py](stackone_account_ids.py) for more details.
This example will hardcode the account ID.
"""

account_id = "45072196112816593343"
employee_id = "c28xIQaWQ6MzM5MzczMDA2NzMzMzkwNzIwNA"


def quickstart():
    toolset = StackOneToolSet()

    # Filter by vertical and add the account ID
    tools = toolset.get_tools(vertical="hris", account_id=account_id)

    # Use a specific tool
    employee_tool = tools.get_tool("get_employee")
    if employee_tool:
        employee = employee_tool.execute({"id": employee_id})
        print(employee)


if __name__ == "__main__":
    quickstart()

"""
## Next Steps

Check out some examples:
- [Error Handling](error-handling.md)
- [StackOne Account IDs](stackone_account_ids.md)
- [Available Tools](available_tools.md)
- [OpenAI Integration](openai-integration.md)
- [LangChain Integration](langchain-integration.md)
- [CrewAI Integration](crewai-integration.md)
- [LangGraph Tool Node](langgraph-tool-node.md)
"""
