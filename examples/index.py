"""
# Getting Started

StackOne AI provides a unified interface for accessing various SaaS tools through AI-friendly APIs.

## Installation

Install StackOne AI using pip:

```bash
pip install stackone-ai
```


## Quick Start

Here's a simple example to get you started:
"""

import os

from dotenv import load_dotenv
from stackone_ai import StackOneToolSet


def quickstart():
    # Load environment variables
    load_dotenv()

    api_key = os.getenv("STACKONE_API_KEY")
    account_id = os.getenv("STACKONE_ACCOUNT_ID")

    # Initialize the toolset
    toolset = StackOneToolSet(api_key=api_key)

    # Get HRIS tools
    tools = toolset.get_tools(vertical="hris", account_id=account_id)

    # Use a specific tool
    employee_tool = tools.get_tool("get_employee")
    if employee_tool:
        employee = employee_tool.execute({"id": "employee_id"})
        print(employee)


"""
## Authentication

StackOne AI requires two key pieces of information:
- `STACKONE_API_KEY`: Your API key from StackOne
- `STACKONE_ACCOUNT_ID`: Your account ID

You can set these as environment variables or pass them directly to the StackOneToolSet constructor.

## Next Steps

Check out some examples:
- [Basic Tool Usage](basic-tool-usage.md)
- [Error Handling](error-handling.md)
- [OpenAI Integration](openai-integration.md)
"""
