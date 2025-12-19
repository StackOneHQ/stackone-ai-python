"""
StackOne AI SDK provides an AI-friendly interface for accessing various SaaS tools through the StackOne API.

This SDK is available on [PyPI](https://pypi.org/project/stackone-ai/) for python projects. There is a node version in the works.

# Installation

```bash
# Using uv
uv add stackone-ai

# Using pip
pip install stackone-ai
```

# How to use these docs

All examples are complete and runnable.
We use [uv](https://docs.astral.sh/uv/getting-started/installation/) for easy python dependency management.

Install uv:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```
To run this example, clone the repo, install the dependencies (one-time setup) and run the script:

```bash
git clone https://github.com/stackoneHQ/stackone-ai-python.git
cd stackone-ai-python

# Install dependencies
uv sync --all-extras

# Run the example
uv run examples/index.py
```

# Authentication

Set the `STACKONE_API_KEY` environment variable:

```bash
export STACKONE_API_KEY=<your-api-key>
```

or load from a .env file:
"""

from dotenv import load_dotenv

load_dotenv()

"""
# Account IDs

StackOne uses account IDs to identify different integrations.
See the example [stackone-account-ids.md](stackone-account-ids.md) for more details.

This example will hardcode the account ID:
"""

account_id = "45072196112816593343"

"""
# Quickstart
"""

from stackone_ai import StackOneToolSet


def quickstart():
    toolset = StackOneToolSet()

    # Get all BambooHR-related tools using MCP-backed fetch_tools()
    tools = toolset.fetch_tools(actions=["bamboohr_*"], account_ids=[account_id])

    # Use a specific tool
    employee_tool = tools.get_tool("bamboohr_list_employees")
    assert employee_tool is not None

    employees = employee_tool.execute()
    assert employees is not None


if __name__ == "__main__":
    quickstart()

"""
# Next Steps

Check out some more documentation:

- [StackOne Account IDs](stackone-account-ids.md)
- [Error Handling](error-handling.md)
- [Available Tools](available-tools.md)
- [File Uploads](file-uploads.md)

Or get started with an integration:

- [OpenAI](openai-integration.md)
- [LangChain](langchain-integration.md)
- [CrewAI](crewai-integration.md)
- [LangGraph](langgraph-tool-node.md)
"""
