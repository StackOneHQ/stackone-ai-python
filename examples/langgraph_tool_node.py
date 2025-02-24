"""
TODO!!

This example demonstrates how to use StackOne tools with LangGraph.

```bash
uv run examples/langgraph_tool_node.py
```
"""

from dotenv import load_dotenv

from stackone_ai import StackOneToolSet

load_dotenv()

account_id = "45072196112816593343"
employee_id = "c28xIQaWQ6MzM5MzczMDA2NzMzMzkwNzIwNA"


def langgraph_tool_node() -> None:
    """Demonstrate basic LangGraph integration with StackOne tools."""
    toolset = StackOneToolSet()
    tools = toolset.get_tools("hris_*", account_id=account_id)

    # Verify we have the tools we need
    assert len(tools) > 0, "Expected at least one HRIS tool"
    employee_tool = tools.get_tool("hris_get_employee")
    assert employee_tool is not None, "Expected hris_get_employee tool"

    # TODO: Add LangGraph specific integration
    # For now, just verify the tools are properly configured
    langchain_tools = tools.to_langchain()
    assert len(langchain_tools) > 0, "Expected LangChain tools"
    assert all(hasattr(tool, "_run") for tool in langchain_tools), "Expected all tools to have _run method"


if __name__ == "__main__":
    langgraph_tool_node()
