"""
This example demonstrates how to use StackOne tools with LangChain.

```bash
uv run examples/langchain_integration.py
```
"""

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from stackone_ai import StackOneToolSet

load_dotenv()

account_id = "45072196112816593343"
employee_id = "c28xIQaWQ6MzM5MzczMDA2NzMzMzkwNzIwNA"


def langchain_integration() -> None:
    toolset = StackOneToolSet()
    tools = toolset.fetch_tools(actions=["bamboohr_*"], account_ids=[account_id])

    # Convert to LangChain format and verify
    langchain_tools = tools.to_langchain()
    assert len(langchain_tools) > 0, "Expected at least one LangChain tool"

    # Verify tool structure
    for tool in langchain_tools:
        assert hasattr(tool, "name"), "Expected tool to have name"
        assert hasattr(tool, "description"), "Expected tool to have description"
        assert hasattr(tool, "_run"), "Expected tool to have _run method"
        assert hasattr(tool, "args_schema"), "Expected tool to have args_schema"

    # Create model with tools
    model = ChatOpenAI(model="gpt-4o-mini")
    model_with_tools = model.bind_tools(langchain_tools)

    result = model_with_tools.invoke(f"Can you get me information about employee with ID: {employee_id}?")

    assert result.tool_calls is not None
    for tool_call in result.tool_calls:
        tool = tools.get_tool(tool_call["name"])
        if tool:
            result = tool.execute(tool_call["args"])
            assert result is not None
            assert result.get("data") is not None


if __name__ == "__main__":
    langchain_integration()
