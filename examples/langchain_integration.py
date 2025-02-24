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
    tools = toolset.get_tools(vertical="hris", account_id=account_id)

    langchain_tools = tools.to_langchain()

    model = ChatOpenAI(model="gpt-4o-mini")
    model_with_tools = model.bind_tools(langchain_tools)

    result = model_with_tools.invoke(f"Can you get me information about employee with ID: {employee_id}?")

    if result.tool_calls:
        for tool_call in result.tool_calls:
            tool = tools.get_tool(tool_call["name"])
            if tool:
                result = tool.execute(tool_call["args"])
                assert result is not None
                assert result.get("data") is not None


if __name__ == "__main__":
    langchain_integration()
