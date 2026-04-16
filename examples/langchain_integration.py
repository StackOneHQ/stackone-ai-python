"""
This example demonstrates how to use StackOne tools with LangChain.

```bash
uv run examples/langchain_integration.py
```
"""

from __future__ import annotations

import os

try:
    from dotenv import load_dotenv

    load_dotenv()
except ModuleNotFoundError:
    pass

from langchain_openai import ChatOpenAI

from stackone_ai import StackOneToolSet


def langchain_integration() -> None:
    account_id = os.getenv("STACKONE_ACCOUNT_ID")
    if not os.getenv("STACKONE_API_KEY"):
        print("Set STACKONE_API_KEY to run this example.")
        return
    if not account_id:
        print("Set STACKONE_ACCOUNT_ID to run this example.")
        return
    if not os.getenv("OPENAI_API_KEY"):
        print("Set OPENAI_API_KEY to run this example.")
        return

    toolset = StackOneToolSet()
    tools = toolset.fetch_tools(
        actions=["workday_list_workers", "workday_get_worker", "workday_get_current_user"],
        account_ids=[account_id],
    )

    # Convert to LangChain format
    langchain_tools = tools.to_langchain()
    print(f"Loaded {len(langchain_tools)} LangChain tools.")
    for tool in langchain_tools:
        print(f"  - {tool.name}")

    # Create model with tools
    model = ChatOpenAI(model="gpt-5.4")
    model_with_tools = model.bind_tools(langchain_tools)

    result = model_with_tools.invoke("List the first 5 employees")
    print(f"LLM response: {result.content}")

    if result.tool_calls:
        print(f"LLM made {len(result.tool_calls)} tool call(s).")
        for tool_call in result.tool_calls:
            print(f"  - {tool_call['name']}({tool_call['args']})")
            tool = tools.get_tool(tool_call["name"])
            if tool:
                call_result = tool.execute(tool_call["args"])
                print(f"    Result: {str(call_result)[:200]}...")
    else:
        print("No tool calls were made by the model.")


if __name__ == "__main__":
    langchain_integration()
