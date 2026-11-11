"""
This example demonstrates how to use StackOne tools with LangChain.

```bash
uv run examples/langchain_integration.py
```
"""

from __future__ import annotations

import os
import sys

try:
    from dotenv import load_dotenv

    load_dotenv()
except ModuleNotFoundError:
    pass

from langchain_openai import ChatOpenAI

from stackone_ai import StackOneToolSet
from stackone_ai.types import StackOneError, ToolsetError


def langchain_integration() -> None:
    for var in ["STACKONE_API_KEY", "STACKONE_ACCOUNT_ID", "OPENAI_API_KEY"]:
        if not os.getenv(var):
            raise SystemExit(f"Set {var} to run this example.")

    account_id = os.environ["STACKONE_ACCOUNT_ID"]

    toolset = StackOneToolSet()
    tools = toolset.fetch_tools(
        actions=["*_list_*"],
        account_ids=[account_id],
    )
    if not tools:
        raise SystemExit(f"Account {account_id} served no list tools, so the example would prove nothing.")

    # Convert to LangChain format
    langchain_tools = tools.to_langchain()
    print(f"Loaded {len(langchain_tools)} LangChain tools.")
    for tool in langchain_tools:
        print(f"  - {tool.name}")

    # Create model with tools
    model = ChatOpenAI(model="gpt-5.4")  # ty: ignore[unknown-argument]
    model_with_tools = model.bind_tools(langchain_tools)

    result = model_with_tools.invoke("Use one of your tools to list a few records, then summarise them.")
    print(f"LLM response: {result.content}")

    if result.tool_calls:
        print(f"LLM made {len(result.tool_calls)} tool call(s).")
        for tool_call in result.tool_calls:
            print(f"  - {tool_call['name']}({tool_call['args']})")
            tool = tools.get_tool(tool_call["name"])
            if tool is None:
                raise SystemExit(f"The model called {tool_call['name']}, which is not a fetched tool.")
            call_result = tool.execute(tool_call["args"])
            print(f"    Result: {str(call_result)[:200]}...")
    else:
        print("No tool calls were made by the model.")


if __name__ == "__main__":
    try:
        langchain_integration()
    except ToolsetError as exc:
        sys.exit(f"Could not load tools: {exc}")
    except StackOneError as exc:
        sys.exit(f"StackOne API error: {exc}")
