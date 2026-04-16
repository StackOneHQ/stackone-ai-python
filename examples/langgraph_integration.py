"""
This example demonstrates how to use StackOne tools with LangGraph.

LangGraph uses LangChain tools natively with its prebuilt ReAct agent.

```bash
uv run examples/langgraph_integration.py
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
from langgraph.prebuilt import create_react_agent

from stackone_ai import StackOneToolSet


def langgraph_integration() -> None:
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

    # LangGraph uses LangChain tools natively
    langchain_tools = tools.to_langchain()
    print(f"Loaded {len(langchain_tools)} LangGraph tools.")
    for tool in langchain_tools:
        print(f"  - {tool.name}")

    # Create a ReAct agent with LangGraph
    model = ChatOpenAI(model="gpt-5.4")
    agent = create_react_agent(model, langchain_tools)

    result = agent.invoke({"messages": [("user", "List the first 5 employees")]})
    final_message = result["messages"][-1]
    print(f"Agent response:\n{final_message.content}")


if __name__ == "__main__":
    langgraph_integration()
