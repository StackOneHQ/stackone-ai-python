"""
This example demonstrates how to use StackOne tools with LangGraph.

LangGraph uses LangChain tools natively with its prebuilt ReAct agent.

```bash
uv run examples/langgraph_integration.py
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

from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

from stackone_ai import StackOneToolSet
from stackone_ai.types import StackOneAPIError, ToolsetError


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
        actions=["*_list_*"],
        account_ids=[account_id],
    )

    # LangGraph uses LangChain tools natively
    langchain_tools = tools.to_langchain()

    # Hand a rejected call back to the agent instead of killing the graph. Models
    # do guess arguments wrong, and StackOne's 400 names the offending field — so
    # the agent can read it and retry. Without this, one bad guess ends the run.
    for langchain_tool in langchain_tools:
        langchain_tool.handle_tool_error = True
    print(f"Loaded {len(langchain_tools)} LangGraph tools.")
    for tool in langchain_tools:
        print(f"  - {tool.name}")

    # Create a ReAct agent with LangGraph
    model = ChatOpenAI(model="gpt-5.4")  # ty: ignore[unknown-argument]
    agent = create_agent(model, langchain_tools)

    result = agent.invoke(
        {
            "messages": [
                (
                    "user",
                    "Use one of your tools to list a few records, then summarise them. Call it with no arguments unless the schema marks a field required. "
                    "Call it with no arguments unless the schema marks a field required.",
                )
            ]
        }
    )
    final_message = result["messages"][-1]
    print(f"Agent response:\n{final_message.content}")


if __name__ == "__main__":
    try:
        langgraph_integration()
    except ToolsetError as exc:
        sys.exit(f"Could not load tools: {exc}")
    except StackOneAPIError as exc:
        sys.exit(f"Tool call rejected with {exc.status_code}: {exc.response_body}")
