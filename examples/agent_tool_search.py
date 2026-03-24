"""Search and execute example: LLM-driven tool discovery and execution.

There are two ways to give tools to an LLM:

1. ``toolset.openai()`` — fetches ALL tools and converts them to OpenAI format.
   Token cost scales with the number of tools in your catalog.

2. ``toolset.openai(mode="search_and_execute")`` — returns just 2 tools
   (tool_search + tool_execute). The LLM discovers and runs tools on-demand,
   keeping token usage constant regardless of catalog size.

This example demonstrates approach 2 with two patterns:
- Raw client (OpenAI): manual agent loop with ``toolset.execute()``
- LangChain: framework handles tool execution automatically

Prerequisites:
    - STACKONE_API_KEY environment variable
    - STACKONE_ACCOUNT_ID environment variable
    - OPENAI_API_KEY environment variable

Run with:
    uv run python examples/agent_tool_search.py
"""

from __future__ import annotations

import json
import os

try:
    from dotenv import load_dotenv

    load_dotenv()
except ModuleNotFoundError:
    pass

from stackone_ai import StackOneToolSet


def example_openai() -> None:
    """Raw client: OpenAI.

    Shows: init toolset -> get OpenAI tools -> manual agent loop with toolset.execute().
    """
    print("=" * 60)
    print("Example 1: Raw client (OpenAI) — manual execution")
    print("=" * 60)
    print()

    try:
        from openai import OpenAI
    except ImportError:
        print("Skipped: pip install openai")
        print()
        return

    if not os.getenv("OPENAI_API_KEY"):
        print("Skipped: Set OPENAI_API_KEY to run this example.")
        print()
        return

    # 1. Init toolset
    account_id = os.getenv("STACKONE_ACCOUNT_ID")
    toolset = StackOneToolSet(
        account_id=account_id,
        search={"method": "semantic", "top_k": 3},
        execute={"account_ids": [account_id]} if account_id else None,
    )

    # 2. Get tools in OpenAI format
    openai_tools = toolset.openai(mode="search_and_execute")

    # 3. Create OpenAI client and run agent loop
    client = OpenAI()
    messages: list[dict] = [
        {
            "role": "system",
            "content": (
                "You are a helpful scheduling assistant. Use tool_search to find relevant tools, "
                "then tool_execute to run them. Always read the parameter schemas from tool_search "
                "results carefully. If a tool needs a user URI, first search for and call a "
                '"get current user" tool to obtain it. If a tool execution fails, try different '
                "parameters or a different tool."
            ),
        },
        {"role": "user", "content": "List my upcoming Calendly events for the next week."},
    ]

    for _step in range(10):
        response = client.chat.completions.create(
            model="gpt-5.4",
            messages=messages,
            tools=openai_tools,
            tool_choice="auto",
        )

        choice = response.choices[0]

        # 4. If no tool calls, print final answer and stop
        if not choice.message.tool_calls:
            print(f"Answer: {choice.message.content}")
            break

        # 5. Execute tool calls manually and feed results back
        messages.append(choice.message.model_dump(exclude_none=True))
        for tool_call in choice.message.tool_calls:
            print(f"  -> {tool_call.function.name}({tool_call.function.arguments})")
            result = toolset.execute(tool_call.function.name, tool_call.function.arguments)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(result),
                }
            )

    print()


def example_langchain() -> None:
    """Framework: LangChain with auto-execution.

    Shows: init toolset -> get LangChain tools -> bind to model -> framework executes tools.
    No toolset.execute() needed — the framework calls _run() on tools automatically.
    """
    print("=" * 60)
    print("Example 2: LangChain — framework handles execution")
    print("=" * 60)
    print()

    try:
        from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
        from langchain_openai import ChatOpenAI
    except ImportError:
        print("Skipped: pip install langchain-openai")
        print()
        return

    if not os.getenv("OPENAI_API_KEY"):
        print("Skipped: Set OPENAI_API_KEY to run this example.")
        print()
        return

    # 1. Init toolset
    account_id = os.getenv("STACKONE_ACCOUNT_ID")
    toolset = StackOneToolSet(
        account_id=account_id,
        search={"method": "semantic", "top_k": 3},
        execute={"account_ids": [account_id]} if account_id else None,
    )

    # 2. Get tools in LangChain format and bind to model
    langchain_tools = toolset.langchain(mode="search_and_execute")
    tools_by_name = {tool.name: tool for tool in langchain_tools}
    model = ChatOpenAI(model="gpt-5.4").bind_tools(langchain_tools)

    # 3. Run agent loop
    messages = [
        SystemMessage(
            content=(
                "You are a helpful scheduling assistant. Use tool_search to find relevant tools, "
                "then tool_execute to run them. Always read the parameter schemas from tool_search "
                "results carefully. If a tool needs a user URI, first search for and call a "
                '"get current user" tool to obtain it. If a tool execution fails, try different '
                "parameters or a different tool."
            ),
        ),
        HumanMessage(content="List my upcoming Calendly events for the next week."),
    ]

    for _step in range(10):
        response: AIMessage = model.invoke(messages)

        # 4. If no tool calls, print final answer and stop
        if not response.tool_calls:
            print(f"Answer: {response.content}")
            break

        # 5. Framework-compatible execution — invoke LangChain tools directly
        messages.append(response)
        for tool_call in response.tool_calls:
            print(f"  -> {tool_call['name']}({json.dumps(tool_call['args'])})")
            tool = tools_by_name[tool_call["name"]]
            result = tool.invoke(tool_call["args"])
            messages.append(ToolMessage(content=json.dumps(result), tool_call_id=tool_call["id"]))

    print()


def main() -> None:
    """Run all examples."""
    api_key = os.getenv("STACKONE_API_KEY")
    if not api_key:
        print("Set STACKONE_API_KEY to run these examples.")
        return

    example_openai()
    example_langchain()


if __name__ == "__main__":
    main()
