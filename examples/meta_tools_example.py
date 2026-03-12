"""Meta tools example: LLM-driven tool discovery and execution.

There are two ways to give tools to an LLM:

1. ``toolset.openai()`` — fetches ALL tools and converts them to OpenAI format.
   Token cost scales with the number of tools in your catalog.

2. ``toolset.openai(mode="search_and_execute")`` — returns just 2 meta tools
   (tool_search + tool_execute). The LLM discovers and runs tools on-demand,
   keeping token usage constant regardless of catalog size.

This example demonstrates approach 2 with OpenAI and LangChain clients.

Prerequisites:
    - STACKONE_API_KEY environment variable
    - STACKONE_ACCOUNT_ID environment variable
    - GOOGLE_API_KEY environment variable (for Gemini)
    - OPENAI_API_KEY environment variable (optional, for LangChain example)

Run with:
    uv run python examples/meta_tools_example.py
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


def example_gemini() -> None:
    """Complete Gemini integration with meta tools via OpenAI-compatible API.

    Shows: init toolset -> get OpenAI tools -> agent loop -> final answer.
    Uses gemini-3-pro-preview which handles tool schemas and dates well.
    """
    print("=" * 60)
    print("Example 1: Gemini client with meta tools")
    print("=" * 60)
    print()

    try:
        from openai import OpenAI
    except ImportError:
        print("Skipped: pip install openai")
        print()
        return

    google_key = os.getenv("GOOGLE_API_KEY")
    if not google_key:
        print("Skipped: Set GOOGLE_API_KEY to run this example.")
        print()
        return

    # 1. Init toolset
    account_id = os.getenv("STACKONE_ACCOUNT_ID")
    toolset = StackOneToolSet(
        account_id=account_id,
        search={"method": "semantic", "top_k": 3},
        execute={"account_ids": [account_id]} if account_id else None,
    )

    # 2. Get meta tools in OpenAI format
    meta_tools = toolset.get_meta_tools()
    openai_tools = toolset.openai(mode="search_and_execute")

    # 3. Create Gemini client (OpenAI-compatible) and run agent loop
    client = OpenAI(
        api_key=google_key,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    )
    messages: list[dict] = [
        {"role": "user", "content": "List my upcoming Calendly events for the next week."},
    ]

    for _step in range(10):
        response = client.chat.completions.create(
            model="gemini-3-pro-preview",
            messages=messages,
            tools=openai_tools,
            tool_choice="auto",
        )

        choice = response.choices[0]

        # 4. If no tool calls, print final answer and stop
        if not choice.message.tool_calls:
            print(f"Answer: {choice.message.content}")
            break

        # 5. Execute tool calls and feed results back
        messages.append(choice.message.model_dump(exclude_none=True))
        for tool_call in choice.message.tool_calls:
            print(f"  -> {tool_call.function.name}({tool_call.function.arguments})")
            tool = meta_tools.get_tool(tool_call.function.name)
            result = tool.execute(tool_call.function.arguments) if tool else {"error": "Unknown tool"}
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(result),
                }
            )

    print()


def example_langchain() -> None:
    """Complete LangChain integration with meta tools.

    Shows: init toolset -> bind tools to ChatOpenAI -> agent loop -> final answer.
    """
    print("=" * 60)
    print("Example 2: LangChain client with meta tools")
    print("=" * 60)
    print()

    try:
        from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
        from langchain_google_genai import ChatGoogleGenerativeAI
    except ImportError:
        print("Skipped: pip install langchain-google-genai")
        print()
        return

    if not os.getenv("GOOGLE_API_KEY"):
        print("Skipped: Set GOOGLE_API_KEY to run this example.")
        print()
        return

    # 1. Init toolset
    account_id = os.getenv("STACKONE_ACCOUNT_ID")
    toolset = StackOneToolSet(
        account_id=account_id,
        search={"method": "semantic", "top_k": 3},
        execute={"account_ids": [account_id]} if account_id else None,
    )

    # 2. Get meta tools in LangChain format and bind to model
    meta_tools = toolset.get_meta_tools()
    langchain_tools = meta_tools.to_langchain()
    model = ChatGoogleGenerativeAI(model="gemini-3-pro-preview").bind_tools(langchain_tools)

    # 3. Run agent loop
    messages = [HumanMessage(content="List my upcoming Calendly events for the next week.")]

    for _step in range(10):
        response: AIMessage = model.invoke(messages)

        # 4. If no tool calls, print final answer and stop
        if not response.tool_calls:
            print(f"Answer: {response.content}")
            break

        # 5. Execute tool calls and feed results back
        messages.append(response)
        for tool_call in response.tool_calls:
            print(f"  -> {tool_call['name']}({json.dumps(tool_call['args'])})")
            tool = meta_tools.get_tool(tool_call["name"])
            result = tool.execute(tool_call["args"]) if tool else {"error": "Unknown tool"}
            messages.append(ToolMessage(content=json.dumps(result), tool_call_id=tool_call["id"]))

    print()


def main() -> None:
    """Run all meta tools examples."""
    api_key = os.getenv("STACKONE_API_KEY")
    if not api_key:
        print("Set STACKONE_API_KEY to run these examples.")
        return

    example_gemini()
    example_langchain()


if __name__ == "__main__":
    main()
