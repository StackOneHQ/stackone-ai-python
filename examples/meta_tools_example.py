"""Meta tools example: LLM-driven tool discovery and execution.

Instead of loading all tools upfront, the LLM autonomously searches for
relevant tools and executes them — keeping token usage minimal.

Prerequisites:
    - STACKONE_API_KEY environment variable
    - STACKONE_ACCOUNT_ID environment variable (comma-separated for multiple)
    - OPENAI_API_KEY or GOOGLE_API_KEY environment variable

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

_account_ids = [aid.strip() for aid in os.getenv("STACKONE_ACCOUNT_ID", "").split(",") if aid.strip()]


def example_openai_meta_tools() -> None:
    """Meta tools with OpenAI Chat Completions.

    The LLM receives only tool_search and tool_execute — two small tool
    definitions regardless of how many tools exist. It searches for what
    it needs and executes.
    """
    print("=" * 60)
    print("Example 1: Meta tools with OpenAI")
    print("=" * 60)
    print()

    try:
        from openai import OpenAI
    except ImportError:
        print("Skipped: OpenAI library not installed. Install with: pip install openai")
        print()
        return

    openai_key = os.getenv("OPENAI_API_KEY")
    google_key = os.getenv("GOOGLE_API_KEY")

    if openai_key:
        client = OpenAI()
        model = "gpt-5.1"
        provider = "OpenAI"
    elif google_key:
        client = OpenAI(
            api_key=google_key,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        )
        model = "gemini-3-pro-preview"
        provider = "Gemini"
    else:
        print("Skipped: Set OPENAI_API_KEY or GOOGLE_API_KEY to run this example.")
        print()
        return

    print(f"Using {provider} ({model})")
    print()

    toolset = StackOneToolSet(search={"method": "semantic", "top_k": 3})

    # Get meta tools — returns a Tools collection with tool_search + tool_execute
    meta_tools = toolset.get_meta_tools(account_ids=_account_ids or None)
    openai_tools = meta_tools.to_openai()

    print(f"Meta tools: {[t.name for t in meta_tools]}")
    print()

    messages: list[dict] = [
        {
            "role": "system",
            "content": (
                "You are a helpful scheduling assistant. "
                "Use tool_search to find relevant tools, then tool_execute to run them. "
                "If a tool execution fails, try different parameters or a different tool. "
                "Do not repeat the same failed call."
            ),
        },
        {
            "role": "user",
            "content": "List my upcoming Calendly events for the next week.",
        },
    ]

    # Agent loop — let the LLM drive search and execution
    max_iterations = 10
    for iteration in range(max_iterations):
        print(f"--- Iteration {iteration + 1} ---")

        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=openai_tools,
            tool_choice="auto",
        )

        choice = response.choices[0]

        if not choice.message.tool_calls:
            print(f"\n{provider} final response: {choice.message.content}")
            break

        # Add assistant message with tool calls
        # Use model_dump with exclude_none to avoid null values that Gemini rejects
        messages.append(choice.message.model_dump(exclude_none=True))

        # Execute each tool call
        for tool_call in choice.message.tool_calls:
            print(f"LLM called: {tool_call.function.name}({tool_call.function.arguments})")

            tool = meta_tools.get_tool(tool_call.function.name)
            if tool is None:
                result = {"error": f"Unknown tool: {tool_call.function.name}"}
            else:
                result = tool.execute(tool_call.function.arguments)

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(result),
                }
            )

    print()


def example_langchain_meta_tools() -> None:
    """Meta tools with LangChain.

    The meta tools convert to LangChain format just like any other Tools collection.
    """
    print("=" * 60)
    print("Example 2: Meta tools with LangChain")
    print("=" * 60)
    print()

    try:
        from langchain_core.tools import BaseTool  # noqa: F401
    except ImportError:
        print("Skipped: LangChain not installed. Install with: pip install langchain-core")
        print()
        return

    toolset = StackOneToolSet(search={"method": "semantic", "top_k": 3})
    meta_tools = toolset.get_meta_tools(account_ids=_account_ids or None)

    langchain_tools = meta_tools.to_langchain()

    print(f"Created {len(langchain_tools)} LangChain tools:")
    for tool in langchain_tools:
        print(f"  - {tool.name}: {tool.description}")
    print()
    print("These tools are ready to use with LangChain agents (AgentExecutor, create_react_agent, etc.)")
    print()


def main() -> None:
    """Run all meta tools examples."""
    api_key = os.getenv("STACKONE_API_KEY")
    if not api_key:
        print("Set STACKONE_API_KEY to run these examples.")
        return

    example_openai_meta_tools()
    example_langchain_meta_tools()


if __name__ == "__main__":
    main()
