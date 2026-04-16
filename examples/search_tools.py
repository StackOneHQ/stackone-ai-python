"""
Tool discovery with the StackOne AI SDK.

Covers the full spectrum of tool discovery: direct fetch, semantic search,
local search, auto search, and the search-and-execute pattern with OpenAI.

```bash
uv run examples/search_tools.py
```
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


def direct_fetch(account_id: str) -> None:
    """Fetch tools by action glob pattern -- no search involved."""
    print("=== Direct Fetch ===")

    toolset = StackOneToolSet()
    tools = toolset.fetch_tools(actions=["workday_*"], account_ids=[account_id])

    print(f"Fetched {len(tools)} tools matching 'workday_*'")
    for tool in tools:
        print(f"  - {tool.name}")

    tool = tools.get_tool("workday_list_workers")
    if tool:
        print(f"Found tool: {tool.name}")
        print(f"  Description: {tool.description[:80]}...")
    else:
        print("Tool 'workday_list_workers' not found in fetched tools")


def semantic_search(account_id: str) -> None:
    """Search tools using the semantic search API (default)."""
    print("\n=== Semantic Search ===")

    toolset = StackOneToolSet(search={"method": "semantic", "top_k": 5})
    tools = toolset.search_tools("manage employees", account_ids=[account_id], top_k=5)

    print(f"Semantic search returned {len(tools)} tools for 'manage employees'")
    for tool in tools:
        print(f"  - {tool.name}: {tool.description[:60]}...")


def local_search(account_id: str) -> None:
    """Search tools using local BM25+TF-IDF (no API call to search endpoint)."""
    print("\n=== Local Search ===")

    toolset = StackOneToolSet(search={"method": "local", "top_k": 5})
    tools = toolset.search_tools("create time off request", account_ids=[account_id], search="local")

    print(f"Local search returned {len(tools)} tools for 'create time off request'")
    for tool in tools:
        print(f"  - {tool.name}: {tool.description[:60]}...")


def auto_search(account_id: str) -> None:
    """Auto search: tries semantic first, falls back to local on failure."""
    print("\n=== Auto Search ===")

    toolset = StackOneToolSet(search={"method": "auto", "top_k": 5})
    tools = toolset.search_tools("list employee compensation", account_ids=[account_id], search="auto")

    print(f"Auto search returned {len(tools)} tools for 'list employee compensation'")
    for tool in tools:
        print(f"  - {tool.name}: {tool.description[:60]}...")

    # The get_search_tool() helper returns a callable for agent loops
    search_tool = toolset.get_search_tool()
    tools = search_tool("send a message", top_k=3, account_ids=[account_id])
    print(f"SearchTool callable returned {len(tools)} tools for 'send a message'")


def search_and_execute(account_id: str) -> None:
    """Two-tool pattern: tool_search + tool_execute driven by an OpenAI model."""
    print("\n=== Search & Execute (OpenAI) ===")

    try:
        from openai import OpenAI
    except ImportError:
        print("Skipping: 'openai' package is required (pip install openai)")
        return

    if not os.getenv("OPENAI_API_KEY"):
        print("Skipping: OPENAI_API_KEY is required")
        return

    # Configure toolset with search + execute defaults
    toolset = StackOneToolSet(
        search={"method": "semantic", "top_k": 3},
        execute={"account_ids": [account_id]},
        timeout=120,  # increase for slow providers (default: 60s)
    )
    openai_tools = toolset.openai(mode="search_and_execute")

    print(f"Registered {len(openai_tools)} meta-tools with OpenAI:")
    for t in openai_tools:
        print(f"  - {t['function']['name']}")

    # Simple agent loop
    client = OpenAI()
    messages: list[dict] = [
        {"role": "system", "content": "You are a helpful HR assistant."},
        {"role": "user", "content": "Find tools that can list employees and then list them."},
    ]

    for step in range(5):
        response = client.chat.completions.create(
            model="gpt-5.1",
            messages=messages,
            tools=openai_tools,
            tool_choice="auto",
        )
        message = response.choices[0].message

        if not message.tool_calls:
            print(f"Assistant: {message.content}")
            break

        # Append the assistant message (with tool calls) to history
        messages.append({"role": "assistant", "content": message.content, "tool_calls": message.tool_calls})

        for tool_call in message.tool_calls:
            name = tool_call.function.name
            args = json.loads(tool_call.function.arguments)
            print(f"  Step {step + 1}: calling {name}({json.dumps(args, indent=2)})")

            result = toolset.execute(name, args)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(result),
                }
            )

    print("Agent loop complete.")


def main() -> None:
    if not os.getenv("STACKONE_API_KEY"):
        print("Set STACKONE_API_KEY to run this example.")
        return

    account_id = os.getenv("STACKONE_ACCOUNT_ID")
    if not account_id:
        print("Set STACKONE_ACCOUNT_ID to run this example.")
        return

    direct_fetch(account_id)
    semantic_search(account_id)
    local_search(account_id)
    auto_search(account_id)
    search_and_execute(account_id)


if __name__ == "__main__":
    main()
