"""Search tool patterns: callable wrapper and config overrides.

For semantic search basics, see semantic_search_example.py.
For full agent execution, see agent_tool_search.py.

Prerequisites:
    - STACKONE_API_KEY environment variable
    - STACKONE_ACCOUNT_ID environment variable

Run with:
    uv run python examples/search_tool_example.py
"""

from __future__ import annotations

import os

try:
    from dotenv import load_dotenv

    load_dotenv()
except ModuleNotFoundError:
    pass

from stackone_ai import StackOneToolSet


def main() -> None:
    api_key = os.getenv("STACKONE_API_KEY")
    account_id = os.getenv("STACKONE_ACCOUNT_ID")

    if not api_key:
        print("Set STACKONE_API_KEY to run this example.")
        return
    if not account_id:
        print("Set STACKONE_ACCOUNT_ID to run this example.")
        return

    # --- Example 1: get_search_tool() callable ---
    print("=== get_search_tool() callable ===\n")

    toolset = StackOneToolSet(api_key=api_key, account_id=account_id, search={})
    search_tool = toolset.get_search_tool()

    queries = ["cancel an event", "list employees", "send a message"]
    for query in queries:
        tools = search_tool(query, top_k=3)
        names = [t.name for t in tools]
        print(f'  "{query}" -> {", ".join(names) or "(none)"}')

    # --- Example 2: Constructor top_k vs per-call override ---
    print("\n=== Constructor top_k vs per-call override ===\n")

    toolset_3 = StackOneToolSet(api_key=api_key, account_id=account_id, search={"top_k": 3})

    query = "manage employee records"

    tools_3 = toolset_3.search_tools(query)
    print(f"Constructor top_k=3: got {len(tools_3)} tools")

    tools_override = toolset_3.search_tools(query, top_k=10)
    print(f"Per-call top_k=10 (overrides constructor 3): got {len(tools_override)} tools")


if __name__ == "__main__":
    main()
