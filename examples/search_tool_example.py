"""Search tool patterns: callable wrapper and config overrides.

For semantic search basics, see semantic_search_example.py.
For full agent execution, see agent_tool_search.py.

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
    account_id = os.getenv("STACKONE_ACCOUNT_ID", "")
    _account_ids = [a.strip() for a in account_id.split(",") if a.strip()] if account_id else []

    # --- Example 1: get_search_tool() callable ---
    print("=== get_search_tool() callable ===\n")

    toolset = StackOneToolSet(search={})
    search_tool = toolset.get_search_tool()

    queries = ["cancel an event", "list employees", "send a message"]
    for query in queries:
        tools = search_tool(query, top_k=3, account_ids=_account_ids)
        names = [t.name for t in tools]
        print(f'  "{query}" -> {", ".join(names) or "(none)"}')

    # --- Example 2: Constructor top_k vs per-call override ---
    print("\n=== Constructor top_k vs per-call override ===\n")

    toolset_3 = StackOneToolSet(search={"top_k": 3})
    toolset_10 = StackOneToolSet(search={"top_k": 10})

    query = "manage employee records"

    tools_3 = toolset_3.search_tools(query, account_ids=_account_ids)
    print(f"Constructor top_k=3: got {len(tools_3)} tools")

    tools_10 = toolset_10.search_tools(query, account_ids=_account_ids)
    print(f"Constructor top_k=10: got {len(tools_10)} tools")

    # Per-call override: constructor says 3 but this call says 10
    tools_override = toolset_3.search_tools(query, top_k=10, account_ids=_account_ids)
    print(f"Per-call top_k=10 (overrides constructor 3): got {len(tools_override)} tools")


if __name__ == "__main__":
    main()
