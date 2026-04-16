"""
Comprehensive auth management example showing all ways to configure API keys and account IDs.

```bash
uv run examples/auth_management.py
```
"""

from __future__ import annotations

import os

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

from stackone_ai import StackOneToolSet


def api_key_setup() -> None:
    """Demonstrate the two ways to provide an API key."""
    print("\n--- API Key Setup ---")

    # 1. Default: reads STACKONE_API_KEY from the environment
    toolset_from_env = StackOneToolSet()
    print("Created toolset from STACKONE_API_KEY env var (default)")

    # 2. Explicit: pass the key directly
    api_key = os.environ["STACKONE_API_KEY"]
    toolset_explicit = StackOneToolSet(api_key=api_key)
    print("Created toolset with explicit api_key")

    # Both toolsets are functionally equivalent
    print(f"Toolset from env:      {toolset_from_env}")
    print(f"Toolset from explicit: {toolset_explicit}")


def single_account(account_id: str) -> None:
    """Fetch tools scoped to a single account."""
    print("\n--- Single Account via fetch_tools ---")

    toolset = StackOneToolSet()
    tools = toolset.fetch_tools(actions=["workday_*"], account_ids=[account_id])
    print(f"Fetched {len(tools)} tools for configured account")


def set_accounts_globally(account_id: str) -> None:
    """Use set_accounts() to bind accounts at the toolset level."""
    print("\n--- set_accounts() (global on toolset) ---")

    toolset = StackOneToolSet()
    toolset.set_accounts([account_id])
    print("Set global accounts via set_accounts()")

    tools = toolset.fetch_tools(actions=["workday_*"])
    print(f"Fetched {len(tools)} tools (account inherited from toolset)")


def per_tool_override(account_id: str) -> None:
    """Override the account ID on a tools collection or an individual tool."""
    print("\n--- Per-Tool Account Override ---")

    toolset = StackOneToolSet()
    tools = toolset.fetch_tools(actions=["workday_*"], account_ids=[account_id])

    # Override on the entire tools collection
    tools.set_account_id("overridden-account")
    print("Called tools.set_account_id('overridden-account')")

    # Override on a single tool
    tool = tools.get_tool("workday_list_workers")
    if tool:
        tool.set_account_id("per-tool-account")
        current = tool.get_account_id()
        print(f"Single tool account_id is now: '{current}'")
    else:
        print("Tool 'workday_list_workers' not found -- skipping single-tool override")


def main() -> None:
    api_key = os.getenv("STACKONE_API_KEY")
    if not api_key:
        print("STACKONE_API_KEY is not set. Please export it or add it to .env")
        return

    account_id = os.getenv("STACKONE_ACCOUNT_ID")
    if not account_id:
        print("Set STACKONE_ACCOUNT_ID to run this example.")
        return

    api_key_setup()
    single_account(account_id)
    set_accounts_globally(account_id)
    per_tool_override(account_id)

    print("\nAll auth management examples completed.")


if __name__ == "__main__":
    main()
