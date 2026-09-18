"""
Find an action in natural language and run it, without loading a tool catalog.

This example is runnable with the following command:
```bash
uv run examples/search_and_execute.py
```

Prerequisites: STACKONE_API_KEY in .env and at least one active linked account.

Expected output: the ranked actions matching the query, the JSON Schema for the
best one, and the result of executing it.

This is the recommended way to use the SDK: search() asks every linked connector
and returns ranked actions, so a catalog of hundreds of tools never has to fit in
a model's context.
"""

from __future__ import annotations

import json
import sys

try:
    from dotenv import load_dotenv

    load_dotenv()
except ModuleNotFoundError:
    pass

from stackone_ai import StackOneToolSet
from stackone_ai.types import StackOneError, ToolsetError


def search_and_execute() -> None:
    toolset = StackOneToolSet()

    for account in toolset.fetch_accounts():
        print(f"{account['id']}  {account['provider']}  {account['status']}")

    actions = toolset.search("list recent comments", top_k=3)
    if not actions:
        raise SystemExit("No actions matched. Try a different query, or link an account.")

    print("\nRanked matches:")
    for action in actions:
        print(f"  {action.get('similarity_score', 0.0):.3f}  {action['action_id']}")

    best = actions[0]

    # input_schema is how you find out what an action accepts. Build the call from
    # it: arguments that do not match are dropped by the server without an error,
    # so a guessed parameter looks like it worked.
    print(f"\ninput_schema for {best['action_id']}:")
    print(json.dumps(best.get("input_schema", {}), indent=2)[:600])

    result = toolset.execute(best["action_id"], {"body": {"variables": {"first": 2}}})
    print(f"\nresult: {json.dumps(result, default=str)[:300]}...")


if __name__ == "__main__":
    try:
        search_and_execute()
    except ToolsetError as exc:
        sys.exit(f"Could not reach the StackOne API: {exc}")
    except StackOneError as exc:
        sys.exit(f"StackOne API error: {exc}")
