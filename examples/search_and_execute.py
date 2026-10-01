"""
Find an action in natural language and run it, without loading a tool catalog.

This example is runnable with the following command:
```bash
uv run examples/search_and_execute.py
```

Prerequisites: STACKONE_API_KEY in .env and at least one active linked account.

Expected output: the ranked actions matching the query, the JSON Schema for the
best one, the arguments taken from its example_request, the result of executing it,
and whether feedback was recorded.

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
from stackone_ai.types import StackOneError, ToolsetError, ToolsetLoadError


def _score(action: dict) -> float:
    """A hit's similarity_score as a number: 0 when absent, NaN when not numeric."""
    score = action.get("similarity_score")
    try:
        return float(0 if score is None else score)
    except (TypeError, ValueError):
        return float("nan")


def search_and_execute() -> None:
    toolset = StackOneToolSet()

    for account in toolset.fetch_accounts():
        print(f"{account['id']}  {account['provider']}  {account['status']}")

    actions = toolset.search("list recent comments", top_k=3)
    if not actions:
        raise SystemExit("No actions matched. Try a different query, or link an account.")

    print("\nRanked matches:")
    for action in actions:
        print(f"  {_score(action):.3f}  {action['action_id']}")

    best = actions[0]

    # input_schema is how you find out what an action accepts. Build the call from it
    # and from example_request: arguments that do not match are dropped by the server
    # without an error, so a guessed parameter looks like it worked.
    print(f"\ninput_schema for {best['action_id']}:")
    print(json.dumps(best.get("input_schema", {}), indent=2)[:600])

    # example_request is a call the server accepts for this action, in the nested form
    # execute() takes. Start from it rather than guessing; execute() sets action_id itself.
    arguments = dict(best.get("example_request") or {})
    arguments.pop("action_id", None)
    print(f"\narguments: {json.dumps(arguments)}")

    # session_id links this call, and the feedback below, to the search that found
    # the action. It is optional: leave it out and each call stands alone.
    session_id = best.get("session_id")
    result = toolset.execute(best["action_id"], arguments, session_id=session_id)
    print(f"\nresult: {json.dumps(result, default=str)[:300]}...")

    try:
        toolset.submit_feedback(
            "positive", [best["action_id"]], feedback="Found it first try", session_id=session_id
        )
        print("\nfeedback recorded")
    except ToolsetLoadError as exc:
        # Raised when feedback is not enabled for this project. Not worth failing over.
        print(f"\nfeedback skipped: {exc}")


if __name__ == "__main__":
    try:
        search_and_execute()
    except ToolsetError as exc:
        sys.exit(f"Could not reach the StackOne API: {exc}")
    except StackOneError as exc:
        sys.exit(f"StackOne API error: {exc}")
