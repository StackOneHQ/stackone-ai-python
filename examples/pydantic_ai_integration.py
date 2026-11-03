"""StackOne tools with Pydantic AI.

```bash
uv run examples/pydantic_ai_integration.py
```

Install with `uv add 'stackone-ai[pydantic-ai]'` (or
`uv add 'stackone-ai[examples]'` to run this file).
"""

from __future__ import annotations

import os

try:
    from dotenv import load_dotenv

    load_dotenv()
except ModuleNotFoundError:
    pass

try:
    from pydantic_ai import Agent
except ImportError:
    print("Install pydantic-ai to run this example: uv add 'stackone-ai[pydantic-ai]'")
    raise SystemExit(1) from None

from stackone_ai import StackOneToolSet


def individual_tools(account_id: str) -> None:
    """One Pydantic AI tool per action: suits a handful of actions."""
    tools = StackOneToolSet().fetch_tools(actions=["*_list_*"], account_ids=[account_id]).to_pydantic_ai()

    agent = Agent(
        "openai:gpt-5.4",
        system_prompt="You are a helpful assistant that answers by calling tools.",
        tools=tools,
    )
    result = agent.run_sync("Use one of your tools to list a few records, then summarise them.")
    print(f"Individual tools:\n{result.output}\n")


def search_and_execute(account_id: str) -> None:
    """StackOne's search and execute tools: the model finds an action, then runs it."""
    toolset = StackOneToolSet(tool_mode="search_execute")
    tools = toolset.fetch_tools(account_ids=[account_id]).to_pydantic_ai()

    agent = Agent(
        "openai:gpt-5.4",
        system_prompt=(
            "You answer by calling tools. First call stackone_search_actions to find an action, "
            "then call stackone_execute_action with its action_id and arguments built from its "
            "example_request."
        ),
        tools=tools,
    )
    result = agent.run_sync("List a few records from any of my connected apps, then summarise them.")
    print(f"Search and execute:\n{result.output}")


def pydantic_ai_integration() -> None:
    for var in ["STACKONE_API_KEY", "STACKONE_ACCOUNT_ID", "OPENAI_API_KEY"]:
        if not os.getenv(var):
            print(f"Set {var} to run this example.")
            return

    account_id = os.environ["STACKONE_ACCOUNT_ID"]
    individual_tools(account_id)
    search_and_execute(account_id)


if __name__ == "__main__":
    pydantic_ai_integration()
