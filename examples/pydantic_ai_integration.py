"""StackOne tools with Pydantic AI.

```bash
uv run examples/pydantic_ai_integration.py
```

Install with `pip install 'stackone-ai[pydantic-ai]'` (or
`pip install 'stackone-ai[examples]'` to run this file).
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
    print("Install pydantic-ai to run this example: pip install 'stackone-ai[pydantic-ai]'")
    raise SystemExit(1) from None

from stackone_ai import StackOneToolSet


def pydantic_ai_integration() -> None:
    for var in ["STACKONE_API_KEY", "STACKONE_ACCOUNT_ID", "OPENAI_API_KEY"]:
        if not os.getenv(var):
            print(f"Set {var} to run this example.")
            return

    toolset = StackOneToolSet()
    tools = toolset.fetch_tools(
        actions=["workday_list_workers", "workday_get_worker", "workday_get_current_user"],
        account_ids=[os.environ["STACKONE_ACCOUNT_ID"]],
    ).to_pydantic_ai()

    agent = Agent(
        "openai:gpt-5.4",
        system_prompt="You are a helpful HR assistant.",
        tools=tools,
    )
    result = agent.run_sync("List the first 5 employees")
    print(f"Result:\n{result.output}")


if __name__ == "__main__":
    pydantic_ai_integration()
