"""Workday integration: timeout and account scoping for slow providers.

Workday can take 10-15s to respond. This example shows how to configure
timeout and account_ids through the execute config.

Prerequisites:
    - STACKONE_API_KEY environment variable
    - STACKONE_ACCOUNT_ID environment variable (a Workday-connected account)
    - OPENAI_API_KEY environment variable

Run with:
    uv run python examples/workday_integration.py
"""

from __future__ import annotations

import json
import os

try:
    from dotenv import load_dotenv

    load_dotenv()
except ModuleNotFoundError:
    pass

from openai import OpenAI

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

    # Timeout for slow providers, account_id for scoping
    toolset = StackOneToolSet(
        api_key=api_key,
        account_id=account_id,
        search={"method": "auto", "top_k": 5},
        timeout=120,
    )
    client = OpenAI()

    def run_agent(messages: list[dict], tools: list[dict], max_steps: int = 10) -> None:
        """Simple agent loop: call LLM, execute tools, repeat."""
        for _ in range(max_steps):
            response = client.chat.completions.create(model="gpt-5.4", messages=messages, tools=tools)
            choice = response.choices[0]

            if not choice.message.tool_calls:
                print(f"Answer: {choice.message.content}")
                return

            messages.append(choice.message.model_dump(exclude_none=True))
            for tc in choice.message.tool_calls:
                print(f"  -> {tc.function.name}({tc.function.arguments[:80]})")
                result = toolset.execute(tc.function.name, tc.function.arguments)
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": json.dumps(result)})

    # --- Example 1: Search and execute mode ---
    print("=== Search and execute mode ===\n")
    run_agent(
        messages=[
            {"role": "system", "content": "Use tool_search to find tools, then tool_execute to run them."},
            {"role": "user", "content": "List the first 5 employees."},
        ],
        tools=toolset.openai(mode="search_and_execute"),
    )

    # --- Example 2: Normal mode ---
    print("\n=== Normal mode ===\n")
    tools = toolset.fetch_tools(actions=["workday_*_employee*"])
    if len(tools) == 0:
        print("No Workday tools found for this account.")
    else:
        run_agent(
            messages=[{"role": "user", "content": "List the first 5 employees."}],
            tools=tools.to_openai(),
        )


if __name__ == "__main__":
    main()
