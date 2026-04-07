"""Workday integration: timeout and account scoping for slow providers.

Workday can take 10-15s to respond. This example shows how to configure
timeout and account_ids through the execute config.

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

account_id = os.getenv("STACKONE_ACCOUNT_ID", "")

# Timeout and account_ids both live in the execute config
toolset = StackOneToolSet(
    search={"method": "auto", "top_k": 5},
    execute={"account_ids": [account_id], "timeout": 120},
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
            tool = toolset.execute(tc.function.name, tc.function.arguments)
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": json.dumps(tool)})


# --- Example 1: Search and execute mode ---
# LLM gets tool_search + tool_execute, discovers tools autonomously
print("=== Search and execute mode ===\n")
run_agent(
    messages=[
        {"role": "system", "content": "Use tool_search to find tools, then tool_execute to run them."},
        {"role": "user", "content": "List the first 5 employees."},
    ],
    tools=toolset.openai(mode="search_and_execute"),
)

# --- Example 2: Normal mode ---
# Fetch specific tools upfront, pass to LLM
print("\n=== Normal mode ===\n")
tools = toolset.fetch_tools(actions=["workday_*_employee*"])
if len(tools) == 0:
    print("No Workday tools found for this account.")
else:
    run_agent(
        messages=[{"role": "user", "content": "List the first 5 employees."}],
        tools=tools.to_openai(),
    )
