"""
This example demonstrates how to use StackOne tools with OpenAI's function calling.

This example is runnable with the following command:
```bash
uv run examples/openai_integration.py
```

You can find out more about the OpenAI Function Calling API format [here](https://platform.openai.com/docs/guides/function-calling).
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


def handle_tool_calls(tools, tool_calls) -> list[dict]:
    results = []
    for tool_call in tool_calls:
        tool = tools.get_tool(tool_call.function.name)
        if tool:
            results.append(tool.execute(tool_call.function.arguments))
    return results


def openai_integration() -> None:
    account_id = os.getenv("STACKONE_ACCOUNT_ID")
    if not os.getenv("STACKONE_API_KEY"):
        print("Set STACKONE_API_KEY to run this example.")
        return
    if not account_id:
        print("Set STACKONE_ACCOUNT_ID to run this example.")
        return

    client = OpenAI()
    toolset = StackOneToolSet()

    # Filter tools to only the ones we need to avoid context window limits
    tools = toolset.fetch_tools(
        actions=[
            "workday_list_workers",
            "workday_get_worker",
            "workday_get_current_user",
        ],
        account_ids=[account_id],
    )
    openai_tools = tools.to_openai()
    print(f"Loaded {len(openai_tools)} tools for OpenAI function calling.")

    messages = [
        {"role": "system", "content": "You are a helpful HR assistant."},
        {
            "role": "user",
            "content": "List the first 5 employees",
        },
    ]

    response = client.chat.completions.create(
        model="gpt-5.4",
        messages=messages,
        tools=openai_tools,
        tool_choice="auto",
    )

    tool_calls = response.choices[0].message.tool_calls
    if not tool_calls:
        print("No tool calls were made by the model.")
        return

    print(f"LLM made {len(tool_calls)} tool call(s):")
    for tc in tool_calls:
        print(f"  - {tc.function.name}({tc.function.arguments})")

    # Handle the tool calls
    results = handle_tool_calls(tools, tool_calls)
    print(f"Received {len(results)} tool call result(s).")
    for i, result in enumerate(results):
        print(f"  Result {i + 1}: {str(result)[:200]}...")

    # Continue the conversation with all tool call results
    messages.append(response.choices[0].message.model_dump(exclude_none=True))
    for tc, result in zip(tool_calls, results, strict=False):
        messages.append(
            {
                "role": "tool",
                "tool_call_id": tc.id,
                "content": json.dumps(result),
            }
        )

    final_response = client.chat.completions.create(
        model="gpt-5.4",
        messages=messages,
        tools=openai_tools,
        tool_choice="auto",
    )
    print(f"Final response:\n{final_response.choices[0].message.content}")


if __name__ == "__main__":
    openai_integration()
