"""
Use StackOne tools with OpenAI function calling.

This example is runnable with the following command:
```bash
uv run examples/openai_integration.py
```

Prerequisites: STACKONE_API_KEY and OPENAI_API_KEY in .env, and at least one
active linked account. No specific provider is required — the example discovers
what your key can reach.

Expected output: the number of tools loaded, the tool call the model chose, the
result of running it, and the model's final answer. It exits non-zero if your
key reaches no tools, rather than pretending to succeed with an empty toolbox.

You can find out more about the OpenAI Function Calling API format
[here](https://platform.openai.com/docs/guides/function-calling).
"""

from __future__ import annotations

import json
import sys

try:
    from dotenv import load_dotenv

    load_dotenv()
except ModuleNotFoundError:
    pass

from openai import OpenAI

from stackone_ai import StackOneToolSet, Tools
from stackone_ai.types import StackOneError, ToolsetError


def handle_tool_calls(tools: Tools, tool_calls: list) -> list[dict]:
    """Run each tool call the model asked for, returning one result per call."""
    results: list[dict] = []
    for tool_call in tool_calls:
        tool = tools.get_tool(tool_call.function.name)
        if tool is None:
            results.append({"error": f"model called unknown tool {tool_call.function.name}"})
            continue
        try:
            results.append(tool.execute(tool_call.function.arguments))
        except StackOneError as exc:
            # Hand the failure back to the model rather than crashing the loop:
            # it can retry with different arguments or explain what went wrong.
            results.append({"error": str(exc)})
    return results


def openai_integration() -> None:
    client = OpenAI()
    toolset = StackOneToolSet()

    # Read-only list actions, whichever provider this key happens to reach. Keeping
    # the filter narrow matters: an unfiltered catalog is hundreds of tools and will
    # not fit a model's context.
    tools = toolset.fetch_tools(actions=["*_list_*"])
    if not tools:
        raise SystemExit(
            "No tools matched. Check your linked accounts with toolset.fetch_accounts() — "
            "an empty catalog here means the example would prove nothing."
        )

    openai_tools = tools.to_openai()[:20]
    names = {t["function"]["name"] for t in openai_tools}
    print(f"Loaded {len(openai_tools)} tools, e.g. {sorted(names)[:3]}")

    messages: list[dict] = [
        {"role": "system", "content": "You answer questions by calling the tools you are given."},
        {
            "role": "user",
            "content": "Use one of your tools to list a few records, then summarise them.",
        },
    ]

    response = client.chat.completions.create(
        model="gpt-5.4", messages=messages, tools=openai_tools, tool_choice="auto"
    )

    tool_calls = response.choices[0].message.tool_calls
    if not tool_calls:
        print("No tool calls were made by the model.")
        return

    print(f"LLM made {len(tool_calls)} tool call(s):")
    for tc in tool_calls:
        print(f"  - {tc.function.name}({tc.function.arguments})")

    results = handle_tool_calls(tools, tool_calls)
    for i, result in enumerate(results):
        print(f"  Result {i + 1}: {str(result)[:200]}...")

    messages.append(response.choices[0].message.model_dump(exclude_none=True))
    for tc, result in zip(tool_calls, results, strict=False):
        messages.append({"role": "tool", "tool_call_id": tc.id, "content": json.dumps(result, default=str)})

    final_response = client.chat.completions.create(
        model="gpt-5.4", messages=messages, tools=openai_tools, tool_choice="auto"
    )
    print(f"Final response:\n{final_response.choices[0].message.content}")


if __name__ == "__main__":
    try:
        openai_integration()
    except ToolsetError as exc:
        sys.exit(f"Could not load tools: {exc}")
    except StackOneError as exc:
        sys.exit(f"StackOne API error: {exc}")
