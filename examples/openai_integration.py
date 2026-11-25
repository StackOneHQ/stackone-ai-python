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

It uses the Responses API, which GPT-6 models need for function tools. More on
the format [here](https://platform.openai.com/docs/guides/function-calling).
"""

from __future__ import annotations

import sys
from typing import Any

try:
    from dotenv import load_dotenv

    load_dotenv()
except ModuleNotFoundError:
    pass

from openai import OpenAI

from stackone_ai import StackOneToolSet
from stackone_ai.types import StackOneError, ToolsetError

MODEL = "gpt-6.2-sol"
MAX_TURNS = 5


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

    # to_openai() returns Chat Completions tools. The Responses API takes the same
    # function definition one level up, so unwrap it. GPT-6 models need the Responses
    # API for function tools; Chat Completions refuses them unless reasoning is off.
    openai_tools: list[Any] = [{"type": "function", **t["function"]} for t in tools.to_openai()[:20]]
    names = {t["name"] for t in openai_tools}
    print(f"Loaded {len(openai_tools)} tools, e.g. {sorted(names)[:3]}")

    instructions = "You answer questions by calling the tools you are given."
    response = client.responses.create(
        model=MODEL,
        instructions=instructions,
        input="Use one of your tools to list a few records, then summarise them.",
        tools=openai_tools,
    )

    # The model may call tools more than once (paging, say), so loop until it answers.
    # Bounded, so a model that keeps calling tools cannot run up a bill.
    calls_made = 0
    for turn in range(MAX_TURNS + 1):
        function_calls = [item for item in response.output if item.type == "function_call"]
        if not function_calls:
            break
        if turn == MAX_TURNS:
            raise SystemExit(f"The model was still calling tools after {MAX_TURNS} turns.")
        outputs: list[Any] = []
        for call in function_calls:
            calls_made += 1
            print(f"Tool call: {call.name}({call.arguments})")
            # execute_openai_tool_calls() reads the Chat Completions shape; give it that
            # shape and send its content back as the Responses API's function_call_output.
            [message] = tools.execute_openai_tool_calls(
                [{"id": call.call_id, "function": {"name": call.name, "arguments": call.arguments}}]
            )
            print(f"  Result: {message['content'][:200]}...")
            outputs.append(
                {"type": "function_call_output", "call_id": call.call_id, "output": message["content"]}
            )

        # previous_response_id carries the model's turn, reasoning included, so only the
        # tool results need sending.
        response = client.responses.create(
            model=MODEL,
            instructions=instructions,
            previous_response_id=response.id,
            input=outputs,
            tools=openai_tools,
        )

    if calls_made == 0:
        raise SystemExit("The model answered without calling a tool, so the example proved nothing.")
    print(f"LLM made {calls_made} tool call(s).")
    print(f"Final response:\n{response.output_text}")


if __name__ == "__main__":
    try:
        openai_integration()
    except ToolsetError as exc:
        sys.exit(f"Could not load tools: {exc}")
    except StackOneError as exc:
        sys.exit(f"StackOne API error: {exc}")
