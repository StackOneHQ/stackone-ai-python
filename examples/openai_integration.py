"""
This example demonstrates how to use StackOne tools with OpenAI's function calling.

This example is runnable with the following command:
```bash
uv run examples/openai_integration.py
```

You can find out more about the OpenAI Function Calling API format [here](https://platform.openai.com/docs/guides/function-calling).
"""

from dotenv import load_dotenv
from openai import OpenAI
from stackone_ai import StackOneToolSet

load_dotenv()

account_id = "45072196112816593343"
employee_id = "c28xIQaWQ6MzM5MzczMDA2NzMzMzkwNzIwNA"


def handle_tool_calls(tools, tool_calls) -> list[dict]:
    results = []
    for tool_call in tool_calls:
        tool = tools.get_tool(tool_call.function.name)
        if tool:
            results.append(tool.execute(tool_call.function.arguments))
    return results


def openai_integration() -> None:
    client = OpenAI()
    toolset = StackOneToolSet()

    all_tools = toolset.get_tools(vertical="hris", account_id=account_id)

    needed_tool_names = [
        "hris_get_employee",
        "hris_list_employee_employments",
        "hris_get_employee_employment",
    ]

    # Filter tools to only the ones we need
    # We need this because otherwise we can go over a context window limit
    # TODO: better filtering options.
    filtered_tools = [tool for tool in all_tools.tools if tool.name in needed_tool_names]
    tools = type(all_tools)(filtered_tools)
    openai_tools = tools.to_openai()

    messages = [
        {"role": "system", "content": "You are a helpful HR assistant."},
        {
            "role": "user",
            "content": f"Can you get me information about employee with ID: {employee_id}?",
        },
    ]

    while True:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            tools=openai_tools,
            tool_choice="auto",
        )

        if not response.choices[0].message.tool_calls:
            print("Response:", response.choices[0].message.content)
            break

        results = handle_tool_calls(tools, response.choices[0].message.tool_calls)
        assert results is not None

        messages.extend(
            [
                {"role": "assistant", "content": None, "tool_calls": response.choices[0].message.tool_calls},
                {
                    "role": "tool",
                    "tool_call_id": response.choices[0].message.tool_calls[0].id,
                    "content": str(results[0]),
                },
            ]
        )


if __name__ == "__main__":
    openai_integration()
