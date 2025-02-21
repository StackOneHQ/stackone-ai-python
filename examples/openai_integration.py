"""
This example demonstrates how to use StackOne tools with OpenAI's function calling.

```bash
uv run examples/openai_integration.py
```
"""

import asyncio

from dotenv import load_dotenv
from openai import OpenAI
from stackone_ai import StackOneToolSet

load_dotenv()

account_id = "45072196112816593343"
employee_id = "c28xIQaWQ6MzM5MzczMDA2NzMzMzkwNzIwNA"


async def handle_tool_calls(tools, tool_calls) -> list[dict]:
    results = []
    for tool_call in tool_calls:
        tool = tools.get_tool(tool_call.function.name)
        if tool:
            results.append(tool.execute(tool_call.function.arguments))
    return results


async def main() -> None:
    client = OpenAI()
    toolset = StackOneToolSet()
    tools = toolset.get_tools(vertical="hris", account_id=account_id)
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

        results = await handle_tool_calls(tools, response.choices[0].message.tool_calls)
        if not results:
            print("Error: Failed to execute tools")
            break

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
    asyncio.run(main())
