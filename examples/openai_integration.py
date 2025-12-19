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

    # Filter tools to only the ones we need to avoid context window limits
    tools = toolset.fetch_tools(
        actions=[
            "bamboohr_get_employee",
            "bamboohr_list_employee_employments",
            "bamboohr_get_employee_employment",
        ],
        account_ids=[account_id],
    )
    openai_tools = tools.to_openai()

    messages = [
        {"role": "system", "content": "You are a helpful HR assistant."},
        {
            "role": "user",
            "content": f"Can you get me information about employee with ID: {employee_id}?",
        },
    ]

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        tools=openai_tools,
        tool_choice="auto",
    )

    # Verify we got a response with tool calls
    assert response.choices[0].message.tool_calls is not None, "Expected tool calls in response"

    # Handle the tool calls and verify results
    results = handle_tool_calls(tools, response.choices[0].message.tool_calls)
    assert results is not None and len(results) > 0, "Expected tool call results"
    assert "data" in results[0], "Expected data in tool call result"

    # Verify we can continue the conversation with the results
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

    # Verify the final response
    final_response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        tools=openai_tools,
        tool_choice="auto",
    )
    assert final_response.choices[0].message.content is not None, "Expected final response content"


if __name__ == "__main__":
    openai_integration()
