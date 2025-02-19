import asyncio
import os

from dotenv import load_dotenv
from openai import OpenAI
from openai.types.chat import ChatCompletionMessage
from stackone_ai import StackOneToolSet
from stackone_ai.models import Tools

"""
This example demonstrates how to use StackOne tools with OpenAI's function calling.

The example shows:
1. Setting up StackOne tools for use with OpenAI
2. Converting tools to OpenAI's function format
3. Making chat completions with function calling
4. Handling tool execution results
5. Managing conversation context

Example OpenAI function format:
{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get current temperature for a given location.",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City and country e.g. Bogotá, Colombia"
                }
            },
            "required": ["location"],
            "additionalProperties": False
        },
        "strict": True
    }
}

Example tool call output:
[
    {
        "id": "call_12345xyz",
        "type": "function",
        "function": {
            "name": "get_weather",
            "arguments": "{\"location\":\"Paris, France\"}"
        }
    }
]
"""


async def handle_tool_calls(tools: Tools, tool_calls: list) -> list[dict]:
    """
    Execute tool calls and return results

    Args:
        tools: Tools instance containing available tools
        tool_calls: List of OpenAI tool calls

    Returns:
        List of tool execution results
    """
    results = []
    for tool_call in tool_calls:
        tool = tools.get_tool(tool_call.function.name)
        if not tool:
            print(f"Warning: Tool {tool_call.function.name} not found")
            continue

        result = tool.execute(tool_call.function.arguments)
        results.append(result)
    return results


async def process_chat_response(
    client: OpenAI,
    tools: Tools,
    message: ChatCompletionMessage,
    messages: list[dict],
) -> str:
    """
    Process chat completion response and handle any tool calls

    Args:
        client: OpenAI client
        tools: Tools instance containing available tools
        message: Message from chat completion
        messages: List of conversation messages

    Returns:
        Final response text
    """
    if message.tool_calls:
        # Execute the tool calls
        results = await handle_tool_calls(tools, message.tool_calls)

        if not results:
            return "Sorry, I couldn't find the appropriate tools to help with that."

        # Add results to conversation
        messages.append({"role": "assistant", "content": None, "tool_calls": message.tool_calls})
        messages.append(
            {
                "role": "tool",
                "tool_call_id": message.tool_calls[0].id,
                "content": str(results[0]),
            }
        )

        # Get final response
        final_response = client.chat.completions.create(
            model="gpt-4",
            messages=messages,
        )
        return final_response.choices[0].message.content

    return message.content


async def main() -> None:
    # Load environment variables
    load_dotenv()

    api_key = os.getenv("STACKONE_API_KEY")
    account_id = os.getenv("STACKONE_ACCOUNT_ID")
    openai_api_key = os.getenv("OPENAI_API_KEY")

    if not all([api_key, account_id, openai_api_key]):
        raise ValueError(
            "Missing required environment variables."
            + "Please set STACKONE_API_KEY, STACKONE_ACCOUNT_ID, and OPENAI_API_KEY"
        )

    try:
        # Initialize OpenAI and StackOne clients
        client = OpenAI(api_key=openai_api_key)
        toolset = StackOneToolSet(api_key=api_key)

        # Get HRIS tools and convert to OpenAI format
        tools = toolset.get_tools(vertical="hris", account_id=account_id)
        openai_tools = tools.to_openai()

        # Example chat completion with tools
        messages = [
            {"role": "system", "content": "You are a helpful HR assistant."},
            {
                "role": "user",
                "content": "Can you get me information about employee with ID:"
                + "c28xIQaWQ6MzM5MzczMDA2NzMzMzkwNzIwNA?",
            },
        ]

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            tools=openai_tools,
            tool_choice="auto",  # Let the model choose when to use tools
        )

        # Handle the response
        result = await process_chat_response(client, tools, response.choices[0].message, messages)
        print("Response:", result)

    except Exception as e:
        print(f"Error: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
