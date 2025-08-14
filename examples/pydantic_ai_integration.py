"""
This example demonstrates how to use StackOne tools with Pydantic AI.

```bash
uv run examples/pydantic_ai_integration.py
```
"""

import asyncio

from dotenv import load_dotenv
from pydantic_ai import Agent

from stackone_ai import StackOneToolSet

load_dotenv()

account_id = "45072196112816593343"
employee_id = "c28xIQaWQ6MzM5MzczMDA2NzMzMzkwNzIwNA"


async def pydantic_ai_integration() -> None:
    """Example of using StackOne tools with Pydantic AI"""
    # Initialize StackOne toolset
    toolset = StackOneToolSet()
    tools = toolset.get_tools("hris_*", account_id=account_id)

    # Convert to Pydantic AI format
    pydantic_ai_tools = tools.to_pydantic_ai()
    assert len(pydantic_ai_tools) > 0, "Expected at least one Pydantic AI tool"

    # Create a Pydantic AI agent with StackOne tools
    agent = Agent(
        model="openai:gpt-4o-mini",
        tools=pydantic_ai_tools,
    )

    # Test the integration
    result = await agent.run(
        f"Can you get me information about employee with ID: {employee_id}? Use the HRIS tools."
    )

    print("Agent response:", result.data)

    # Verify tool calls were made
    assert result.all_messages(), "Expected messages from the agent"
    tool_calls_made = any(msg.kind == "tool_call" for msg in result.all_messages())
    print(f"Tool calls were made: {tool_calls_made}")


if __name__ == "__main__":
    asyncio.run(pydantic_ai_integration())
