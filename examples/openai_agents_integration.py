"""
This example demonstrates how to use StackOne tools with OpenAI Agents SDK.

```bash
uv run examples/openai_agents_integration.py
```
"""

from agents import Agent, Runner
from dotenv import load_dotenv

from stackone_ai import StackOneToolSet

load_dotenv()

account_id = "45072196112816593343"
employee_id = "c28xIQaWQ6MzM5MzczMDA2NzMzMzkwNzIwNA"


def openai_agents_integration() -> None:
    """Example of using StackOne tools with OpenAI Agents SDK"""
    # Initialize StackOne toolset
    toolset = StackOneToolSet()
    tools = toolset.get_tools("hris_*", account_id=account_id)

    # Convert to OpenAI Agents format
    openai_agents_tools = tools.to_openai_agents()
    assert len(openai_agents_tools) > 0, "Expected at least one OpenAI Agents tool"

    # Create an OpenAI Agent with StackOne tools
    agent = Agent(
        name="HR Assistant",
        instructions="You are an HR assistant that can help with employee information using HRIS tools.",
        tools=openai_agents_tools,
    )

    # Test the integration
    result = Runner.run_sync(
        agent,
        f"Can you get me information about employee with ID: {employee_id}? Use the HRIS tools available to you.",
    )

    print("Agent response:", result.final_output)

    # Verify tool usage
    assert result.messages, "Expected messages from the agent"
    tool_calls_made = any(
        hasattr(msg, "tool_calls") and msg.tool_calls for msg in result.messages if hasattr(msg, "tool_calls")
    )
    print(f"Tool calls were made: {tool_calls_made}")


if __name__ == "__main__":
    openai_agents_integration()
