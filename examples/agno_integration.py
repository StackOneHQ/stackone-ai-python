"""
This example demonstrates how to use StackOne tools with Agno agents.

This example is runnable with the following command:
```bash
uv run examples/agno_integration.py
```

You can find out more about Agno framework at https://docs.agno.com
"""

from agno.agent import Agent
from agno.models.openai import OpenAIChat
from dotenv import load_dotenv

from stackone_ai import StackOneToolSet

load_dotenv()

account_id = "45072196112816593343"
employee_id = "c28xIQaWQ6MzM5MzczMDA2NzMzMzkwNzIwNA"


def agno_integration() -> None:
    """Demonstrate StackOne tools with Agno agents"""
    toolset = StackOneToolSet()

    # Filter tools to only the ones we need to avoid context window limits
    tools = toolset.get_tools(
        [
            "hris_get_employee",
            "hris_list_employee_employments",
            "hris_get_employee_employment",
        ],
        account_id=account_id,
    )

    # Convert to Agno format
    agno_tools = tools.to_agno()

    # Create an Agno agent with the tools
    agent = Agent(
        name="HR Assistant Agent",
        role="Helpful HR assistant that can access employee data",
        model=OpenAIChat(id="gpt-4o-mini"),
        tools=agno_tools,
        instructions=[
            "You are a helpful HR assistant.",
            "Use the provided tools to access employee information.",
            "Always be professional and respectful when handling employee data.",
        ],
        show_tool_calls=True,
        markdown=True,
    )

    # Test the agent with a query
    query = f"Can you get me information about employee with ID: {employee_id}?"

    print(f"Query: {query}")
    print("Agent response:")

    response = agent.run(query)
    print(response.content)

    # Verify we got a meaningful response
    assert response.content is not None, "Expected response content"
    assert len(response.content) > 0, "Expected non-empty response"


def agno_async_integration() -> None:
    """Demonstrate async StackOne tools with Agno agents"""
    import asyncio

    async def run_async_agent() -> None:
        toolset = StackOneToolSet()

        # Filter tools to only the ones we need
        tools = toolset.get_tools(
            ["hris_get_employee"],
            account_id=account_id,
        )

        # Convert to Agno format
        agno_tools = tools.to_agno()

        # Create an Agno agent
        agent = Agent(
            name="Async HR Agent",
            role="Async HR assistant",
            model=OpenAIChat(id="gpt-4o-mini"),
            tools=agno_tools,
            instructions=["You are an async HR assistant."],
        )

        # Run the agent asynchronously
        query = f"Get employee information for ID: {employee_id}"
        response = await agent.arun(query)

        print(f"Async query: {query}")
        print("Async agent response:")
        print(response.content)

        # Verify response
        assert response.content is not None, "Expected async response content"

    # Run the async example
    asyncio.run(run_async_agent())


if __name__ == "__main__":
    print("=== StackOne + Agno Integration Demo ===\n")

    print("1. Basic Agno Integration:")
    agno_integration()

    print("\n2. Async Agno Integration:")
    agno_async_integration()

    print("\n=== Demo completed successfully! ===")
