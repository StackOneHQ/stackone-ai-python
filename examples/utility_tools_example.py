#!/usr/bin/env python
"""
Example demonstrating utility tools for dynamic tool discovery and execution.

Utility tools allow AI agents to search for relevant tools based on natural language queries
and execute them dynamically without hardcoding tool names.

Prerequisites:
- STACKONE_API_KEY environment variable set
- STACKONE_ACCOUNT_ID environment variable set (comma-separated for multiple)
- At least one linked account in StackOne (this example uses BambooHR)

This example is runnable with the following command:
```bash
uv run examples/utility_tools_example.py
```
"""

import os

from stackone_ai import StackOneToolSet

try:
    from dotenv import load_dotenv

    load_dotenv()
except ModuleNotFoundError:
    pass

# Read account IDs from environment — supports comma-separated values
_account_ids = [aid.strip() for aid in os.getenv("STACKONE_ACCOUNT_ID", "").split(",") if aid.strip()]


def example_utility_tools_basic():
    """Basic example of using utility tools for tool discovery"""
    print("Example 1: Dynamic tool discovery\n")

    # Initialize StackOne toolset
    toolset = StackOneToolSet()

    # Get all available tools using MCP-backed fetch_tools()
    all_tools = toolset.fetch_tools(account_ids=_account_ids)
    print(f"Total tools available: {len(all_tools)}")

    if not all_tools:
        print("No tools found. Check your linked accounts.")
        return

    # Get utility tools for dynamic discovery
    utility_tools = all_tools.utility_tools()

    # Search for employee management tools
    result = utility_tools.get_search_tool()(query="manage employees create update list", top_k=5)

    print("Found relevant tools:")
    for tool in result.get("tools", []):
        print(f"  - {tool['name']} (score: {tool['score']:.2f}): {tool['description']}")

    print()


def example_utility_tools_with_execution():
    """Example of discovering and executing tools dynamically"""
    print("Example 2: Dynamic tool execution\n")

    # Initialize toolset
    toolset = StackOneToolSet()

    # Get all tools using MCP-backed fetch_tools()
    all_tools = toolset.fetch_tools(account_ids=_account_ids)

    if not all_tools:
        print("No tools found. Check your linked accounts.")
        return

    utility_tools = all_tools.utility_tools()

    # Step 1: Search for relevant tools
    search_result = utility_tools.get_search_tool()(query="list all employees", top_k=1)

    tools_found = search_result.get("tools", [])
    if tools_found:
        best_tool = tools_found[0]
        print(f"Best matching tool: {best_tool['name']}")
        print(f"Description: {best_tool['description']}")
        print(f"Relevance score: {best_tool['score']:.2f}")

        # Step 2: Execute the found tool
        try:
            print(f"\nExecuting {best_tool['name']}...")
            result = utility_tools.get_execute_tool()(toolName=best_tool["name"], params={"limit": 5})
            print(f"Execution result: {result}")
        except Exception as e:
            print(f"Execution failed (expected in example): {e}")

    print()


def example_with_openai():
    """Example of using utility tools with OpenAI"""
    print("Example 3: Using utility tools with OpenAI\n")

    try:
        from openai import OpenAI

        # Initialize OpenAI client
        client = OpenAI()

        # Initialize StackOne toolset
        toolset = StackOneToolSet()

        # Get BambooHR tools and their utility tools using MCP-backed fetch_tools()
        bamboohr_tools = toolset.fetch_tools(account_ids=_account_ids, actions=["bamboohr_*"])
        utility_tools = bamboohr_tools.utility_tools()

        # Convert to OpenAI format
        openai_tools = utility_tools.to_openai()

        # Create a chat completion with utility tools
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {
                    "role": "system",
                    "content": "You are an HR assistant. Use tool_search to find appropriate tools, then tool_execute to execute them.",
                },
                {"role": "user", "content": "Can you help me find tools for managing employee records?"},
            ],
            tools=openai_tools,
            tool_choice="auto",
        )

        print("OpenAI Response:", response.choices[0].message.content)

        if response.choices[0].message.tool_calls:
            print("\nTool calls made:")
            for tool_call in response.choices[0].message.tool_calls:
                print(f"  - {tool_call.function.name}")

    except ImportError:
        print("OpenAI library not installed. Install with: pip install openai")
    except Exception as e:
        print(f"OpenAI example failed: {e}")

    print()


def example_with_langchain():
    """Example of using tools with LangChain"""
    print("Example 4: Using tools with LangChain\n")

    try:
        from langchain.agents import AgentExecutor, create_tool_calling_agent
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_openai import ChatOpenAI

        # Initialize StackOne toolset
        toolset = StackOneToolSet()

        # Get tools and convert to LangChain format using MCP-backed fetch_tools()
        tools = toolset.fetch_tools(account_ids=_account_ids, actions=["bamboohr_list_*"])
        langchain_tools = tools.to_langchain()

        # Get utility tools as well
        utility_tools = tools.utility_tools()
        langchain_utility_tools = utility_tools.to_langchain()

        # Combine all tools
        all_langchain_tools = list(langchain_tools) + list(langchain_utility_tools)

        print(f"Available tools for LangChain: {len(all_langchain_tools)}")
        for tool in all_langchain_tools:
            print(f"  - {tool.name}: {tool.description}")

        # Create LangChain agent
        llm = ChatOpenAI(model="gpt-4", temperature=0)

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are an HR assistant. Use the utility tools to discover and execute relevant tools.",
                ),
                ("human", "{input}"),
                ("placeholder", "{agent_scratchpad}"),
            ]
        )

        agent = create_tool_calling_agent(llm, all_langchain_tools, prompt)
        agent_executor = AgentExecutor(agent=agent, tools=all_langchain_tools, verbose=True)

        # Run the agent
        result = agent_executor.invoke({"input": "Find tools that can list employee data"})

        print(f"\nAgent result: {result['output']}")

    except ImportError as e:
        print(f"LangChain dependencies not installed: {e}")
        print("Install with: pip install langchain-openai")
    except Exception as e:
        print(f"LangChain example failed: {e}")

    print()


def main():
    """Run all examples"""
    print("=" * 60)
    print("StackOne AI SDK - Utility Tools Examples")
    print("=" * 60)
    print()

    if not os.getenv("STACKONE_API_KEY"):
        print("Set STACKONE_API_KEY to run these examples.")
        return

    if not _account_ids:
        print("Set STACKONE_ACCOUNT_ID to run these examples.")
        print("(Comma-separated for multiple accounts)")
        return

    # Basic examples that work without external APIs
    example_utility_tools_basic()
    example_utility_tools_with_execution()

    # Examples that require OpenAI API
    if os.getenv("OPENAI_API_KEY"):
        example_with_openai()
        example_with_langchain()
    else:
        print("Set OPENAI_API_KEY to run OpenAI and LangChain examples\n")

    print("=" * 60)
    print("Examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
