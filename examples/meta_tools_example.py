#!/usr/bin/env python
"""
Example demonstrating meta tools for dynamic tool discovery and execution.

Meta tools allow AI agents to search for relevant tools based on natural language queries
and execute them dynamically without hardcoding tool names.
"""

import os

from dotenv import load_dotenv

from stackone_ai import StackOneToolSet

# Load environment variables
load_dotenv()


def example_meta_tools_basic():
    """Basic example of using meta tools for tool discovery"""
    print("Example 1: Dynamic tool discovery\n")

    # Initialize StackOne toolset
    toolset = StackOneToolSet()

    # Get all available tools using MCP-backed fetch_tools()
    all_tools = toolset.fetch_tools(actions=["bamboohr_*"])
    print(f"Total BambooHR tools available: {len(all_tools)}")

    # Get meta tools for dynamic discovery
    meta_tools = all_tools.meta_tools()

    # Get the filter tool to search for relevant tools
    filter_tool = meta_tools.get_tool("meta_search_tools")
    if filter_tool:
        # Search for employee management tools
        result = filter_tool.call(query="manage employees create update list", limit=5, minScore=0.0)

        print("Found relevant tools:")
        for tool in result.get("tools", []):
            print(f"  - {tool['name']} (score: {tool['score']:.2f}): {tool['description']}")

    print()


def example_meta_tools_with_execution():
    """Example of discovering and executing tools dynamically"""
    print("Example 2: Dynamic tool execution\n")

    # Initialize toolset
    toolset = StackOneToolSet()

    # Get all tools using MCP-backed fetch_tools()
    all_tools = toolset.fetch_tools()
    meta_tools = all_tools.meta_tools()

    # Step 1: Search for relevant tools
    filter_tool = meta_tools.get_tool("meta_search_tools")
    execute_tool = meta_tools.get_tool("meta_execute_tool")

    if filter_tool and execute_tool:
        # Find tools for listing employees
        search_result = filter_tool.call(query="list all employees", limit=1)

        tools_found = search_result.get("tools", [])
        if tools_found:
            best_tool = tools_found[0]
            print(f"Best matching tool: {best_tool['name']}")
            print(f"Description: {best_tool['description']}")
            print(f"Relevance score: {best_tool['score']:.2f}")

            # Step 2: Execute the found tool
            try:
                print(f"\nExecuting {best_tool['name']}...")
                result = execute_tool.call(toolName=best_tool["name"], params={"limit": 5})
                print(f"Execution result: {result}")
            except Exception as e:
                print(f"Execution failed (expected in example): {e}")

    print()


def example_with_openai():
    """Example of using meta tools with OpenAI"""
    print("Example 3: Using meta tools with OpenAI\n")

    try:
        from openai import OpenAI

        # Initialize OpenAI client
        client = OpenAI()

        # Initialize StackOne toolset
        toolset = StackOneToolSet()

        # Get BambooHR tools and their meta tools using MCP-backed fetch_tools()
        bamboohr_tools = toolset.fetch_tools(actions=["bamboohr_*"])
        meta_tools = bamboohr_tools.meta_tools()

        # Convert to OpenAI format
        openai_tools = meta_tools.to_openai()

        # Create a chat completion with meta tools
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {
                    "role": "system",
                    "content": "You are an HR assistant. Use meta_search_tools to find appropriate tools, then meta_execute_tool to execute them.",
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
        tools = toolset.fetch_tools(actions=["bamboohr_list_*"])
        langchain_tools = tools.to_langchain()

        # Get meta tools as well
        meta_tools = tools.meta_tools()
        langchain_meta_tools = meta_tools.to_langchain()

        # Combine all tools
        all_langchain_tools = list(langchain_tools) + list(langchain_meta_tools)

        print(f"Available tools for LangChain: {len(all_langchain_tools)}")
        for tool in all_langchain_tools:
            print(f"  - {tool.name}: {tool.description}")

        # Create LangChain agent
        llm = ChatOpenAI(model="gpt-4", temperature=0)

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are an HR assistant. Use the meta tools to discover and execute relevant tools.",
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
    print("StackOne AI SDK - Meta Tools Examples")
    print("=" * 60)
    print()

    # Basic examples that work without external APIs
    example_meta_tools_basic()
    example_meta_tools_with_execution()

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
