#!/usr/bin/env python
"""
Example demonstrating semantic search for AI-powered tool discovery.

Semantic search understands natural language intent and synonyms, so queries like
"fire someone" or "check my to-do list" resolve to the right StackOne actions —
unlike keyword matching which requires exact tool names.

This example is runnable with the following command:
```bash
uv run examples/semantic_search_example.py
```

Prerequisites:
- STACKONE_API_KEY environment variable set
- At least one linked account in StackOne
"""

import os

from dotenv import load_dotenv

from stackone_ai import StackOneToolSet

load_dotenv()


def example_search_tools():
    """High-level semantic search returning a Tools collection.

    search_tools() is the recommended way to use semantic search. It:
    1. Fetches all available tools from your linked accounts
    2. Queries the semantic search API with your natural language query
    3. Filters results to only connectors available in your accounts
    4. Returns a Tools collection ready for any framework (.to_openai(), .to_langchain(), etc.)
    """
    print("Example 1: search_tools() — high-level semantic search\n")

    toolset = StackOneToolSet()

    # Search using natural language — no need to know exact tool names
    tools = toolset.search_tools(
        "manage employee records",
        top_k=5,
        min_score=0.3,
    )

    print(f"Found {len(tools)} matching tools:")
    for tool in tools:
        print(f"  - {tool.name}: {tool.description[:80]}...")

    # The result is a standard Tools collection — convert to any framework format
    openai_tools = tools.to_openai()
    print(f"\nConverted to {len(openai_tools)} OpenAI function definitions")

    print()


def example_search_tools_with_connector():
    """Semantic search filtered by connector.

    Use the connector parameter to scope results to a specific provider,
    for example when you know the user works with BambooHR.
    """
    print("Example 2: search_tools() with connector filter\n")

    toolset = StackOneToolSet()

    # Search within a specific connector
    tools = toolset.search_tools(
        "create time off request",
        connector="bamboohr",
        top_k=3,
        min_score=0.3,
    )

    print(f"Found {len(tools)} BambooHR tools for 'create time off request':")
    for tool in tools:
        print(f"  - {tool.name}")

    print()


def example_search_action_names():
    """Lightweight search returning action names and scores without fetching tools.

    search_action_names() is useful when you want to inspect search results
    before committing to fetching full tool definitions — for example, to
    show the user a list of options.
    """
    print("Example 3: search_action_names() — lightweight inspection\n")

    toolset = StackOneToolSet()

    results = toolset.search_action_names(
        "time off requests",
        top_k=5,
        min_score=0.3,
    )

    print("Search results (action names + scores):")
    for r in results:
        print(f"  {r.action_name} ({r.connector_key}) — score: {r.similarity_score:.2f}")
        print(f"    {r.description[:80]}...")

    print()


def example_utility_tools_semantic():
    """Using utility tools with semantic search for agent loops.

    When building agent loops (search → select → execute), pass
    semantic_client to utility_tools() to upgrade tool_search from
    local BM25+TF-IDF to cloud-based semantic search.
    """
    print("Example 4: Utility tools with semantic search\n")

    toolset = StackOneToolSet()

    # Fetch tools for your accounts
    tools = toolset.fetch_tools()

    # Pass semantic_client to switch tool_search to semantic mode
    utility = tools.utility_tools(semantic_client=toolset.semantic_client)

    # tool_search now uses semantic search under the hood
    search_tool = utility.get_tool("tool_search")
    if search_tool:
        result = search_tool.call(query="onboard a new team member", limit=5)
        print("Semantic tool_search results:")
        for tool_info in result.get("tools", []):
            print(f"  - {tool_info['name']} (score: {tool_info['score']:.2f})")
            print(f"    {tool_info['description'][:80]}...")

    print()


def example_openai_agent_loop():
    """Complete agent loop: semantic search → OpenAI → execute.

    This demonstrates the full pattern for building an AI agent that
    discovers tools via semantic search and executes them via OpenAI.
    """
    print("Example 5: OpenAI agent loop with semantic search\n")

    try:
        from openai import OpenAI
    except ImportError:
        print("OpenAI library not installed. Install with: pip install openai")
        print()
        return

    if not os.getenv("OPENAI_API_KEY"):
        print("Set OPENAI_API_KEY to run this example")
        print()
        return

    client = OpenAI()
    toolset = StackOneToolSet()

    # Step 1: Discover relevant tools using semantic search
    tools = toolset.search_tools("list employees and their details", top_k=3)
    print(f"Discovered {len(tools)} tools via semantic search")
    for tool in tools:
        print(f"  - {tool.name}")

    # Step 2: Convert to OpenAI format and call the LLM
    openai_tools = tools.to_openai()

    messages = [
        {"role": "system", "content": "You are a helpful HR assistant."},
        {"role": "user", "content": "Can you list the first 5 employees?"},
    ]

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        tools=openai_tools,
        tool_choice="auto",
    )

    # Step 3: Execute the tool calls
    if response.choices[0].message.tool_calls:
        print("\nLLM chose to call:")
        for tool_call in response.choices[0].message.tool_calls:
            print(f"  - {tool_call.function.name}({tool_call.function.arguments})")

            tool = tools.get_tool(tool_call.function.name)
            if tool:
                result = tool.execute(tool_call.function.arguments)
                print(f"  Result keys: {list(result.keys()) if isinstance(result, dict) else type(result)}")
    else:
        print(f"\nLLM response: {response.choices[0].message.content}")

    print()


def example_langchain_semantic():
    """Semantic search with LangChain tools.

    search_tools() returns a Tools collection that converts directly
    to LangChain format — no extra steps needed.
    """
    print("Example 6: Semantic search with LangChain\n")

    try:
        from langchain_core.tools import BaseTool  # noqa: F401
    except ImportError:
        print("LangChain not installed. Install with: pip install langchain-core")
        print()
        return

    toolset = StackOneToolSet()

    # Semantic search → LangChain tools in two lines
    tools = toolset.search_tools("employee management", top_k=5)
    langchain_tools = tools.to_langchain()

    print(f"Created {len(langchain_tools)} LangChain tools from semantic search:")
    for tool in langchain_tools:
        print(f"  - {tool.name}: {tool.description[:80]}...")

    print()


def main():
    """Run all semantic search examples."""
    print("=" * 60)
    print("StackOne AI SDK — Semantic Search Examples")
    print("=" * 60)
    print()

    # Core patterns (require STACKONE_API_KEY)
    if not os.getenv("STACKONE_API_KEY"):
        print("Set STACKONE_API_KEY to run these examples")
        return

    example_search_tools()
    example_search_tools_with_connector()
    example_search_action_names()
    example_utility_tools_semantic()

    # Framework integration patterns
    example_openai_agent_loop()
    example_langchain_semantic()

    print("=" * 60)
    print("Examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
