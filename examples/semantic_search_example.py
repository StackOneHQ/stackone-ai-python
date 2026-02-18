#!/usr/bin/env python
"""
Example demonstrating semantic search for AI-powered tool discovery.

Semantic search understands natural language intent and synonyms, so queries like
"book a meeting" or "cancel an event" resolve to the right StackOne actions —
unlike keyword matching which requires exact tool names.

This example uses a Calendly-linked account to demonstrate how semantic search
discovers scheduling, event, and organization management tools from natural
language queries.

This example is runnable with the following command:
```bash
uv run examples/semantic_search_example.py
```

Prerequisites:
- STACKONE_API_KEY environment variable set
- STACKONE_ACCOUNT_ID environment variable set (required for examples that fetch tools)
- At least one linked account in StackOne (this example uses Calendly)

Note: search_action_names() works with just STACKONE_API_KEY — no account ID needed.
"""

import logging
import os

from dotenv import load_dotenv

from stackone_ai import StackOneToolSet

load_dotenv()

# Show SDK warnings (e.g., semantic search fallback to local search)
logging.basicConfig(level=logging.WARNING)

# Read account IDs from environment — supports comma-separated values
_account_ids = [aid.strip() for aid in os.getenv("STACKONE_ACCOUNT_ID", "").split(",") if aid.strip()]


def example_search_action_names():
    """Lightweight search returning action names and scores without fetching tools.

    search_action_names() queries the semantic search API directly — it does NOT
    need account IDs or MCP. This makes it the simplest way to try semantic search.

    When called without account_ids, results come from the full StackOne catalog
    (all connectors). When called with account_ids, results are filtered to only
    connectors available in your linked accounts.
    """
    print("=" * 60)
    print("Example 1: search_action_names() — lightweight discovery")
    print("=" * 60)
    print()
    print("This searches the StackOne action catalog using semantic vectors.")
    print("No account ID needed — results come from all available connectors.")
    print()

    toolset = StackOneToolSet()

    query = "get user schedule"
    print(f'Searching for: "{query}"')
    print()

    results = toolset.search_action_names(query, top_k=5)

    print(f"Top {len(results)} matches from the full catalog:")
    for r in results:
        print(f"  [{r.similarity_score:.2f}] {r.action_name} ({r.connector_key})")
        print(f"         {r.description}")
    print()

    # Show filtering effect when account_ids are available
    if _account_ids:
        print(f"Now filtering to your linked accounts ({', '.join(_account_ids)})...")
        filtered = toolset.search_action_names(query, account_ids=_account_ids, top_k=5)
        print(f"Filtered to {len(filtered)} matches (only your connectors):")
        for r in filtered:
            print(f"  [{r.similarity_score:.2f}] {r.action_name} ({r.connector_key})")
    else:
        print("Tip: Set STACKONE_ACCOUNT_ID to see results filtered to your linked connectors.")

    print()


def example_search_tools():
    """High-level semantic search returning a Tools collection.

    search_tools() is the recommended way to use semantic search. It:
    1. Queries the semantic search API with your natural language query
    2. Fetches tool definitions from your linked accounts via MCP
    3. Matches semantic results to available tools (filtering out connectors you don't have)
    4. Returns a Tools collection ready for any framework (.to_openai(), .to_langchain(), etc.)
    """
    print("=" * 60)
    print("Example 2: search_tools() — full tool discovery")
    print("=" * 60)
    print()

    toolset = StackOneToolSet()

    query = "cancel an event"
    print(f'Step 1: Searching for "{query}" via semantic search...')
    print()

    tools = toolset.search_tools(query, account_ids=_account_ids, top_k=5)

    connectors = {t.name.split("_")[0] for t in tools}
    print(f"Found {len(tools)} tools from your linked account(s) ({', '.join(sorted(connectors))}):")
    for tool in tools:
        print(f"  - {tool.name}")
        print(f"    {tool.description}")
    print()

    # Show OpenAI conversion
    print("Step 2: Converting to OpenAI function-calling format...")
    openai_tools = tools.to_openai()
    print(f"Created {len(openai_tools)} OpenAI function definitions:")
    for fn in openai_tools:
        func = fn["function"]
        param_names = list(func["parameters"].get("properties", {}).keys())
        print(f"  - {func['name']}({', '.join(param_names[:3])}{'...' if len(param_names) > 3 else ''})")
    print()


def example_search_tools_with_connector():
    """Semantic search filtered by connector.

    Use the connector parameter to scope results to a specific provider,
    for example when you know the user works with Calendly.
    """
    print("=" * 60)
    print("Example 3: search_tools() with connector filter")
    print("=" * 60)
    print()

    toolset = StackOneToolSet()

    query = "book a meeting"
    connector = "calendly"
    print(f'Searching for "{query}" filtered to connector="{connector}"...')
    print()

    tools = toolset.search_tools(
        query,
        connector=connector,
        account_ids=_account_ids,
        top_k=3,
    )

    print(f"Found {len(tools)} {connector} tools:")
    for tool in tools:
        print(f"  - {tool.name}")
        print(f"    {tool.description}")
    print()


def example_utility_tools_semantic():
    """Using utility tools with semantic search for agent loops.

    When building agent loops (search -> select -> execute), pass
    semantic_client to utility_tools() to upgrade tool_search from
    local BM25+TF-IDF to cloud-based semantic search.

    Note: tool_search queries the full backend catalog (all connectors),
    not just the ones in your linked accounts.
    """
    print("=" * 60)
    print("Example 4: Utility tools with semantic search")
    print("=" * 60)
    print()

    toolset = StackOneToolSet()

    print("Step 1: Fetching tools from your linked accounts via MCP...")
    tools = toolset.fetch_tools(account_ids=_account_ids)
    print(f"Loaded {len(tools)} tools.")
    print()

    print("Step 2: Creating utility tools with semantic search enabled...")
    print("  Passing semantic_client upgrades tool_search from local keyword")
    print("  matching (BM25+TF-IDF) to cloud-based semantic vector search.")
    utility = tools.utility_tools(semantic_client=toolset.semantic_client)

    search_tool = utility.get_tool("tool_search")
    if search_tool:
        query = "cancel an event or meeting"
        print()
        print(f'Step 3: Calling tool_search with query="{query}"...')
        print("  (This searches the full StackOne catalog, not just your linked tools)")
        print()
        result = search_tool.call(query=query, limit=5)
        tools_data = result.get("tools", [])
        print(f"tool_search returned {len(tools_data)} results:")
        for tool_info in tools_data:
            print(f"  [{tool_info['score']:.2f}] {tool_info['name']}")
            print(f"         {tool_info['description']}")

    print()


def example_openai_agent_loop():
    """Complete agent loop: semantic search -> OpenAI -> execute.

    This demonstrates the full pattern for building an AI agent that
    discovers tools via semantic search and executes them via OpenAI.
    """
    print("=" * 60)
    print("Example 5: OpenAI agent loop with semantic search")
    print("=" * 60)
    print()

    try:
        from openai import OpenAI
    except ImportError:
        print("Skipped: OpenAI library not installed. Install with: pip install openai")
        print()
        return

    if not os.getenv("OPENAI_API_KEY"):
        print("Skipped: Set OPENAI_API_KEY to run this example.")
        print()
        return

    client = OpenAI()
    toolset = StackOneToolSet()

    query = "list upcoming events"
    print(f'Step 1: Discovering tools for "{query}" via semantic search...')
    tools = toolset.search_tools(query, account_ids=_account_ids, top_k=3)
    print(f"Found {len(tools)} tools:")
    for tool in tools:
        print(f"  - {tool.name}")
    print()

    print("Step 2: Sending tools to OpenAI as function definitions...")
    openai_tools = tools.to_openai()

    messages = [
        {"role": "system", "content": "You are a helpful scheduling assistant."},
        {"role": "user", "content": "Can you show me my upcoming events?"},
    ]

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        tools=openai_tools,
        tool_choice="auto",
    )

    if response.choices[0].message.tool_calls:
        print("Step 3: OpenAI chose to call these tools:")
        for tool_call in response.choices[0].message.tool_calls:
            print(f"  - {tool_call.function.name}({tool_call.function.arguments})")

            tool = tools.get_tool(tool_call.function.name)
            if tool:
                result = tool.execute(tool_call.function.arguments)
                print(f"    Response keys: {list(result.keys()) if isinstance(result, dict) else type(result)}")
    else:
        print(f"OpenAI responded with text: {response.choices[0].message.content}")

    print()


def example_langchain_semantic():
    """Semantic search with LangChain tools.

    search_tools() returns a Tools collection that converts directly
    to LangChain format — no extra steps needed.
    """
    print("=" * 60)
    print("Example 6: Semantic search with LangChain")
    print("=" * 60)
    print()

    try:
        from langchain_core.tools import BaseTool  # noqa: F401
    except ImportError:
        print("Skipped: LangChain not installed. Install with: pip install langchain-core")
        print()
        return

    toolset = StackOneToolSet()

    query = "remove a user from the team"
    print(f'Step 1: Searching for "{query}" via semantic search...')
    tools = toolset.search_tools(query, account_ids=_account_ids, top_k=5)
    print(f"Found {len(tools)} tools.")
    print()

    print("Step 2: Converting to LangChain tools...")
    langchain_tools = tools.to_langchain()

    print(f"Created {len(langchain_tools)} LangChain tools (ready for use with agents):")
    for tool in langchain_tools:
        print(f"  - {tool.name} (type: {type(tool).__name__})")
        print(f"    {tool.description}")

    print()


def main():
    """Run all semantic search examples."""
    print()
    print("############################################################")
    print("#   StackOne AI SDK — Semantic Search Examples              #")
    print("############################################################")
    print()

    if not os.getenv("STACKONE_API_KEY"):
        print("Set STACKONE_API_KEY to run these examples.")
        return

    # --- Examples that work without account IDs ---
    example_search_action_names()

    # --- Examples that require account IDs (MCP needs x-account-id) ---
    if not _account_ids:
        print("=" * 60)
        print("Remaining examples require STACKONE_ACCOUNT_ID")
        print("=" * 60)
        print()
        print("Set STACKONE_ACCOUNT_ID (comma-separated for multiple) to run")
        print("examples that fetch full tool definitions from your linked accounts:")
        print("  - search_tools() with natural language queries")
        print("  - search_tools() with connector filter")
        print("  - Utility tools with semantic search")
        print("  - OpenAI agent loop")
        print("  - LangChain integration")
        return

    example_search_tools()
    example_search_tools_with_connector()
    example_utility_tools_semantic()

    # Framework integration patterns
    example_openai_agent_loop()
    example_langchain_semantic()

    print("############################################################")
    print("#   All examples completed!                                 #")
    print("############################################################")


if __name__ == "__main__":
    main()
