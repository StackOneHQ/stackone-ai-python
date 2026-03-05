#!/usr/bin/env python
"""
Example demonstrating semantic search for AI-powered tool discovery.

Semantic search understands natural language intent and synonyms, so queries like
"book a meeting" or "cancel an event" resolve to the right StackOne actions —
unlike keyword matching which requires exact tool names.

This example uses a Calendly-linked account to demonstrate how semantic search
discovers scheduling, event, and organization management tools from natural
language queries.


How Semantic Search Works (Overview)
=====================================

The SDK provides three paths for semantic tool discovery, each with a different
trade-off between speed, filtering, and completeness:

1. search_tools(query)  — Full discovery (recommended for agent frameworks)

   This is the method you should use when integrating with OpenAI, LangChain,
   CrewAI, or any other agent framework.

   Recommended usage — pass ``connector`` to scope to a single provider:

       tools = toolset.search_tools("book a meeting", connector="calendly")

   This is faster and returns more relevant results than searching all
   connectors. When the target provider is known, always pass ``connector``.

   When ``connector`` is not specified, the SDK searches all connectors
   available in the user's linked accounts in parallel:

   a) Fetch tools from the user's linked accounts via MCP
   b) Extract available connectors (e.g. {bamboohr, calendly})
   c) Search each connector in parallel via the semantic search API
   d) Collect results, sort by relevance score
   e) If top_k was specified, keep only the top K results
   f) Match results back to the fetched tool definitions
   g) Return a Tools collection sorted by relevance score

   Key point: only the user's own connectors are searched — no wasted results
   from connectors the user doesn't have. When top_k is not specified, the
   backend decides how many results to return per connector. If the semantic
   API is unavailable, the SDK falls back to local BM25+TF-IDF search
   automatically.

2. search_action_names(query)  — Lightweight preview

   Queries the semantic API directly and returns metadata (name, connector,
   score, description) without fetching full tool definitions. Useful for
   inspecting results before committing to a full fetch. When account_ids are
   provided, each connector is searched in parallel (same as search_tools).

3. get_search_tool()  — Agent-loop pattern

   Returns a callable SearchTool that wraps search_tools(). Call it
   with a natural language query to get a Tools collection back.
   Designed for agent loops where the LLM decides what to search for.


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

from stackone_ai import StackOneToolSet

try:
    from dotenv import load_dotenv

    load_dotenv()
except ModuleNotFoundError:
    pass

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

    # Constructor-level config sets defaults; per-call params override them.
    # Here we set method="semantic" at the constructor level.
    toolset = StackOneToolSet(search={"method": "semantic"})

    query = "get user schedule"

    # --- top_k behavior ---
    # When top_k is NOT specified, the backend decides how many results to return.
    # When top_k IS specified, results are explicitly limited to that number.
    print(f'Searching for: "{query}" (no top_k — backend decides count)')
    results_default = toolset.search_action_names(query)
    print(f"  Backend returned {len(results_default)} results (its default)")
    print()

    print(f'Searching for: "{query}" (top_k=3 — explicitly limited)')
    results_limited = toolset.search_action_names(query, top_k=3)
    print(f"  Got exactly {len(results_limited)} results")
    print()

    # Show the limited results
    print(f"Top {len(results_limited)} matches from the full catalog:")
    for r in results_limited:
        print(f"  [{r.similarity_score:.2f}] {r.action_name} ({r.connector_key})")
        print(f"         {r.description}")
    print()

    # Show filtering effect when account_ids are available
    if _account_ids:
        print(f"Now filtering to your linked accounts ({', '.join(_account_ids)})...")
        print("  (Each connector is searched in parallel — only your connectors are queried)")
        filtered = toolset.search_action_names(query, account_ids=_account_ids, top_k=5)
        print(f"  Filtered to {len(filtered)} matches (only your connectors):")
        for r in filtered:
            print(f"    [{r.similarity_score:.2f}] {r.action_name} ({r.connector_key})")
    else:
        print("Tip: Set STACKONE_ACCOUNT_ID to see results filtered to your linked connectors.")

    print()


def example_search_tools():
    """High-level semantic search returning a Tools collection.

    search_tools() is the recommended way to use semantic search. It:
    1. Fetches tool definitions from your linked accounts via MCP
    2. Searches each of your connectors in parallel via the semantic search API
    3. Sorts results by relevance and matches back to tool definitions
    4. Returns a Tools collection ready for any framework (.to_openai(), .to_langchain(), etc.)

    Search config can be set at the constructor level:
        toolset = StackOneToolSet(search={"method": "semantic", "top_k": 5})
    Per-call parameters (e.g. top_k, search) override the constructor defaults.
    """
    print("=" * 60)
    print("Example 2: search_tools() — full tool discovery")
    print("=" * 60)
    print()

    # Constructor-level search config: always use semantic search with top_k=5
    toolset = StackOneToolSet(search={"method": "semantic", "top_k": 5})

    query = "cancel an event"
    print(f'Step 1: Searching for "{query}" via semantic search (constructor config)...')
    print()

    # top_k and method are already set via the constructor — no need to pass them here
    tools = toolset.search_tools(query, account_ids=_account_ids)

    connectors = tools.get_connectors()
    print(f"Found {len(tools)} tools from your linked account(s) ({', '.join(sorted(connectors))}):")
    for tool in tools:
        print(f"  - {tool.name}")
        print(f"    {tool.description}")
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


def example_search_tool_agent_loop():
    """Using get_search_tool() for agent loops.

    get_search_tool() returns a callable that wraps search_tools().
    Call it with a query to get a Tools collection back — designed
    for agent loops where the LLM decides what to search for.
    """
    print("=" * 60)
    print("Example 4: Search tool for agent loops")
    print("=" * 60)
    print()

    toolset = StackOneToolSet()

    print("Step 1: Fetching tools from your linked accounts via MCP...")
    all_tools = toolset.fetch_tools(account_ids=_account_ids)
    print(f"Loaded {len(all_tools)} tools.")
    print()

    print("Step 2: Getting a callable search tool...")
    search_tool = toolset.get_search_tool()

    query = "cancel an event or meeting"
    print()
    print(f'Step 3: Calling search_tool("{query}")...')
    print("  (Searches are scoped to your linked connectors)")
    print()
    tools = search_tool(query, top_k=5, account_ids=_account_ids)
    print(f"search_tool returned {len(tools)} tools:")
    for tool in tools:
        print(f"  - {tool.name}")
        print(f"    {tool.description}")

    print()


def example_openai_agent_loop():
    """Complete agent loop: semantic search -> LLM -> execute.

    This demonstrates the full pattern for building an AI agent that
    discovers tools via semantic search and executes them via an LLM.

    Supports both OpenAI and Google Gemini (via its OpenAI-compatible API).
    Set OPENAI_API_KEY for OpenAI, or GOOGLE_API_KEY for Gemini.
    """
    print("=" * 60)
    print("Example 5: LLM agent loop with semantic search")
    print("=" * 60)
    print()

    try:
        from openai import OpenAI
    except ImportError:
        print("Skipped: OpenAI library not installed. Install with: pip install openai")
        print()
        return

    # Support both OpenAI and Gemini (via OpenAI-compatible endpoint)
    openai_key = os.getenv("OPENAI_API_KEY")
    google_key = os.getenv("GOOGLE_API_KEY")

    if openai_key:
        client = OpenAI()
        model = "gpt-4o-mini"
        provider = "OpenAI"
    elif google_key:
        client = OpenAI(
            api_key=google_key,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        )
        model = "gemini-2.5-flash"
        provider = "Gemini"
    else:
        print("Skipped: Set OPENAI_API_KEY or GOOGLE_API_KEY to run this example.")
        print()
        return

    print(f"Using {provider} ({model})")
    print()

    toolset = StackOneToolSet()

    query = "list upcoming events"
    print(f'Step 1: Discovering tools for "{query}" via semantic search...')
    tools = toolset.search_tools(query, account_ids=_account_ids, top_k=3)
    print(f"Found {len(tools)} tools:")
    for tool in tools:
        print(f"  - {tool.name}")
    print()

    print(f"Step 2: Sending tools to {provider} as function definitions...")
    openai_tools = tools.to_openai()

    messages = [
        {"role": "system", "content": "You are a helpful scheduling assistant."},
        {"role": "user", "content": "Can you show me my upcoming events?"},
    ]

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        tools=openai_tools,
        tool_choice="auto",
    )

    if response.choices[0].message.tool_calls:
        print(f"Step 3: {provider} chose to call these tools:")
        for tool_call in response.choices[0].message.tool_calls:
            print(f"  - {tool_call.function.name}({tool_call.function.arguments})")

            tool = tools.get_tool(tool_call.function.name)
            if tool:
                result = tool.execute(tool_call.function.arguments)
                print(
                    f"    Response keys: {list(result.keys()) if isinstance(result, dict) else type(result)}"
                )
    else:
        print(f"{provider} responded with text: {response.choices[0].message.content}")

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
        print("  - Search tool for agent loops")
        print("  - OpenAI agent loop")
        print("  - LangChain integration")
        return

    example_search_tools()
    example_search_tools_with_connector()
    example_search_tool_agent_loop()

    # Framework integration patterns
    example_openai_agent_loop()
    example_langchain_semantic()

    print("############################################################")
    print("#   All examples completed!                                 #")
    print("############################################################")


if __name__ == "__main__":
    main()
